# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Dependency-aware flow execution using structured concurrency primitives.

Provides clean dependency management and context inheritance for operation graphs,
using Events for synchronization and CapacityLimiter for concurrency control.
"""

import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import UUID

from anyio import get_cancelled_exc_class

from lionagi.ln import AlcallParams
from lionagi.ln.concurrency import CapacityLimiter, ConcurrencyEvent
from lionagi.models.note import Note
from lionagi.operations.node import Operation
from lionagi.protocols.generic.event import Event
from lionagi.protocols.types import ID, Condition, EventStatus
from lionagi.utils import to_dict

if TYPE_CHECKING:
    from lionagi.protocols.graph.graph import Graph
    from lionagi.session.session import Branch, Session


logger = logging.getLogger(__name__)

# Maximum concurrency when None is specified (effectively unlimited)
UNLIMITED_CONCURRENCY = int(os.environ.get("LIONAGI_MAX_CONCURRENCY", "10000"))


class DependencyAwareExecutor:
    """Executes operation graphs with dependency management and context inheritance."""

    def __init__(
        self,
        session: "Session",
        graph: "Graph",
        context: dict[str, Any] | None = None,
        max_concurrent: int = 5,
        verbose: bool = False,
        default_branch: "Branch" = None,
        alcall_params: AlcallParams | None = None,
    ):
        """Initialize the executor.

        Args:
            session: The session for branch management
            graph: The operation graph to execute
            context: Initial execution context
            max_concurrent: Maximum concurrent operations
            verbose: Enable verbose logging
            default_branch: Optional default branch for operations
        """
        self.session = session
        self.graph = graph
        # Note acts as a typed cognitive workspace for flow-level context that
        # accumulates across operations.  Callers pass a plain dict; we wrap it
        # so internal code can use Note's path-indexing and deep-update APIs.
        self.context: Note = Note(**(context or {}))
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self._alcall = alcall_params or AlcallParams()
        self._default_branch = default_branch
        self.on_progress = None  # callback(op_id, ref_id, status, elapsed_s)

        # Track results and completion
        self.results = {}
        self.completion_events = {}  # operation_id -> Event
        self.operation_branches = {}  # operation_id -> Branch
        self.failed_operations = set()
        self.skipped_operations = set()
        self.cancelled_operations = set()
        self._op_start_times = {}  # operation_id -> monotonic start time

        # Initialize completion events for all operations
        # and check for already completed operations
        for node in graph.internal_nodes.values():
            if isinstance(node, Operation):
                self.completion_events[node.id] = ConcurrencyEvent()

                # Resume mode: terminal operations are treated as already done
                # and their completion events must be set for dependents.
                if node.execution.status in Event._TERMINAL_STATUSES:
                    self.completion_events[node.id].set()
                    if node.execution.status == EventStatus.COMPLETED:
                        self.results[node.id] = node.response
                    elif node.execution.status == EventStatus.FAILED:
                        self.failed_operations.add(node.id)
                        self.results[node.id] = {"error": str(node.execution.error)}
                    elif node.execution.status == EventStatus.SKIPPED:
                        self.skipped_operations.add(node.id)
                    elif node.execution.status == EventStatus.CANCELLED:
                        self.cancelled_operations.add(node.id)

    async def execute(self) -> dict[str, Any]:
        """Execute the operation graph."""
        if not self.graph.is_acyclic():
            raise ValueError("Graph must be acyclic for flow execution")

        # Validate edge conditions before execution
        self._validate_edge_conditions()

        # Pre-allocate ALL branches upfront to avoid any locking during execution
        await self._preallocate_all_branches()

        # Create capacity limiter for concurrency control
        # None means no limit, use the configured unlimited value
        capacity = (
            self.max_concurrent
            if self.max_concurrent is not None
            else UNLIMITED_CONCURRENCY
        )
        limiter = CapacityLimiter(capacity)

        nodes = [
            n for n in self.graph.internal_nodes.values() if isinstance(n, Operation)
        ]
        await self._alcall(nodes, self._execute_operation, limiter=limiter)

        operations = [
            n for n in self.graph.internal_nodes.values() if isinstance(n, Operation)
        ]
        completed_ops = [
            op.id for op in operations if op.execution.status == EventStatus.COMPLETED
        ]
        failed_ops = [
            op.id for op in operations if op.execution.status == EventStatus.FAILED
        ]
        cancelled_ops = [
            op.id for op in operations if op.execution.status == EventStatus.CANCELLED
        ]

        result = {
            "completed_operations": completed_ops,
            "failed_operations": failed_ops,
            "operation_results": self.results,
            # Expose the plain dict so callers can use normal dict operations
            # (e.g. equality checks, key lookups) without Note wrapping.
            "final_context": self.context.content,
            "skipped_operations": list(self.skipped_operations),
            "cancelled_operations": cancelled_ops,
        }

        # Validate results before returning
        self._validate_execution_results(result)

        return result

    async def _preallocate_all_branches(self):
        """Pre-allocate ALL branches including for context inheritance to eliminate runtime locking."""
        operations_needing_branches = []

        # First pass: identify all operations that need branches
        for node in self.graph.internal_nodes.values():
            if not isinstance(node, Operation):
                continue

            if node.execution.status in Event._TERMINAL_STATUSES:
                continue

            # Operations pinned to a branch must resolve to a session branch.
            # Falling back silently makes graph intent hard to debug.
            if node.branch_id:
                try:
                    branch = self._get_session_branch(node.branch_id)
                    self.operation_branches[node.id] = branch
                except Exception:
                    raise ValueError(
                        f"Branch {node.branch_id} not found for operation {node.id}"
                    ) from None
                continue

            # Check if operation needs a new branch
            predecessors = self.graph.get_predecessors(node)
            if predecessors or node.metadata.get("inherit_context"):
                operations_needing_branches.append(node)

        if not operations_needing_branches:
            return

        # Create all branches in a single lock acquisition
        async with self.session.branches.async_lock:
            # For context inheritance, we need to create placeholder branches
            # that will be updated once dependencies complete
            for operation in operations_needing_branches:
                # Create a fresh branch from the executor default. A caller-
                # supplied default_branch must apply to preallocated child
                # branches too, not just root operations.
                base_branch = self._default_branch or self.session.default_branch
                branch_clone = base_branch.clone(sender=self.session.id)

                # Store in our operation branches map and include it through
                # Session so branch wiring matches normal session branches.
                self.operation_branches[operation.id] = branch_clone
                try:
                    self.session.include_branches(branch_clone)
                except Exception:
                    logger.debug(
                        "Skipping branch clone registration: include failed "
                        "(likely a mock object in test context)."
                    )

                # Mark branches that need context inheritance for later update
                if operation.metadata.get("inherit_context"):
                    branch_clone.metadata = branch_clone.metadata or {}
                    branch_clone.metadata["pending_context_inheritance"] = True
                    branch_clone.metadata["inherit_from_operation"] = (
                        operation.metadata.get("primary_dependency")
                    )

        if self.verbose:
            logger.debug("Pre-allocated %d branches", len(operations_needing_branches))

    async def _execute_operation(self, operation: Operation, limiter: CapacityLimiter):
        """Execute a single operation with dependency waiting.

        The state machine (idempotency, status transitions, error handling)
        lives in Event.invoke(). This method handles flow-level concerns:
        dependency waiting, branch assignment, result storage, edge conditions.
        """
        # Skip if operation is already in a terminal state
        if operation.execution.status in Event._TERMINAL_STATUSES:
            if self.verbose:
                logger.debug(
                    "Skipping %s operation: %s",
                    operation.execution.status.value,
                    str(operation.id)[:8],
                )
            # Ensure resumed terminal states are represented in the result maps.
            if operation.execution.status == EventStatus.COMPLETED:
                self.results[operation.id] = operation.response
            elif operation.execution.status == EventStatus.FAILED:
                self.failed_operations.add(operation.id)
                self.results.setdefault(
                    operation.id,
                    {"error": str(operation.execution.error)},
                )
            elif operation.execution.status == EventStatus.SKIPPED:
                self.skipped_operations.add(operation.id)
            elif operation.execution.status == EventStatus.CANCELLED:
                self.cancelled_operations.add(operation.id)
            # Signal completion for any waiting operations
            self.completion_events[operation.id].set()
            return

        try:
            # Check if this operation should be skipped due to edge conditions
            should_execute = await self._check_edge_conditions(operation)

            if not should_execute:
                # Mark as skipped
                operation.execution.status = EventStatus.SKIPPED
                self.skipped_operations.add(operation.id)

                if self.verbose:
                    logger.debug(
                        "Skipping operation due to edge conditions: %s",
                        str(operation.id)[:8],
                    )

                # Signal completion so dependent operations can proceed
                self.completion_events[operation.id].set()
                return

            # Wait for dependencies
            await self._wait_for_dependencies(operation)

            # Acquire capacity to limit concurrency
            async with limiter:
                original_parameters = (
                    deepcopy(operation.parameters)
                    if isinstance(operation.parameters, dict)
                    else operation.parameters
                )

                try:
                    # Prepare operation context just for this invocation. The
                    # original parameters are restored before returning so a
                    # graph spec does not accumulate stale context between runs.
                    self._prepare_operation(operation)

                    ref_id = operation.metadata.get(
                        "reference_id", str(operation.id)[:8]
                    )
                    branch = self.operation_branches.get(
                        operation.id,
                        self._default_branch or self.session.default_branch,
                    )
                    branch_name = getattr(branch, "name", None) or ref_id

                    import time as _time

                    self._op_start_times[operation.id] = _time.monotonic()

                    if self.on_progress:
                        self.on_progress(str(operation.id), branch_name, "started", 0)
                    if self.verbose:
                        logger.debug("Executing operation: %s", ref_id)

                    operation._branch = branch
                    await operation.invoke()

                    elapsed = _time.monotonic() - self._op_start_times.get(
                        operation.id, _time.monotonic()
                    )

                    # Store results based on status (set by Event.invoke()).
                    if operation.execution.status == EventStatus.COMPLETED:
                        self.results[operation.id] = operation.response

                        # Merge any context emitted by the operation into the
                        # flow-level Note workspace using deep merge to preserve
                        # nested keys rather than overwriting them wholesale.
                        if (
                            isinstance(operation.response, dict)
                            and "context" in operation.response
                        ):
                            from lionagi.libs.nested import deep_update

                            deep_update(
                                self.context.content, operation.response["context"]
                            )

                        if self.on_progress:
                            self.on_progress(
                                str(operation.id), branch_name, "completed", elapsed
                            )
                        if self.verbose:
                            logger.debug(
                                "Completed operation: %s (%.1fs)", ref_id, elapsed
                            )

                    elif operation.execution.status == EventStatus.FAILED:
                        self.failed_operations.add(operation.id)
                        self.results[operation.id] = {
                            "error": str(operation.execution.error)
                        }
                        if self.on_progress:
                            self.on_progress(
                                str(operation.id), branch_name, "failed", elapsed
                            )
                        if self.verbose:
                            logger.error(
                                "Operation %s failed (%.1fs): %s",
                                ref_id,
                                elapsed,
                                operation.execution.error,
                            )
                finally:
                    if isinstance(original_parameters, dict):
                        operation.parameters = original_parameters

        except (get_cancelled_exc_class(), KeyboardInterrupt, SystemExit):
            # Event.invoke() already set CANCELLED status — just propagate
            self.cancelled_operations.add(operation.id)
            self.completion_events[operation.id].set()
            raise

        except Exception as e:
            # Event.invoke() already set FAILED status and re-raised
            self.failed_operations.add(operation.id)
            if operation.id not in self.results:
                self.results[operation.id] = {"error": str(e)}

            if self.verbose:
                logger.error("Operation %s failed: %s", str(operation.id)[:8], e)

        finally:
            # Signal completion regardless of success/failure/skip
            self.completion_events[operation.id].set()

    async def _check_edge_conditions(self, operation: Operation) -> bool:
        """
        Check if operation should execute based on edge conditions.

        Returns True if at least one valid path exists to this operation,
        or if there are no incoming edges (head nodes).
        Returns False if all incoming edges have failed conditions.
        """
        # Get all incoming edges
        incoming_edges = [
            edge
            for edge in self.graph.internal_edges.values()
            if edge.tail == operation.id
        ]

        # If no incoming edges, this is a head node - always execute
        if not incoming_edges:
            return True

        # Check each incoming edge
        has_valid_path = False

        for edge in incoming_edges:
            # Wait for the head operation to complete first
            if edge.head in self.completion_events:
                await self.completion_events[edge.head].wait()

            # Check if the head operation was skipped
            if edge.head in self.skipped_operations:
                continue  # This path is not valid

            # Build context for edge condition evaluation
            result_value = self.results.get(edge.head)
            if result_value is not None and not isinstance(
                result_value, (str, int, float, bool)
            ):
                result_value = to_dict(result_value, recursive=True)

            # Edge condition `apply()` expects a plain dict with dict.get() semantics,
            # so expose the Note's content rather than the Note itself.
            ctx = {"result": result_value, "context": self.context.content}

            # Use edge.check_condition() which handles None conditions
            if await edge.check_condition(ctx):
                has_valid_path = True
                break  # At least one valid path found

        return has_valid_path

    async def _wait_for_dependencies(self, operation: Operation):
        """Wait for all dependencies to complete."""
        # Special handling for aggregations
        if operation.metadata.get("aggregation"):
            sources = operation.parameters.get("aggregation_sources", [])
            if self.verbose and sources:
                logger.debug(
                    "Aggregation %s waiting for %d sources",
                    str(operation.id)[:8],
                    len(sources),
                )

            for source_ref in sources:
                source_id = self._coerce_operation_id(source_ref)
                if source_id in self.completion_events:
                    await self.completion_events[source_id].wait()

        # Regular dependency checking
        predecessors = self.graph.get_predecessors(operation)
        for pred in predecessors:
            if self.verbose:
                logger.debug(
                    "Operation %s waiting for %s",
                    str(operation.id)[:8],
                    str(pred.id)[:8],
                )
            await self.completion_events[pred.id].wait()

    def _prepare_operation(self, operation: Operation):
        """Prepare operation with context and branch assignment."""
        # Update operation context with predecessors
        predecessors = self.graph.get_predecessors(operation)
        if predecessors:
            # Use a Note as a local workspace to accumulate predecessor results
            # before merging them into the operation's context parameter.
            pred_ctx = Note()
            for pred in predecessors:
                # Skip if predecessor was skipped
                if pred.id in self.skipped_operations:
                    continue

                if pred.id in self.results:
                    result = self.results[pred.id]
                    if result is not None and not isinstance(
                        result, (str, int, float, bool)
                    ):
                        result = to_dict(result, recursive=True)
                    pred_ctx[f"{str(pred.id)}_result"] = result

            pred_context = pred_ctx.content
            if "context" not in operation.parameters:
                operation.parameters["context"] = pred_context
            else:
                # Handle case where context might be a string
                existing_context = operation.parameters["context"]
                if isinstance(existing_context, dict):
                    existing_context.update(pred_context)
                else:
                    # If it's a string or other type, create a new dict
                    operation.parameters["context"] = {
                        "original_context": existing_context,
                        **pred_context,
                    }

        if operation.metadata.get("aggregation"):
            operation.parameters["aggregation_inputs"] = self._aggregation_inputs(
                operation
            )

        if operation.metadata.get("form"):
            self._prepare_form_inputs(operation, predecessors)

        # Add execution context from the flow-level Note workspace
        if self.context:
            if "context" not in operation.parameters:
                operation.parameters["context"] = self.context.content.copy()
            else:
                # Handle case where context might be a string
                existing_context = operation.parameters["context"]
                if isinstance(existing_context, dict):
                    existing_context.update(self.context.content)
                else:
                    # If it's a string or other type, create a new dict
                    operation.parameters["context"] = {
                        "original_context": existing_context,
                        **self.context.content,
                    }

        # Determine and assign branch
        branch = self._resolve_branch_for_operation(operation)
        self.operation_branches[operation.id] = branch

    def _resolve_branch_for_operation(self, operation: Operation) -> "Branch":
        """Resolve which branch an operation should use - all branches are pre-allocated."""
        # All branches should be pre-allocated
        if operation.id in self.operation_branches:
            branch = self.operation_branches[operation.id]

            # Handle deferred context inheritance
            if (
                hasattr(branch, "metadata")
                and branch.metadata
                and branch.metadata.get("pending_context_inheritance")
            ):
                primary_dep_id = branch.metadata.get("inherit_from_operation")
                if primary_dep_id and primary_dep_id in self.results:
                    # Find the primary dependency's branch
                    primary_branch = self.operation_branches.get(
                        primary_dep_id,
                        self._default_branch or self.session.default_branch,
                    )

                    # Copy the messages from primary branch to this branch
                    # This avoids creating a new branch and thus avoids locking
                    # Access messages through the MessageManager
                    if hasattr(branch, "_message_manager") and hasattr(
                        primary_branch, "_message_manager"
                    ):
                        branch._message_manager.messages.clear()
                        for msg in primary_branch._message_manager.messages:
                            if hasattr(msg, "clone"):
                                branch._message_manager.messages.append(msg.clone())
                            else:
                                branch._message_manager.messages.append(msg)

                    # Clear the pending flag
                    branch.metadata["pending_context_inheritance"] = False

                    if self.verbose:
                        logger.debug(
                            "Operation %s inherited context from %s",
                            str(operation.id)[:8],
                            str(primary_dep_id)[:8],
                        )

            return branch

        # Fallback to default branch (should not happen with proper pre-allocation)
        if self.verbose:
            logger.warning(
                "Operation %s using default branch (not pre-allocated)",
                str(operation.id)[:8],
            )

        if hasattr(self, "_default_branch") and self._default_branch:
            return self._default_branch
        return self.session.default_branch

    def _coerce_operation_id(self, value: Any) -> UUID | Any:
        """Return a UUID for operation-like references when possible."""
        try:
            return ID.get_id(value)
        except Exception:
            return value

    def _get_session_branch(self, branch_id: Any) -> "Branch":
        """Resolve a branch from the session, tolerating string/UUID key drift."""
        try:
            return self.session.branches[branch_id]
        except Exception:
            for branch in self.session.branches:
                if str(getattr(branch, "id", None)) == str(branch_id):
                    return branch
            raise

    def _aggregation_inputs(self, operation: Operation) -> list[dict[str, Any]]:
        """Build stable aggregation inputs from completed predecessor results."""
        sources = operation.parameters.get("aggregation_sources") or [
            pred.id for pred in self.graph.get_predecessors(operation)
        ]
        inputs = []
        for source_ref in sources:
            source_id = self._coerce_operation_id(source_ref)
            status = None
            if source_id in self.graph.internal_nodes:
                node = self.graph.internal_nodes[source_id]
                status = getattr(getattr(node, "execution", None), "status", None)
            result = self.results.get(source_id)
            if result is not None and not isinstance(result, (str, int, float, bool)):
                result = to_dict(result, recursive=True)
            inputs.append(
                {
                    "id": str(source_id),
                    "status": (
                        status.value if isinstance(status, EventStatus) else status
                    ),
                    "result": result,
                }
            )
        return inputs

    def _prepare_form_inputs(
        self,
        operation: Operation,
        predecessors: list[Operation],
    ) -> None:
        """Inject form inputs under ``form_inputs`` only.

        Form field names are user-controlled and can collide with branch
        operation parameters such as ``instruction`` or ``context``. Keep them
        namespaced so the operation contract stays unambiguous.
        """
        input_fields = operation.metadata.get("form_input_fields") or (
            operation.parameters.get("form_input_fields", [])
        )
        collect_fields = set(
            operation.metadata.get("form_collect_input_fields")
            or operation.parameters.get("form_collect_input_fields", [])
        )
        values = dict(operation.parameters.get("form_inputs") or {})

        def set_form_value(field: str, value: Any) -> None:
            if field not in collect_fields:
                values[field] = value
                return
            if field not in values or values[field] is None:
                values[field] = []
            elif not isinstance(values[field], list):
                values[field] = [values[field]]
            if isinstance(value, list):
                values[field].extend(value)
            else:
                values[field].append(value)

        for field in input_fields:
            if field in self.context.content and field not in values:
                if field in collect_fields:
                    value = self.context.content[field]
                    values[field] = value if isinstance(value, list) else [value]
                else:
                    values[field] = self.context.content[field]

        for pred in predecessors:
            if pred.id in self.skipped_operations:
                continue
            result = self.results.get(pred.id)
            if result is None:
                continue
            if not isinstance(result, (str, int, float, bool)):
                result = to_dict(result, recursive=True)
            if not isinstance(result, dict):
                continue

            output_fields = pred.metadata.get("form_output_fields") or (
                pred.parameters.get("form_output_fields", [])
                if isinstance(pred.parameters, dict)
                else []
            )
            for field in input_fields:
                if field in result and (not output_fields or field in output_fields):
                    set_form_value(field, result[field])

        operation.parameters["form_inputs"] = values

    def _validate_edge_conditions(self):
        """Validate that all edge conditions are properly configured."""
        for edge in self.graph.internal_edges.values():
            if edge.condition is not None:
                if not isinstance(edge.condition, Condition):
                    raise TypeError(
                        f"Edge {edge.id} has invalid condition type: {type(edge.condition)}. "
                        "Must be Condition or None."
                    )

                # Ensure condition has apply method
                if not hasattr(edge.condition, "apply"):
                    raise AttributeError(
                        f"Edge {edge.id} condition missing 'apply' method."
                    )

    def _validate_execution_results(self, results: dict[str, Any]):
        """Validate execution results for consistency."""
        completed = set(results.get("completed_operations", []))
        skipped = set(results.get("skipped_operations", []))
        failed = set(results.get("failed_operations", []))
        cancelled = set(results.get("cancelled_operations", []))

        terminal_sets = {
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "cancelled": cancelled,
        }

        # Check for operations in more than one terminal status list.
        overlaps = []
        labels = list(terminal_sets)
        for i, left_label in enumerate(labels):
            for right_label in labels[i + 1 :]:
                overlap = terminal_sets[left_label] & terminal_sets[right_label]
                if overlap:
                    overlaps.append((left_label, right_label, overlap))
        if overlaps:
            raise RuntimeError(
                f"Operations {overlaps} appear in multiple terminal status lists! "
                "This indicates a bug in flow result accounting."
            )

        # Verify skipped operations have proper status
        for node in self.graph.internal_nodes.values():
            if isinstance(node, Operation) and node.id in skipped:
                if node.execution.status != EventStatus.SKIPPED:
                    # Log warning but don't fail - status might not be perfectly synced
                    if self.verbose:
                        logger.warning(
                            "Skipped operation %s has status %s instead of SKIPPED",
                            node.id,
                            node.execution.status,
                        )


async def flow(
    session: "Session",
    graph: "Graph",
    *,
    branch: "Branch" = None,
    context: dict[str, Any] | None = None,
    parallel: bool = True,
    max_concurrent: int = None,
    verbose: bool = False,
    alcall_params: AlcallParams | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Execute a graph using structured concurrency primitives.

    This provides clean dependency management and context inheritance
    using Events and CapacityLimiter for proper coordination.

    Args:
        session: Session for branch management and multi-branch execution
        graph: The workflow graph containing Operation nodes
        branch: Optional specific branch to use for single-branch operations
        context: Initial context
        parallel: Whether to execute independent operations in parallel
        max_concurrent: Max concurrent operations (1 if not parallel)
        verbose: Enable verbose logging
        alcall_params: Parameters for async parallel call execution

    Returns:
        Execution results with completed operations and final context
    """

    # Handle concurrency limits
    if not parallel:
        max_concurrent = 1

    # Execute using the dependency-aware executor
    executor = DependencyAwareExecutor(
        session=session,
        graph=graph,
        context=context,
        max_concurrent=max_concurrent,
        verbose=verbose,
        default_branch=branch,
        alcall_params=alcall_params,
    )
    if on_progress is not None:
        executor.on_progress = on_progress

    return await executor.execute()


def cleanup_flow_results(
    result: dict[str, Any], keep_only: list[str] = None
) -> dict[str, Any]:
    """
    Clean up flow execution results to reduce memory usage.

    Args:
        result: Flow execution result dictionary
        keep_only: List of operation IDs to keep results for (optional)

    Returns:
        Modified result dictionary with reduced memory footprint
    """
    if not isinstance(result, dict) or "operation_results" not in result:
        return result

    terminal_lists = (
        "completed_operations",
        "failed_operations",
        "skipped_operations",
        "cancelled_operations",
    )

    # If keep_only is specified, keep matching operation payloads and terminal
    # status entries together so callers do not see contradictory summaries.
    if keep_only is not None:
        keep = set(keep_only)
        filtered_results = {
            op_id: res
            for op_id, res in result["operation_results"].items()
            if op_id in keep
        }
        result["operation_results"] = filtered_results
        for key in terminal_lists:
            result[key] = [op_id for op_id in result.get(key, []) if op_id in keep]
    else:
        # Clear all results to free memory
        result["operation_results"] = {}
        for key in terminal_lists:
            result[key] = []

    return result
