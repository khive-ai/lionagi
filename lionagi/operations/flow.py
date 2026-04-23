# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Dependency-aware flow execution using structured concurrency primitives.

Provides clean dependency management and context inheritance for operation graphs,
using Events for synchronization and CapacityLimiter for concurrency control.
"""

import logging
import os
from typing import TYPE_CHECKING, Any
from uuid import UUID

from anyio import get_cancelled_exc_class

from lionagi.ln import AlcallParams
from lionagi.ln.concurrency import CapacityLimiter, ConcurrencyEvent
from lionagi.operations.control import ControlDecision
from lionagi.operations.node import Operation
from lionagi.protocols.generic.event import Event
from lionagi.protocols.types import EventStatus
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
        self.context = context or {}
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self._alcall = alcall_params or AlcallParams()
        self._default_branch = default_branch
        self.on_progress = None  # callback(op_id, ref_id, status, elapsed_s)

        # Track results and completion
        self.results = {}
        self.completion_events = {}  # operation_id -> Event
        self.operation_branches = {}  # operation_id -> Branch
        self.skipped_operations = set()  # Track skipped operations
        self._op_start_times = {}  # operation_id -> monotonic start time
        self._limiter: CapacityLimiter | None = None

        # Control flow state
        self._halted = False
        self._halt_reason: str | None = None
        self._control_decisions: dict[Any, ControlDecision] = {}
        self._control_handlers: dict[str, Any] = (
            {}
        )  # custom control_type → async handler
        self._retry_counts: dict[Any, int] = {}  # operation_id -> retry count

        # Initialize completion events for all operations
        # and mark already-terminal operations as done.
        for node in graph.internal_nodes.values():
            if isinstance(node, Operation):
                self.completion_events[node.id] = ConcurrencyEvent()

                if node.execution.status in Event._TERMINAL_STATUSES:
                    self.completion_events[node.id].set()
                    if hasattr(node, "response") and node.response is not None:
                        self.results[node.id] = node.response

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
        self._limiter = limiter

        # Only dispatch operations that still need execution — terminal
        # ops from prior executor runs already have their completion events
        # set and results stored from __init__, so dependents can resolve
        # without re-dispatching the completed work.
        nodes = [
            n
            for n in self.graph.internal_nodes.values()
            if isinstance(n, Operation)
            and n.execution.status not in Event._TERMINAL_STATUSES
        ]
        if nodes:
            await self._alcall(nodes, self._execute_operation, limiter=limiter)

        # Return results - only include actually completed operations
        completed_ops = [
            op_id
            for op_id in self.results.keys()
            if op_id not in self.skipped_operations
        ]

        result = {
            "completed_operations": completed_ops,
            "operation_results": self.results,
            "final_context": self.context,
            "skipped_operations": list(self.skipped_operations),
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

            # Skip if operation already has a branch_id
            if node.branch_id:
                try:
                    # Ensure the branch exists in our local map
                    branch = self.session.branches[node.branch_id]
                    self.operation_branches[node.id] = branch
                except Exception:
                    logger.debug(
                        "Branch %s not found in session for node %s; "
                        "will be assigned during execution.",
                        node.branch_id,
                        node.id,
                    )
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
                # Create a fresh branch for now
                branch_clone = self.session.default_branch.clone(sender=self.session.id)

                # Store in our operation branches map
                self.operation_branches[operation.id] = branch_clone

                # Add to session branches collection directly
                # Check if this is a real branch (not a mock)
                try:
                    # Try to validate the ID
                    if hasattr(branch_clone, "id"):
                        branch_id = branch_clone.id
                        # Only add to collections if it's a valid ID
                        if isinstance(branch_id, (str, UUID)) or (
                            hasattr(branch_id, "__str__")
                            and not hasattr(branch_id, "_mock_name")
                        ):
                            self.session.branches.collections[branch_id] = branch_clone
                            self.session.branches.progression.append(branch_id)
                except Exception:
                    logger.debug(
                        "Skipping branch clone registration: validation failed "
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
            # Ensure results are available for dependencies
            if operation.id not in self.results and operation.response is not None:
                self.results[operation.id] = operation.response
            # Signal completion for any waiting operations
            self.completion_events[operation.id].set()
            return

        try:
            # Check halt flag — skip non-control ops if flow is halted
            if self._halted and not operation.is_control:
                operation.execution.status = EventStatus.SKIPPED
                self.skipped_operations.add(operation.id)
                if self.verbose:
                    logger.debug(
                        "Skipping %s: flow halted (%s)",
                        str(operation.id)[:8],
                        self._halt_reason,
                    )
                self.completion_events[operation.id].set()
                return

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

            # Re-check halt after awaiting — a control node may have set it
            # while this op was blocked on dependencies.
            if self._halted and not operation.is_control:
                operation.execution.status = EventStatus.SKIPPED
                self.skipped_operations.add(operation.id)
                if self.verbose:
                    logger.debug(
                        "Skipping %s: flow halted after dep wait (%s)",
                        str(operation.id)[:8],
                        self._halt_reason,
                    )
                self.completion_events[operation.id].set()
                return

            # Re-check route skips — route decisions may have added this op
            # to skipped_operations while it was waiting.
            if operation.id in self.skipped_operations:
                operation.execution.status = EventStatus.SKIPPED
                if self.verbose:
                    logger.debug(
                        "Skipping %s: routed away after dep wait",
                        str(operation.id)[:8],
                    )
                self.completion_events[operation.id].set()
                return

            # Control operations: evaluate and apply decision instead of LLM invoke
            if operation.is_control:
                decision = await self._evaluate_control(operation)
                self._control_decisions[operation.id] = decision
                await self._apply_control_decision(operation, decision)
                self.results[operation.id] = decision.model_dump()
                if self.verbose:
                    logger.debug(
                        "Control [%s] %s: %s (%s)",
                        operation.control_type,
                        str(operation.id)[:8],
                        decision.action,
                        decision.reason,
                    )
                self.completion_events[operation.id].set()
                return

            # Acquire capacity to limit concurrency
            async with limiter:
                # Prepare operation context
                self._prepare_operation(operation)

                ref_id = operation.metadata.get("reference_id", str(operation.id)[:8])
                branch = self.operation_branches.get(
                    operation.id, self.session.default_branch
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

                # Store results based on status (set by Event.invoke())
                if operation.execution.status == EventStatus.COMPLETED:
                    self.results[operation.id] = operation.response

                    # Update context if response contains context
                    if (
                        isinstance(operation.response, dict)
                        and "context" in operation.response
                    ):
                        self.context.update(operation.response["context"])

                    if self.on_progress:
                        self.on_progress(
                            str(operation.id), branch_name, "completed", elapsed
                        )
                    if self.verbose:
                        logger.debug("Completed operation: %s (%.1fs)", ref_id, elapsed)

                elif operation.execution.status == EventStatus.FAILED:
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

        except (get_cancelled_exc_class(), KeyboardInterrupt, SystemExit):
            # Event.invoke() already set CANCELLED status — just propagate
            self.completion_events[operation.id].set()
            raise

        except Exception as e:
            # Event.invoke() already set FAILED status and re-raised
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

            ctx = {"result": result_value, "context": self.context}

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

            # Wait for ALL sources (sources are now strings from builder.py)
            for source_id_str in sources:
                # Convert string back to UUID for lookup
                # Check all operations to find matching ID
                for op_id in self.completion_events.keys():
                    if str(op_id) == source_id_str:
                        await self.completion_events[op_id].wait()
                        break

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
            pred_context = {}
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
                    pred_context[f"{str(pred.id)}_result"] = result

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

        # Add execution context
        if self.context:
            if "context" not in operation.parameters:
                operation.parameters["context"] = self.context.copy()
            else:
                # Handle case where context might be a string
                existing_context = operation.parameters["context"]
                if isinstance(existing_context, dict):
                    existing_context.update(self.context)
                else:
                    # If it's a string or other type, create a new dict
                    operation.parameters["context"] = {
                        "original_context": existing_context,
                        **self.context,
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
                        primary_dep_id, self.session.default_branch
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

    def _validate_edge_conditions(self):
        """Validate that all edge conditions are properly configured."""
        for edge in self.graph.internal_edges.values():
            if edge.condition is not None:
                from lionagi.protocols._concepts import Condition

                if not isinstance(edge.condition, Condition):
                    raise TypeError(
                        f"Edge {edge.id} has invalid condition type: {type(edge.condition)}. "
                        "Must be a Condition subclass or None."
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

        # Check for operations in both lists
        overlap = completed & skipped
        if overlap:
            raise RuntimeError(
                f"Operations {overlap} appear in both completed and skipped lists! "
                "This indicates a bug in edge condition handling."
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

    # ── Control node evaluation ─────────────────────────────────

    async def _evaluate_control(self, op: Operation) -> ControlDecision:
        """Evaluate a control operation and return a decision."""
        ct = op.control_type
        policy = op.control_policy or {}

        handler = self._control_handlers.get(ct)
        if handler:
            return await handler(op, self)

        if ct == "quorum":
            return self._eval_quorum(op, policy)
        elif ct == "halt":
            return ControlDecision(
                action="halt", reason=policy.get("reason", "halt signal")
            )
        elif ct == "iterate":
            return await self._eval_iterate(op)
        else:
            return ControlDecision(
                action="proceed",
                reason=f"unknown control_type: {ct}, defaulting to proceed",
            )

    def _eval_quorum(self, op: Operation, policy: dict) -> ControlDecision:
        """Check if enough dependencies completed successfully."""
        predecessors = self.graph.get_predecessors(op)
        if not predecessors:
            return ControlDecision(
                action="proceed", reason="No deps, quorum trivially met"
            )

        total = len(predecessors)
        completed = sum(
            1
            for p in predecessors
            if p.id in self.results
            and p.id not in self.skipped_operations
            and not (isinstance(self.results[p.id], dict) and "error" in self.results[p.id])
        )
        ratio = completed / total if total > 0 else 1.0
        min_fraction = policy.get("min_fraction", 0.75)

        if ratio >= min_fraction:
            return ControlDecision(
                action="proceed",
                reason=f"Quorum met: {completed}/{total} = {ratio:.0%} >= {min_fraction:.0%}",
                metadata={"completed": completed, "total": total, "ratio": ratio},
            )
        return ControlDecision(
            action="halt",
            reason=f"Quorum not met: {completed}/{total} = {ratio:.0%} < {min_fraction:.0%}",
            metadata={"completed": completed, "total": total, "ratio": ratio},
        )

    async def _eval_iterate(self, op: Operation) -> ControlDecision:
        """LLM-backed control — invoke the op, interpret result as iterate/proceed."""
        branch = self.operation_branches.get(op.id, self.session.default_branch)
        op._branch = branch
        await op.invoke()

        response = op.response
        should_continue = False
        reason = str(response) if response else "no response"

        if isinstance(response, dict):
            should_continue = response.get("should_continue", False)
            reason = response.get("reason", reason)

        if should_continue:
            return ControlDecision(
                action="iterate",
                reason=reason,
                metadata={
                    "next_steps": (
                        response.get("next_steps")
                        if isinstance(response, dict)
                        else None
                    )
                },
            )
        return ControlDecision(action="proceed", reason=reason)

    async def _apply_control_decision(self, op: Operation, decision: ControlDecision):
        """Apply a control decision to the flow state."""
        if decision.action == "halt":
            self._halted = True
            self._halt_reason = decision.reason
        elif decision.action == "abort":
            self._halted = True
            self._halt_reason = f"ABORT: {decision.reason}"
        elif decision.action == "route":
            target_set = {str(t) for t in decision.targets}
            successors = self.graph.get_successors(op)
            for node in successors:
                if isinstance(node, Operation) and str(node.id) not in target_set:
                    if node.id not in self.results:
                        self.skipped_operations.add(node.id)
        elif decision.action == "retry":
            await self.retry_operations(
                decision.targets,
                max_retries=decision.metadata.get("max_retries", 3),
            )

    async def retry_operations(
        self,
        target_ids: list[str],
        *,
        max_retries: int = 3,
    ) -> list[str]:
        """Re-execute target operations via graph.replace_node().

        For each target: creates a fresh event via as_fresh_event(),
        replaces the old node in the graph (inheriting all edges),
        and executes the fresh node. Downstream nodes that depend
        on the target will see the retried output.

        Returns list of new operation IDs that were dispatched.
        """
        retried = []
        target_set = set(target_ids)

        for node in list(self.graph.internal_nodes.values()):
            if not isinstance(node, Operation):
                continue
            if str(node.id) not in target_set:
                continue

            count = self._retry_counts.get(node.id, 0)
            if count >= max_retries:
                if self.verbose:
                    logger.debug(
                        "Retry limit reached for %s (%d/%d)",
                        str(node.id)[:8], count, max_retries,
                    )
                continue

            fresh = node.as_fresh_event(copy_meta=True)

            if node.id in self.operation_branches:
                fresh._branch = self.operation_branches[node.id]
                self.operation_branches[fresh.id] = fresh._branch

            self.graph.replace_node(node.id, fresh)

            self.completion_events[fresh.id] = ConcurrencyEvent()
            self._retry_counts[fresh.id] = count + 1

            if self.verbose:
                logger.debug(
                    "Retry %s → %s (attempt %d/%d)",
                    str(node.id)[:8],
                    str(fresh.id)[:8],
                    count + 1,
                    max_retries,
                )

            retried.append(fresh)

        if retried and self._limiter:
            await self._alcall(
                retried, self._execute_operation, limiter=self._limiter
            )

        return [str(n.id) for n in retried]

    def register_control_handler(self, control_type: str, handler):
        """Register a custom control handler: async (op, executor) -> ControlDecision."""
        self._control_handlers[control_type] = handler


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

    # If keep_only is specified, only keep those results
    if keep_only is not None:
        filtered_results = {
            op_id: res
            for op_id, res in result["operation_results"].items()
            if op_id in keep_only
        }
        result["operation_results"] = filtered_results
        # Update completed_operations to match
        result["completed_operations"] = [
            op_id
            for op_id in result.get("completed_operations", [])
            if op_id in keep_only
        ]
    else:
        # Clear all results to free memory
        result["operation_results"] = {}
        result["completed_operations"] = []

    return result


async def flow_with_cleanup(
    session: "Session",
    graph: "Graph",
    context: dict[str, Any] | None = None,
    parallel: bool = True,
    max_concurrent: int = 5,
    verbose: bool = False,
    branch: "Branch" = None,
    alcall_params: AlcallParams | None = None,
    cleanup_results: bool = True,
    keep_only: list[str] = None,
) -> dict[str, Any]:
    """
    Execute flow with automatic cleanup to prevent memory accumulation.

    Args:
        session: Session instance for branch management
        graph: Operation graph to execute
        context: Initial context data
        parallel: Execute independent operations in parallel
        max_concurrent: Max concurrent operations (1 if not parallel)
        verbose: Enable verbose logging
        branch: Default branch for operations
        alcall_params: Parameters for async parallel call execution
        cleanup_results: Whether to clean up operation results after execution
        keep_only: List of operation IDs to keep results for (if cleanup_results=True)

    Returns:
        Execution results (potentially with cleaned up memory footprint)
    """
    # Execute the flow normally
    result = await flow(
        session=session,
        graph=graph,
        context=context,
        parallel=parallel,
        max_concurrent=max_concurrent,
        verbose=verbose,
        branch=branch,
        alcall_params=alcall_params,
    )

    # Clean up session memory
    if hasattr(session, "cleanup_memory"):
        session.cleanup_memory()

    # Clean up results if requested
    if cleanup_results:
        result = cleanup_flow_results(result, keep_only=keep_only)

    return result
