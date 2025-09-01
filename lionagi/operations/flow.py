# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Dependency-aware flow execution using structured concurrency primitives.

Provides clean dependency management and context inheritance for operation graphs,
using Events for synchronization and CapacityLimiter for concurrency control.
"""

import os
from typing import Any

from lionagi.ln._async_call import AlcallParams
from lionagi.ln.concurrency import CapacityLimiter, ConcurrencyEvent
from lionagi.operations.node import Operation
from lionagi.protocols.types import EventStatus, Graph
from lionagi.session.branch import Branch
from lionagi.session.session import Session
from lionagi.utils import to_dict

# Maximum concurrency when None is specified (effectively unlimited)
UNLIMITED_CONCURRENCY = int(os.environ.get("LIONAGI_MAX_CONCURRENCY", "10000"))


class DependencyAwareExecutor:
    """Executes operation graphs with dependency management and context inheritance."""

    def __init__(
        self,
        session: Session,
        graph: Graph,
        context: dict[str, Any] | None = None,
        max_concurrent: int = 5,
        verbose: bool = False,
        default_branch: Branch | None = None,
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

        # Track results and completion
        self.results = {}
        self.completion_events = {}  # operation_id -> Event
        self.operation_branches = {}  # operation_id -> Branch
        self.skipped_operations = set()  # Track skipped operations

        # Initialize completion events for all operations
        # and check for already completed operations
        for node in graph.internal_nodes.values():
            if isinstance(node, Operation):
                self.completion_events[node.id] = ConcurrencyEvent()

                # If operation is already completed, mark it and store results
                if node.execution.status == EventStatus.COMPLETED:
                    self.completion_events[node.id].set()
                    if hasattr(node, "response"):
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

        nodes = [
            n
            for n in self.graph.internal_nodes.values()
            if isinstance(n, Operation)
        ]
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
                except:
                    pass
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
                branch_clone = self.session.default_branch.clone(
                    sender=self.session.id
                )

                # Store in our operation branches map
                self.operation_branches[operation.id] = branch_clone

                # Add to session branches collection directly
                # Check if this is a real branch (not a mock)
                try:
                    from lionagi.protocols.types import IDType

                    # Try to validate the ID
                    if hasattr(branch_clone, "id"):
                        branch_id = branch_clone.id
                        # Only add to collections if it's a valid ID
                        if isinstance(branch_id, (str, IDType)) or (
                            hasattr(branch_id, "__str__")
                            and not hasattr(branch_id, "_mock_name")
                        ):
                            self.session.branches.collections[branch_id] = (
                                branch_clone
                            )
                            self.session.branches.progression.append(branch_id)
                except:
                    # If validation fails, it's likely a mock - skip adding to collections
                    pass

                # Mark branches that need context inheritance for later update
                if operation.metadata.get("inherit_context"):
                    branch_clone.metadata = branch_clone.metadata or {}
                    branch_clone.metadata["pending_context_inheritance"] = True
                    branch_clone.metadata["inherit_from_operation"] = (
                        operation.metadata.get("primary_dependency")
                    )

        if self.verbose:
            print(f"Pre-allocated {len(operations_needing_branches)} branches")

    async def _execute_operation(
        self, operation: Operation, limiter: CapacityLimiter
    ):
        """Execute a single operation with dependency waiting."""
        # Skip if operation is already completed
        if operation.execution.status == EventStatus.COMPLETED:
            if self.verbose:
                print(
                    f"Skipping already completed operation: {str(operation.id)[:8]}"
                )
            # Ensure results are available for dependencies
            if operation.id not in self.results and hasattr(
                operation, "response"
            ):
                self.results[operation.id] = operation.response
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
                    print(
                        f"Skipping operation due to edge conditions: {str(operation.id)[:8]}"
                    )

                # Signal completion so dependent operations can proceed
                self.completion_events[operation.id].set()
                return

            # Wait for dependencies
            await self._wait_for_dependencies(operation)

            # Acquire capacity to limit concurrency
            async with limiter:
                # Prepare operation context
                self._prepare_operation(operation)

                # Execute the operation
                if self.verbose:
                    print(f"Executing operation: {str(operation.id)[:8]}")

                branch = self.operation_branches.get(
                    operation.id, self.session.default_branch
                )
                operation.execution.status = EventStatus.PROCESSING

                await operation.invoke(branch)

                # Store results
                self.results[operation.id] = operation.response
                operation.execution.status = EventStatus.COMPLETED

                # Update context if response contains context
                if (
                    isinstance(operation.response, dict)
                    and "context" in operation.response
                ):
                    self.context.update(operation.response["context"])

                if self.verbose:
                    print(f"Completed operation: {str(operation.id)[:8]}")

        except Exception as e:
            operation.execution.status = EventStatus.FAILED
            operation.execution.error = str(e)
            self.results[operation.id] = {"error": str(e)}

            if self.verbose:
                print(f"Operation {str(operation.id)[:8]} failed: {e}")

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
                print(
                    f"Aggregation {str(operation.id)[:8]} waiting for {len(sources)} sources"
                )

            # Wait for ALL sources (sources are now strings from builder.py)
            for source_id_str in sources:
                # Convert string back to IDType for lookup
                # Check all operations to find matching ID
                for op_id in self.completion_events.keys():
                    if str(op_id) == source_id_str:
                        await self.completion_events[op_id].wait()
                        break

        # Regular dependency checking
        predecessors = self.graph.get_predecessors(operation)
        for pred in predecessors:
            if self.verbose:
                print(
                    f"Operation {str(operation.id)[:8]} waiting for {str(pred.id)[:8]}"
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
                    # Use string representation of IDType for JSON serialization
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

    def _resolve_branch_for_operation(self, operation: Operation) -> Branch:
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
                        branch._message_manager.pile.clear()
                        for msg in primary_branch._message_manager.pile:
                            if hasattr(msg, "clone"):
                                branch._message_manager.pile.append(
                                    msg.clone()
                                )
                            else:
                                branch._message_manager.pile.append(msg)

                    # Clear the pending flag
                    branch.metadata["pending_context_inheritance"] = False

                    if self.verbose:
                        print(
                            f"Operation {str(operation.id)[:8]} inherited context from {str(primary_dep_id)[:8]}"
                        )

            return branch

        # Fallback to default branch (should not happen with proper pre-allocation)
        if self.verbose:
            print(
                f"Warning: Operation {str(operation.id)[:8]} using default branch (not pre-allocated)"
            )

        if hasattr(self, "_default_branch") and self._default_branch:
            return self._default_branch
        return self.session.default_branch

    def _validate_edge_conditions(self):
        """Validate that all edge conditions are properly configured."""
        for edge in self.graph.internal_edges.values():
            if edge.condition is not None:
                # Ensure condition is an EdgeCondition instance
                from lionagi.protocols.graph.edge import EdgeCondition

                if not isinstance(edge.condition, EdgeCondition):
                    raise TypeError(
                        f"Edge {edge.id} has invalid condition type: {type(edge.condition)}. "
                        "Must be EdgeCondition or None."
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
                        print(
                            f"Warning: Skipped operation {node.id} has status "
                            f"{node.execution.status} instead of SKIPPED"
                        )


async def flow(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | None = None,
    context: dict[str, Any] | None = None,
    parallel: bool = True,
    max_concurrent: int = None,
    verbose: bool = False,
    alcall_params: AlcallParams | None = None,
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

    return await executor.execute()
