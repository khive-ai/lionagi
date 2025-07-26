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

from lionagi.libs.concurrency.primitives import CapacityLimiter
from lionagi.libs.concurrency.primitives import Event as ConcurrencyEvent
from lionagi.libs.concurrency.task import create_task_group
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
        self._default_branch = default_branch

        # Track results and completion
        self.results = {}
        self.completion_events = {}  # operation_id -> Event
        self.operation_branches = {}  # operation_id -> Branch

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

        # Execute all operations using structured concurrency
        async with create_task_group() as tg:
            for node in self.graph.internal_nodes.values():
                if isinstance(node, Operation):
                    await tg.start_soon(self._execute_operation, node, limiter)

        # Return results
        return {
            "completed_operations": list(self.results.keys()),
            "operation_results": self.results,
            "final_context": self.context,
        }

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
            # Signal completion regardless of success/failure
            self.completion_events[operation.id].set()

    async def _wait_for_dependencies(self, operation: Operation):
        """Wait for all dependencies to complete."""
        # Special handling for aggregations
        if operation.metadata.get("aggregation"):
            sources = operation.parameters.get("aggregation_sources", [])
            if self.verbose and sources:
                print(
                    f"Aggregation {str(operation.id)[:8]} waiting for {len(sources)} sources"
                )

            # Wait for ALL sources
            for source_id in sources:
                if source_id in self.completion_events:
                    await self.completion_events[source_id].wait()

        # Regular dependency checking
        predecessors = self.graph.get_predecessors(operation)
        for pred in predecessors:
            if self.verbose:
                print(
                    f"Operation {str(operation.id)[:8]} waiting for {str(pred.id)[:8]}"
                )
            await self.completion_events[pred.id].wait()

        # Check edge conditions
        incoming_edges = [
            edge
            for edge in self.graph.internal_edges.values()
            if edge.tail == operation.id
        ]

        for edge in incoming_edges:
            # Wait for head to complete
            if edge.head in self.completion_events:
                await self.completion_events[edge.head].wait()

            # Evaluate edge condition
            if edge.condition is not None:
                result_value = self.results.get(edge.head)
                if result_value is not None and not isinstance(
                    result_value, (str, int, float, bool)
                ):
                    result_value = to_dict(result_value, recursive=True)

                ctx = {"result": result_value, "context": self.context}
                if not await edge.condition.apply(ctx):
                    raise ValueError(
                        f"Edge condition not satisfied for {str(operation.id)[:8]}"
                    )

    def _prepare_operation(self, operation: Operation):
        """Prepare operation with context and branch assignment."""
        # Update operation context with predecessors
        predecessors = self.graph.get_predecessors(operation)
        if predecessors:
            pred_context = {}
            for pred in predecessors:
                if pred.id in self.results:
                    result = self.results[pred.id]
                    if result is not None and not isinstance(
                        result, (str, int, float, bool)
                    ):
                        result = to_dict(result, recursive=True)
                    pred_context[f"{pred.id}_result"] = result

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
                            branch._message_manager.pile.append(msg.clone())

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


async def flow(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | None = None,
    context: dict[str, Any] | None = None,
    parallel: bool = True,
    max_concurrent: int = None,
    verbose: bool = False,
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

    Returns:
        Execution results with completed operations and final context
    """

    # Handle concurrency limits
    if not parallel:
        max_concurrent = 1  # Force sequential execution
    # If max_concurrent is None, it means no limit

    # Execute using the dependency-aware executor
    executor = DependencyAwareExecutor(
        session=session,
        graph=graph,
        context=context,
        max_concurrent=max_concurrent,
        verbose=verbose,
        default_branch=branch,
    )

    return await executor.execute()
