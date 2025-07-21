# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from typing import Any

from lionagi.operations.node import Operation
from lionagi.operations.utils import prepare_session
from lionagi.protocols.types import ID, Edge, Graph, Node
from lionagi.session.branch import Branch
from lionagi.session.session import Session
from lionagi.utils import to_dict


async def flow(
    branch: Branch,
    graph: Graph,
    *,
    context: dict[str, Any] | None = None,
    parallel: bool = True,
    max_concurrent: int = 5,
    verbose: bool = False,
    session: Session | None = None,
) -> dict[str, Any]:
    """
    Execute a graph-based workflow using the branch's operations.

    For simple graphs, executes directly on the branch.
    For parallel execution, uses session for coordination.

    Args:
        branch: The branch to execute operations on
        graph: The workflow graph containing Operation nodes
        context: Initial context
        parallel: Whether to execute independent operations in parallel
        max_concurrent: Max concurrent operations
        verbose: Enable verbose logging
        session: Optional session for multi-branch parallel execution

    Returns:
        Execution results with completed operations and final context
    """
    # Validate graph
    if not graph.is_acyclic():
        raise ValueError("Graph must be acyclic for flow execution")

    session, branch = prepare_session(session, branch)
    if not parallel or max_concurrent == 1:
        return await _execute_sequential(branch, graph, context, verbose)

    return await _execute_parallel(
        session, graph, context, max_concurrent, verbose
    )


async def _execute_sequential(
    branch: Branch, graph: Graph, context: dict[str, Any] | None, verbose: bool
) -> dict[str, Any]:
    """Execute graph sequentially on a single branch."""
    completed = []
    results = {}
    execution_context = context or {}

    # Get execution order (topological sort)
    execution_order = _topological_sort(graph)

    for node_id in execution_order:
        node = graph.internal_nodes[node_id]

        if not isinstance(node, Operation):
            continue

        # Check dependencies using set for fast lookup
        completed_set = set(completed)

        # Check if dependencies and conditions are satisfied
        if not await _dependencies_satisfied_async(
            node, graph, completed_set, results, execution_context
        ):
            continue

        predecessors = graph.get_predecessors(node)

        # Update operation context with predecessors
        if predecessors:
            pred_context = {}
            for pred in predecessors:
                if pred.id in results:
                    result = results[pred.id]
                # Use to_dict for proper serialization of complex types only
                if result is not None and not isinstance(
                    result, (str, int, float, bool)
                ):
                    result = to_dict(result, recursive=True)
                pred_context[f"{pred.id}_result"] = result

            if "context" not in node.parameters:
                node.parameters["context"] = pred_context
            else:
                node.parameters["context"].update(pred_context)

        # Add execution context
        if execution_context:
            if "context" not in node.parameters:
                node.parameters["context"] = execution_context.copy()
            else:
                node.parameters["context"].update(execution_context)

        # Execute operation
        if verbose:
            print(f"Executing operation: {node.id}")

        await node.invoke(branch)

        completed.append(node.id)
        results[node.id] = node.response

        # Update execution context
        if isinstance(node.response, dict) and "context" in node.response:
            execution_context.update(node.response["context"])

    return {
        "completed_operations": completed,
        "operation_results": results,
        "final_context": execution_context,
    }


async def _execute_parallel(
    session: Session,
    graph: Graph,
    context: dict[str, Any] | None,
    max_concurrent: int,
    verbose: bool,
) -> dict[str, Any]:
    """Execute graph in parallel using multiple branches."""
    results = {}
    execution_context = context or {}
    completed = []  # Track completed operations

    # Get operation nodes in topological order
    operation_nodes = []
    execution_order = _topological_sort(graph)
    for node_id in execution_order:
        node = graph.internal_nodes.get(node_id)
        if isinstance(node, Operation):
            operation_nodes.append(node)

    # Use session branches context manager for safe parallel execution
    async with session.branches:
        # Create a pool of worker branches
        worker_branches = []
        for i in range(min(max_concurrent, len(operation_nodes))):
            if i == 0:
                worker_branches.append(session.default_branch)
            else:
                worker_branches.append(session.split(session.default_branch))

        # Process nodes in dependency order
        remaining_nodes = {node.id for node in operation_nodes}
        executing_tasks: dict[ID[Operation], asyncio.Task] = {}
        blocked_nodes = set()  # Nodes that have been checked and found blocked

        max_iterations = 1000  # Prevent infinite loops
        iteration = 0

        while (
            remaining_nodes or executing_tasks
        ) and iteration < max_iterations:
            iteration += 1

            # Check for completed tasks
            completed_in_round = []
            for node_id, task in list(executing_tasks.items()):
                if task.done():
                    try:
                        result = await task
                        results[node_id] = result
                        completed.append(node_id)
                        completed_in_round.append(node_id)
                        if verbose:
                            print(f"Completed operation: {node_id}")
                    except Exception as e:
                        if verbose:
                            print(f"Operation {node_id} failed: {e}")
                        results[node_id] = {"error": str(e)}
                        completed.append(node_id)
                        completed_in_round.append(node_id)
                    finally:
                        del executing_tasks[node_id]

            # Remove completed from remaining
            remaining_nodes -= set(completed_in_round)

            # If new completions, clear blocked nodes to re-check
            if completed_in_round:
                blocked_nodes.clear()

            # Find nodes ready to execute (skip already blocked nodes)
            ready_nodes = []
            completed_set = set(completed)
            newly_blocked = []

            for node in operation_nodes:
                if (
                    node.id in remaining_nodes
                    and node.id not in executing_tasks
                    and node.id not in blocked_nodes
                    and len(executing_tasks) < max_concurrent
                ):
                    if await _dependencies_satisfied_async(
                        node, graph, completed_set, results, execution_context
                    ):
                        ready_nodes.append(node)
                    else:
                        newly_blocked.append(node.id)

            # Update blocked nodes
            blocked_nodes.update(newly_blocked)

            # If no ready nodes but we have remaining and no executing tasks, we're stuck
            if not ready_nodes and remaining_nodes and not executing_tasks:
                if verbose:
                    print(
                        f"Deadlock detected: {len(remaining_nodes)} nodes cannot execute"
                    )
                    remaining_node_names = [
                        n.operation
                        for n in operation_nodes
                        if n.id in remaining_nodes
                    ]
                    print(f"Remaining operations: {remaining_node_names}")
                # Mark remaining nodes as failed
                for node in operation_nodes:
                    if node.id in remaining_nodes:
                        results[node.id] = {
                            "error": "Blocked by unsatisfied conditions"
                        }
                        completed.append(node.id)
                break

            # Start execution for ready nodes
            started_count = 0
            for node in ready_nodes:
                if len(executing_tasks) >= max_concurrent:
                    break

                # Get an available branch (round-robin)
                branch_idx = len(executing_tasks) % len(worker_branches)
                node_branch = worker_branches[branch_idx]

                # Check if node specifies a branch
                branch_id = node.parameters.get("branch_id")
                if branch_id:
                    try:
                        node_branch = session.branches[branch_id]
                    except:
                        pass  # Use the selected worker branch

                # Create task for this node
                task = asyncio.create_task(
                    _execute_node_async(
                        node,
                        node_branch,
                        graph,
                        results,
                        execution_context,
                        verbose,
                    )
                )
                executing_tasks[node.id] = task
                started_count += 1

                if verbose:
                    branch_name = (
                        getattr(node_branch, "name", None) or node_branch.id
                    )
                    print(
                        f"Started operation {node.id} on branch: {branch_name}"
                    )

            # If we started new tasks or have executing tasks, wait for some to complete
            if started_count > 0 or executing_tasks:
                # Wait for at least one task to complete before next iteration
                if executing_tasks:
                    done, pending = await asyncio.wait(
                        executing_tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await asyncio.sleep(0.01)
            elif not remaining_nodes:
                # All done
                break

        if iteration >= max_iterations:
            raise RuntimeError(
                f"Flow execution exceeded maximum iterations ({max_iterations})"
            )

    return {
        "completed_operations": completed,
        "operation_results": results,
        "final_context": execution_context,
    }


async def _execute_node_async(
    node: Operation,
    branch: Branch,
    graph: Graph,
    results: dict[str, Any],
    execution_context: dict[str, Any],
    verbose: bool,
) -> Any:
    """Execute a single node asynchronously."""
    # Update operation context with predecessors
    predecessors = graph.get_predecessors(node)
    if predecessors:
        pred_context = {}
        for pred in predecessors:
            if pred.id in results:
                result = results[pred.id]
                # Use to_dict for proper serialization of complex types only
                if result is not None and not isinstance(
                    result, (str, int, float, bool)
                ):
                    result = to_dict(result, recursive=True)
                pred_context[f"{pred.id}_result"] = result

        if "context" not in node.parameters:
            node.parameters["context"] = pred_context
        else:
            node.parameters["context"].update(pred_context)

    # Add execution context
    if execution_context:
        if "context" not in node.parameters:
            node.parameters["context"] = execution_context.copy()
        else:
            node.parameters["context"].update(execution_context)

    # Execute the operation
    await node.invoke(branch)
    result = node.response

    # Update execution context if needed
    if isinstance(result, dict) and "context" in result:
        execution_context.update(result["context"])

    return result


def _topological_sort(graph: Graph) -> list[str]:
    """Get topological ordering of graph nodes."""
    visited = set()
    stack = []

    def visit(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)

        successors = graph.get_successors(graph.internal_nodes[node_id])
        for successor in successors:
            visit(successor.id)

        stack.append(node_id)

    for node in graph.internal_nodes:
        if node.id not in visited:
            visit(node.id)

    return stack[::-1]


async def _dependencies_satisfied_async(
    node: Node,
    graph: Graph,
    completed: set[str],
    results: dict[str, Any],
    execution_context: dict[str, Any] | None = None,
) -> bool:
    """Check if node dependencies are satisfied and edge conditions pass."""
    # Get all incoming edges to this node
    incoming_edges: list[Edge] = []
    for edge in graph.internal_edges:
        if edge.tail == node.id:
            incoming_edges.append(edge)

    # If no incoming edges, node can execute
    if not incoming_edges:
        return True

    # Check each incoming edge
    at_least_one_satisfied = False
    for edge in incoming_edges:
        # Check if predecessor is completed
        if edge.head not in completed:
            continue

        # Predecessor is completed
        if edge.condition:
            # Evaluate condition
            # Get the result - don't use to_dict if it's already a simple type
            result_value = results.get(edge.head)
            if result_value is not None and not isinstance(
                result_value, (str, int, float, bool)
            ):
                result_value = to_dict(result_value, recursive=True)

            ctx = {"result": result_value, "context": execution_context or {}}
            with contextlib.suppress(Exception):
                if await edge.condition.apply(ctx):
                    at_least_one_satisfied = True
        else:
            # No condition, edge is satisfied
            at_least_one_satisfied = True

    return at_least_one_satisfied
