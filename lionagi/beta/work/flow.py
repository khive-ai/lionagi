# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Compile work operation graphs to core OpGraph and execute with Runner."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from lionagi.beta.core.graph import OpGraph, OpNode
from lionagi.beta.core.morphism import MorphismAdapter
from lionagi.beta.core.runner import Runner
from lionagi.beta.core.types import Principal
from lionagi.beta.resource.graph import Graph
from lionagi.protocols.generic.event import EventStatus

from .control import ControlDecision
from .node import Operation

if TYPE_CHECKING:
    from lionagi.beta.session.session import Branch, OperationDecl, Session

logger = logging.getLogger(__name__)

__all__ = ("OperationResult", "compile_flow_to_graph", "flow", "flow_stream")


@dataclass
class OperationResult:
    """Single operation result from a streaming flow execution."""

    name: str
    result: Any
    error: Exception | None = None
    completed: int = 0
    total: int = 0

    @property
    def success(self) -> bool:
        return self.error is None


def _operation_name(op: Operation) -> str:
    return str(op.metadata.get("name", op.id))


def _unwrap_result(result: dict[str, Any]) -> Any:
    if set(result.keys()) == {"result"}:
        return result["result"]
    return result


def _resolve_operation_branch(
    session: Session,
    branch_spec: Any,
    default_branch: Branch | None,
) -> Branch | None:
    if branch_spec is None:
        return default_branch
    if hasattr(branch_spec, "id") and hasattr(branch_spec, "order"):
        return branch_spec
    try:
        return session.get_branch(branch_spec)
    except Exception as exc:
        logger.debug("Branch '%s' not found, using default: %s", branch_spec, exc)
        return default_branch


def _default_branch(
    session: Session, branch: Branch | UUID | str | None
) -> Branch | None:
    if branch is not None:
        return session.get_branch(branch)
    return getattr(session, "default_branch", None)


def _principal_for_flow(
    default_branch: Branch | None,
    operation_branches: dict[UUID, Branch | None],
    context: dict[str, Any] | None,
) -> Principal:
    branch = default_branch or next(
        (b for b in operation_branches.values() if b is not None),
        None,
    )
    principal = (
        branch.principal.model_copy(deep=True)
        if branch is not None
        else Principal(name="flow")
    )
    if context:
        principal.ctx.update(context)
    return principal


def _runner_for_session(session: Session, max_concurrent: int | None) -> Runner:
    if max_concurrent is None and hasattr(session, "_get_runner"):
        return session._get_runner()
    return Runner(max_concurrent=max_concurrent)


def _control_morphism(op: Operation) -> MorphismAdapter:
    async def _apply(_br: Principal, **_kw: Any) -> dict[str, Any]:
        policy = op.control_policy or {}
        control_type = op.control_type
        if control_type in {"halt", "abort"}:
            return ControlDecision(
                action=control_type,
                reason=policy.get("reason", "halt signal"),
                metadata=policy.get("metadata", {}),
            ).model_dump()
        if control_type == "retry":
            return ControlDecision(
                action="retry",
                targets=policy.get("targets", []),
                reason=policy.get("reason", "retry signal"),
                metadata=policy.get("metadata", {}),
            ).model_dump()
        if control_type in {"skip", "route"}:
            return ControlDecision(
                action=control_type,
                targets=policy.get("targets", []),
                reason=policy.get("reason", f"{control_type} signal"),
                metadata=policy.get("metadata", {}),
            ).model_dump()
        return ControlDecision(
            action="proceed",
            reason=f"control_type {control_type!r} has no Runner mutation",
            metadata=policy.get("metadata", {}),
        ).model_dump()

    return MorphismAdapter.wrap(
        _apply,
        name=f"control.{op.control_type or 'proceed'}",
    )


def _operation_morphism(
    session: Session,
    decl: OperationDecl | None,
    op: Operation,
) -> MorphismAdapter:
    if getattr(op, "morphism", None) is not None:
        return session._morphism_for_operation(op)
    if decl is None:
        return _control_morphism(op)
    return decl.to_morphism(op.parameters, operation=op, name=op.operation_type)


def compile_flow_to_graph(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | UUID | str | None = None,
    verbose: bool = False,
    context: dict[str, Any] | None = None,
) -> tuple[OpGraph, Principal, dict[UUID, Operation], dict[UUID, Branch | None]]:
    if not graph.is_acyclic():
        raise ValueError("Operation graph has cycles - must be a DAG")

    operations = [node for node in graph.nodes if isinstance(node, Operation)]
    for node in graph.nodes:
        if not isinstance(node, Operation):
            raise ValueError(
                f"Graph contains non-Operation node: {node} ({type(node).__name__})"
            )

    default_branch = _default_branch(session, branch)
    operation_branches: dict[UUID, Branch | None] = {}
    op_nodes: dict[UUID, OpNode] = {}
    operations_by_id = {op.id: op for op in operations}

    for op in operations:
        op_branch = _resolve_operation_branch(
            session,
            op.metadata.get("branch"),
            default_branch,
        )
        operation_branches[op.id] = op_branch

        if getattr(op, "morphism", None) is not None:
            decl = None
        else:
            try:
                decl = session.operations.get_decl(op.operation_type)
            except KeyError:
                if not op.is_control:
                    raise
                decl = None

        branch_ref = None
        if op_branch is not None:
            branch_ref = getattr(op_branch, "name", None) or str(op_branch.id)

        direct_morphism = getattr(op, "morphism", None) is not None or (
            decl is not None and decl.handler is None
        )
        node_params = (
            session._morphism_node_params(
                op.parameters,
                session,
                op_branch,
                name=op.operation_type,
                verbose=verbose,
            )
            if direct_morphism
            else {
                "_lionagi_operation": op,
                "_lionagi_session": session,
                "_lionagi_branch": op_branch,
                "_lionagi_branch_ref": branch_ref,
                "_lionagi_operation_type": op.operation_type,
                "_lionagi_operation_name": _operation_name(op),
                "_lionagi_verbose": verbose,
            }
        )

        node = OpNode(
            id=op.id,
            m=_operation_morphism(session, decl, op),
            deps={
                pred.id
                for pred in graph.get_predecessors(op)
                if isinstance(pred, Operation)
            },
            params=node_params,
            control=op.is_control,
        )
        op_nodes[node.id] = node

    roots = {nid for nid, node in op_nodes.items() if not node.deps}
    principal = _principal_for_flow(default_branch, operation_branches, context)
    return (
        OpGraph(nodes=op_nodes, roots=roots),
        principal,
        operations_by_id,
        operation_branches,
    )


def _mark_completed(op: Operation, result: Any) -> None:
    op.execution.response = result
    op.execution.error = None
    op.execution.status = EventStatus.COMPLETED


async def flow(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | UUID | str | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
    verbose: bool = False,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    op_graph, principal, operations_by_id, _ = compile_flow_to_graph(
        session,
        graph,
        branch=branch,
        verbose=verbose,
        context=context,
    )
    runner = _runner_for_session(session, max_concurrent)
    runner_results: dict[UUID, dict[str, Any]] = {}
    failed: list[UUID | str] = []

    try:
        runner_results = await runner.run(principal, op_graph)
    except Exception as exc:
        if stop_on_error:
            raise
        logger.exception("Operation graph failed: %s", exc)
        failed.append("__runner__")

    operation_results: dict[UUID, Any] = {}
    results_by_name: dict[str, Any] = {}
    for node_id, result in runner_results.items():
        op = operations_by_id[node_id]
        unwrapped = _unwrap_result(result)
        _mark_completed(op, unwrapped)
        operation_results[node_id] = unwrapped
        results_by_name[_operation_name(op)] = unwrapped
        if verbose:
            logger.debug("Operation '%s' completed", _operation_name(op))

    payload: dict[str, Any] = {
        "operation_results": operation_results,
        "results_by_name": results_by_name,
        "failed_operations": failed,
        "cancelled_operations": [],
        "skipped_operations": [],
    }
    payload.update(results_by_name)
    return payload


async def flow_stream(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | UUID | str | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
    context: dict[str, Any] | None = None,
) -> AsyncGenerator[OperationResult, None]:
    op_graph, principal, operations_by_id, _ = compile_flow_to_graph(
        session,
        graph,
        branch=branch,
        context=context,
    )
    runner = _runner_for_session(session, max_concurrent)
    total = len(operations_by_id)
    completed = 0

    try:
        async for node_id, result in runner.run_stream(principal, op_graph):
            completed += 1
            op = operations_by_id[node_id]
            unwrapped = _unwrap_result(result)
            _mark_completed(op, unwrapped)
            yield OperationResult(
                name=_operation_name(op),
                result=unwrapped,
                completed=completed,
                total=total,
            )
    except Exception as exc:
        if stop_on_error:
            raise
        yield OperationResult(
            name="__runner__",
            result=None,
            error=exc,
            completed=completed,
            total=total,
        )
