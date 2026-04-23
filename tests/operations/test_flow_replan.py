# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for #913: control-node re-plan round 2 execution."""

from unittest.mock import AsyncMock

import pytest

from lionagi.operations.builder import OperationGraphBuilder
from lionagi.operations.flow import DependencyAwareExecutor, flow
from lionagi.operations.node import Operation
from lionagi.protocols._concepts import Condition
from lionagi.protocols.generic.event import EventStatus
from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import OpenAIChatCompletionsRequest
from lionagi.session.branch import Branch
from lionagi.session.session import Session


def _make_mock_branch(name: str = "TestBranch") -> Branch:
    branch = Branch(user="test_user", name=name)

    async def _fake_invoke(**kwargs):
        config = _get_oai_config(
            name="oai_chat",
            endpoint="chat/completions",
            request_options=OpenAIChatCompletionsRequest,
            kwargs={"model": "gpt-4.1-mini"},
        )
        endpoint = Endpoint(config=config)
        fake_call = APICalling(
            payload={"model": "gpt-4.1-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = "mocked_response"
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    mock_invoke = AsyncMock(side_effect=_fake_invoke)
    mock_chat_model = iModel(
        provider="openai", model="gpt-4.1-mini", api_key="test_key"
    )
    mock_chat_model.invoke = mock_invoke
    branch.chat_model = mock_chat_model
    return branch


@pytest.mark.asyncio
async def test_reexecute_graph_with_completed_ops():
    """Round 2 ops must execute even when the graph contains completed ops from Round 1."""
    branch = _make_mock_branch()
    session = Session(default_branch=branch)

    op_a = Operation(operation="chat", parameters={"instruction": "A"})
    op_b = Operation(operation="chat", parameters={"instruction": "B"})
    graph = Graph()
    graph.add_node(op_a)
    graph.add_node(op_b)
    graph.add_edge(Edge(head=op_a.id, tail=op_b.id))

    r1 = await flow(session, graph, parallel=False)
    assert op_a.execution.status == EventStatus.COMPLETED
    assert op_b.execution.status == EventStatus.COMPLETED
    assert len(r1["completed_operations"]) == 2

    # Round 2: add new op depending on completed op_b
    op_c = Operation(operation="chat", parameters={"instruction": "C"})
    graph.add_node(op_c)
    graph.add_edge(Edge(head=op_b.id, tail=op_c.id))

    r2 = await flow(session, graph, parallel=False)
    assert op_c.execution.status == EventStatus.COMPLETED
    assert op_c.id in r2["operation_results"]


@pytest.mark.asyncio
async def test_reexecute_via_builder():
    """Simulate the flow.py pattern: build ops, execute, add more, execute again."""
    branch = _make_mock_branch()
    session = Session(default_branch=branch)
    builder = OperationGraphBuilder("test")

    plan_node = builder.add_operation("chat", branch=branch, instruction="plan")
    r1 = await session.flow(builder.get_graph())
    assert plan_node in r1["operation_results"]

    w1 = builder.add_operation(
        "chat", branch=branch, depends_on=[plan_node], instruction="work1"
    )
    w2 = builder.add_operation(
        "chat", branch=branch, depends_on=[plan_node], instruction="work2"
    )
    r2 = await session.flow(builder.get_graph())
    assert w1 in r2["operation_results"]
    assert w2 in r2["operation_results"]

    ctrl = builder.add_operation(
        "chat", branch=branch, depends_on=[w1, w2], instruction="review"
    )
    r3 = await session.flow(builder.get_graph())
    assert ctrl in r3["operation_results"]

    # Re-plan: NEW ops depending on ctrl
    new1 = builder.add_operation(
        "chat", branch=branch, depends_on=[ctrl], instruction="fix1"
    )
    new2 = builder.add_operation(
        "chat", branch=branch, depends_on=[new1], instruction="fix2"
    )
    r4 = await session.flow(builder.get_graph())
    assert new1 in r4["operation_results"]
    assert new2 in r4["operation_results"]
    assert r4["operation_results"][new1] is not None
    assert r4["operation_results"][new2] is not None


@pytest.mark.asyncio
async def test_as_fresh_event_preserves_parameters():
    """as_fresh_event must carry over Operation.parameters (exclude=True field)."""
    op = Operation(
        operation="chat",
        parameters={
            "instruction": "hello",
            "guidance": "be nice",
            "context": {"items": [1]},
        },
        metadata={"tags": ["old"]},
    )
    op.execution.status = EventStatus.COMPLETED
    op.execution.response = "done"

    fresh = op.as_fresh_event(copy_meta=True)
    assert fresh.parameters == {
        "instruction": "hello",
        "guidance": "be nice",
        "context": {"items": [1]},
    }
    assert fresh.execution.status == EventStatus.PENDING
    assert fresh.id != op.id
    assert fresh.metadata.get("original", {}).get("id") == str(op.id)

    fresh.parameters["context"]["items"].append(2)
    fresh.metadata["tags"].append("new")
    assert op.parameters["context"]["items"] == [1]
    assert op.metadata["tags"] == ["old"]


def test_topo_sort_ops():
    """_topo_sort_ops must order child after parent even if listed first."""
    import pytest as _pt

    from lionagi.cli.orchestrate.flow import FlowOp, _topo_sort_ops

    ops = [
        FlowOp(id="c", agent_id="a", instruction="C", depends_on=["b"]),
        FlowOp(id="a", agent_id="a", instruction="A"),
        FlowOp(id="b", agent_id="a", instruction="B", depends_on=["a"]),
    ]
    result = _topo_sort_ops(ops)
    ids = [o.id for o in result]
    assert ids.index("a") < ids.index("b") < ids.index("c")

    # Unknown dep → fail closed, not silently dropped
    with _pt.raises(ValueError, match="unknown dependency"):
        _topo_sort_ops(
            [FlowOp(id="x", agent_id="a", instruction="X", depends_on=["missing"])]
        )

    # Cycle → detected
    with _pt.raises(ValueError, match="cycle"):
        _topo_sort_ops(
            [
                FlowOp(id="p", agent_id="a", instruction="P", depends_on=["q"]),
                FlowOp(id="q", agent_id="a", instruction="Q", depends_on=["p"]),
            ]
        )


def test_unknown_control_type_raises():
    """Executor must fail closed on unknown control_type, not silently proceed."""
    import pytest as _pt

    from lionagi.operations.flow import DependencyAwareExecutor
    from lionagi.protocols.graph.graph import Graph

    branch = _make_mock_branch()
    session = Session(default_branch=branch)

    op = Operation(operation="chat", parameters={"instruction": "x"})
    op.control_type = "typo_not_registered"

    graph = Graph()
    graph.add_node(op)

    executor = DependencyAwareExecutor(session=session, graph=graph)

    import asyncio

    with _pt.raises(ValueError, match="Unknown control_type"):
        asyncio.get_event_loop().run_until_complete(
            executor._evaluate_control(op)
        )


def test_quorum_threshold_policy_accepts_count_and_fraction():
    branch = _make_mock_branch()
    session = Session(default_branch=branch)

    op_ok = Operation(operation="chat", parameters={"instruction": "ok"})
    op_fail = Operation(operation="chat", parameters={"instruction": "fail"})
    quorum = Operation(operation="chat", parameters={}, control_type="quorum")

    graph = Graph()
    graph.add_node(op_ok)
    graph.add_node(op_fail)
    graph.add_node(quorum)
    graph.add_edge(Edge(head=op_ok.id, tail=quorum.id))
    graph.add_edge(Edge(head=op_fail.id, tail=quorum.id))

    executor = DependencyAwareExecutor(session=session, graph=graph)
    executor.results[op_ok.id] = "done"
    executor.results[op_fail.id] = {"error": "boom"}

    count_decision = executor._eval_quorum(quorum, {"threshold": 1})
    assert count_decision.action == "proceed"

    fraction_decision = executor._eval_quorum(quorum, {"threshold": 0.5})
    assert fraction_decision.action == "proceed"

    strict_decision = executor._eval_quorum(quorum, {"threshold": 2})
    assert strict_decision.action == "halt"

    empty_quorum = Operation(operation="chat", parameters={}, control_type="quorum")
    empty_graph = Graph()
    empty_graph.add_node(empty_quorum)
    empty_executor = DependencyAwareExecutor(session=session, graph=empty_graph)

    assert empty_executor._eval_quorum(empty_quorum, {}).action == "halt"
    assert (
        empty_executor._eval_quorum(empty_quorum, {"allow_empty": True}).action
        == "proceed"
    )


def test_validate_edge_conditions_rejects_sync_apply():
    class SyncCondition(Condition):
        def apply(self, *args, **kwargs) -> bool:
            return True

    branch = _make_mock_branch()
    session = Session(default_branch=branch)
    op_a = Operation(operation="chat", parameters={"instruction": "a"})
    op_b = Operation(operation="chat", parameters={"instruction": "b"})
    graph = Graph()
    graph.add_node(op_a)
    graph.add_node(op_b)
    graph.add_edge(Edge(head=op_a.id, tail=op_b.id, condition=SyncCondition()))

    executor = DependencyAwareExecutor(session=session, graph=graph)

    with pytest.raises(TypeError, match="apply\\(\\) must be async"):
        executor._validate_edge_conditions()


@pytest.mark.asyncio
async def test_terminal_status_events_set_in_init():
    """Completion events for ALL terminal statuses must be set in executor init."""
    branch = _make_mock_branch()
    session = Session(default_branch=branch)

    op_ok = Operation(operation="chat", parameters={"instruction": "ok"})
    op_fail = Operation(operation="chat", parameters={"instruction": "fail"})

    graph = Graph()
    graph.add_node(op_ok)
    graph.add_node(op_fail)

    op_ok.execution.status = EventStatus.COMPLETED
    op_ok.execution.response = "done"
    op_fail.execution.status = EventStatus.FAILED
    op_fail.execution.response = None

    executor = DependencyAwareExecutor(
        session=session, graph=graph, max_concurrent=5
    )

    assert executor.completion_events[op_ok.id].is_set()
    assert executor.completion_events[op_fail.id].is_set()
    assert executor.results[op_ok.id] == "done"
