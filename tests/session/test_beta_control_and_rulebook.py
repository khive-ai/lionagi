import pytest

from lionagi.beta.core.graph import OpGraph, OpNode
from lionagi.beta.core.morphism import MorphismAdapter
from lionagi.beta.core.runner import Runner
from lionagi.beta.core.types import Principal
from lionagi.beta.rules import RuleBook, Validator
from lionagi.beta.session.session import Session
from lionagi.beta.work.builder import OperationGraphBuilder


@pytest.mark.asyncio
async def test_runner_emits_control_action_event():
    async def regular(_principal, **_kw):
        return {"regular": True}

    async def control(_principal, **_kw):
        return {"action": "halt", "reason": "stop here"}

    async def after_halt(_principal, **_kw):
        return {"should_not_run": True}

    regular_node = OpNode(m=MorphismAdapter.wrap(regular, name="regular"))
    control_node = OpNode(
        m=MorphismAdapter.wrap(control, name="control.halt"),
        control=True,
    )
    blocked_node = OpNode(
        m=MorphismAdapter.wrap(after_halt, name="blocked"),
        deps={control_node.id},
    )
    graph = OpGraph(
        nodes={
            regular_node.id: regular_node,
            control_node.id: control_node,
            blocked_node.id: blocked_node,
        },
        roots={regular_node.id, control_node.id},
    )
    runner = Runner()
    seen = []

    async def on_control(_principal, node, action):
        seen.append((node.id, action))

    runner.bus.subscribe("control.action", on_control)

    results = await runner.run(Principal(name="test"), graph)

    assert control_node.id in results
    assert blocked_node.id not in results
    assert seen == [
        (
            control_node.id,
            {
                "action": "halt",
                "targets": [],
                "reason": "stop here",
                "metadata": {},
            },
        )
    ]


@pytest.mark.asyncio
async def test_work_builder_control_compiles_to_runner_control():
    async def echo(params, _ctx):
        return params

    session = Session()
    session.register_operation("echo", echo)

    builder = OperationGraphBuilder()
    builder.add("first", "echo", {"value": 1})
    builder.add_control("gate", "halt", reason="done", depends_on=["first"])
    builder.add("blocked", "echo", {"value": 2}, depends_on=["gate"])

    result = await session.flow(builder.get_graph())

    assert result["results_by_name"]["first"] == {"value": 1}
    assert result["results_by_name"]["gate"]["action"] == "halt"
    assert "blocked" not in result["results_by_name"]


def test_validator_rejects_distinct_registry_and_rulebook():
    with pytest.raises(ValueError, match="registry or rulebook"):
        Validator(registry=RuleBook(), rulebook=RuleBook())
