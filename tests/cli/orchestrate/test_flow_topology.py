# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for _topo_sort_ops (CLI plan validation)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from lionagi.cli.orchestrate.flow import (
    FlowAgent,
    FlowOp,
    FlowPlan,
    _run_flow_inner,
    _topo_sort_ops,
)


def _op(oid: str, deps: list[str] | None = None) -> FlowOp:
    return FlowOp(
        id=oid,
        agent_id="a1",
        instruction="do the thing",
        depends_on=deps,
    )


def test_topo_sort_already_sorted():
    ops = [_op("a"), _op("b", ["a"]), _op("c", ["b"])]
    sorted_ops = _topo_sort_ops(ops)
    assert [o.id for o in sorted_ops] == ["a", "b", "c"]


def test_topo_sort_unsorted_input_is_reordered_parent_before_child():
    ops = [_op("c", ["b"]), _op("b", ["a"]), _op("a")]
    sorted_ops = _topo_sort_ops(ops)
    order = [o.id for o in sorted_ops]
    assert order.index("a") < order.index("b") < order.index("c")


def test_topo_sort_diamond():
    # a -> b, a -> c, b -> d, c -> d
    ops = [_op("d", ["b", "c"]), _op("b", ["a"]), _op("c", ["a"]), _op("a")]
    sorted_ops = _topo_sort_ops(ops)
    order = [o.id for o in sorted_ops]
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_topo_sort_rejects_unknown_dependency():
    ops = [_op("a"), _op("b", ["nonexistent"])]
    with pytest.raises(ValueError, match="unknown dependency"):
        _topo_sort_ops(ops)


def test_topo_sort_allows_dependency_on_existing_op():
    ops = [_op("new", ["old"])]
    sorted_ops = _topo_sort_ops(ops, existing_op_ids={"old"})
    assert [o.id for o in sorted_ops] == ["new"]


def test_topo_sort_rejects_duplicate_op_ids():
    ops = [_op("a"), _op("a")]
    with pytest.raises(ValueError, match="Duplicate op id"):
        _topo_sort_ops(ops)


def test_topo_sort_rejects_self_cycle():
    ops = [_op("a", ["a"])]
    with pytest.raises(ValueError, match="cycle detected"):
        _topo_sort_ops(ops)


def test_topo_sort_rejects_two_node_cycle():
    ops = [_op("a", ["b"]), _op("b", ["a"])]
    with pytest.raises(ValueError, match="cycle detected"):
        _topo_sort_ops(ops)


def test_topo_sort_rejects_three_node_cycle():
    ops = [_op("a", ["c"]), _op("b", ["a"]), _op("c", ["b"])]
    with pytest.raises(ValueError, match="cycle detected"):
        _topo_sort_ops(ops)


def test_topo_sort_empty_list():
    assert _topo_sort_ops([]) == []


def test_topo_sort_single_op_no_deps():
    [only] = _topo_sort_ops([_op("solo")])
    assert only.id == "solo"


class _FakeBuilder:
    def __init__(self):
        self.added = []

    def add_operation(self, operation, **kwargs):
        node_id = f"node-{len(self.added) + 1}"
        self.added.append({"id": node_id, "operation": operation, "kwargs": kwargs})
        return node_id

    def get_graph(self):
        return object()


class _FakeSession:
    def __init__(self, builder, plan):
        self.builder = builder
        self.plan = plan

    async def flow(self, _graph, **_kwargs):
        plan_root = self.builder.added[0]["id"]
        return {"operation_results": {plan_root: SimpleNamespace(plan=self.plan)}}


@pytest.mark.asyncio
async def test_run_flow_inner_dry_run_uses_topologically_sorted_ops(tmp_path):
    plan = FlowPlan(
        agents=[FlowAgent(id="a1", role="researcher")],
        operations=[_op("c", ["b"]), _op("b", ["a"]), _op("a")],
    )
    builder = _FakeBuilder()
    env = SimpleNamespace(
        run=SimpleNamespace(
            artifact_root=tmp_path,
            dag_image_path=tmp_path / "dag.png",
        ),
        session=_FakeSession(builder, plan),
        orc_branch=SimpleNamespace(id=uuid4()),
        builder=builder,
        bare=True,
        effort=None,
        verbose=False,
        team_data=None,
    )

    output = await _run_flow_inner(
        "codex/gpt-5.5",
        "do the task",
        env=env,
        dry_run=True,
    )

    op_ids = [
        line.strip().split()[0]
        for line in output.splitlines()
        if line.startswith("  ") and line.strip().split()[0] in {"a", "b", "c"}
    ]
    assert op_ids == ["a", "b", "c"]
