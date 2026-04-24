# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for _topo_sort_ops (CLI plan validation)."""

import pytest

from lionagi.cli.orchestrate.flow import FlowOp, _topo_sort_ops


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
