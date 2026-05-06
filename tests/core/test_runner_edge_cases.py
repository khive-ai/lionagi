# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge-case tests for lionagi.core.runner: Runner."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from lionagi._errors import ExecutionError
from lionagi.core.graph import OpGraph, OpNode
from lionagi.core.ipu import IPU
from lionagi.core.morphism import MorphismAdapter
from lionagi.core.runner import Runner, _unwrap_exception_group
from lionagi.core.types import Principal

try:
    _EGBase = BaseExceptionGroup
except NameError:
    from exceptiongroup import BaseExceptionGroup as _EGBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _always_ok(br, **kw):
    return {"result": "ok"}


def make_node(
    fn=None,
    *,
    requires=frozenset(),
    provides=frozenset(),
    deps=None,
    params=None,
    control=False,
    name=None,
):
    fn = fn or _always_ok
    node_name = name or fn.__name__
    m = MorphismAdapter.wrap(
        fn,
        name=node_name,
        requires=frozenset(requires),
        provides=frozenset(provides),
    )
    node = OpNode(
        m=m,
        deps=set(deps) if deps else set(),
        params=params or {},
        control=control,
    )
    return node


def make_principal(**ctx):
    return Principal(name="test", ctx=ctx)


def single_node_graph(fn=None, **node_kwargs):
    node = make_node(fn, **node_kwargs)
    g = OpGraph(nodes={node.id: node}, roots={node.id})
    return g, node


# ---------------------------------------------------------------------------
# _unwrap_exception_group
# ---------------------------------------------------------------------------


class TestUnwrapExceptionGroup:
    def test_non_group_exception_returned_as_is(self):
        e = ValueError("x")
        result = _unwrap_exception_group(e)
        assert result is e

    def test_runtime_error_returned_as_is(self):
        e = RuntimeError("rt")
        assert _unwrap_exception_group(e) is e

    def test_single_exception_group_unwrapped(self):
        inner = ValueError("y")
        eg = _EGBase("group", [inner])
        result = _unwrap_exception_group(eg)
        assert result is inner

    def test_single_exception_group_type_preserved(self):
        inner = TypeError("bad type")
        eg = _EGBase("group", [inner])
        result = _unwrap_exception_group(eg)
        assert isinstance(result, TypeError)

    def test_multi_exception_group_not_unwrapped(self):
        eg = _EGBase("group", [ValueError("a"), ValueError("b")])
        result = _unwrap_exception_group(eg)
        assert result is eg

    def test_empty_like_non_group_returned_as_is(self):
        e = KeyError("k")
        assert _unwrap_exception_group(e) is e


# ---------------------------------------------------------------------------
# pre() returning False → AssertionError
# ---------------------------------------------------------------------------


class FalsePreMorphism(MorphismAdapter, kw_only=True):
    name: str = "false_pre"

    async def pre(self, br, **kw):
        return False

    async def apply(self, br, **kw):
        return {"ok": True}


class TestPreReturnsFalse:
    @pytest.mark.asyncio
    async def test_pre_false_raises_assertion_error(self):
        m = FalsePreMorphism()
        node = OpNode(m=m, deps=set())
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner()
        with pytest.raises(AssertionError, match="pre\\(\\) returned False"):
            await runner.run(br, g)

    @pytest.mark.asyncio
    async def test_pre_false_error_message_contains_morphism_name(self):
        m = FalsePreMorphism()
        node = OpNode(m=m, deps=set())
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner()
        with pytest.raises(AssertionError, match="false_pre"):
            await runner.run(br, g)


# ---------------------------------------------------------------------------
# post() returning False → AssertionError
# ---------------------------------------------------------------------------


class FalsePostMorphism(MorphismAdapter, kw_only=True):
    name: str = "false_post"

    async def apply(self, br, **kw):
        return {"ok": True}

    async def post(self, br, result):
        return False


class TestPostReturnsFalse:
    @pytest.mark.asyncio
    async def test_post_false_raises_assertion_error(self):
        m = FalsePostMorphism()
        node = OpNode(m=m, deps=set())
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner()
        with pytest.raises(AssertionError, match="post\\(\\) returned False"):
            await runner.run(br, g)

    @pytest.mark.asyncio
    async def test_post_false_error_message_contains_morphism_name(self):
        m = FalsePostMorphism()
        node = OpNode(m=m, deps=set())
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner()
        with pytest.raises(AssertionError, match="false_post"):
            await runner.run(br, g)


# ---------------------------------------------------------------------------
# halt action stops remaining nodes
# ---------------------------------------------------------------------------


class TestHaltAction:
    @pytest.mark.asyncio
    async def test_halt_stops_remaining_nodes(self):
        executed = []

        async def _node_a(br, **kw):
            executed.append("a")
            return {"a": 1}

        async def _halt_ctrl(br, **kw):
            executed.append("ctrl")
            return {"action": "halt", "reason": "stopping"}

        async def _node_b(br, **kw):
            executed.append("b")
            return {"b": 2}

        na = make_node(_node_a, name="a")
        nc = make_node(_halt_ctrl, name="halt_ctrl", deps=[na.id], control=True)
        nb = make_node(_node_b, name="b", deps=[nc.id])

        g = OpGraph(
            nodes={na.id: na, nc.id: nc, nb.id: nb},
            roots={na.id},
        )
        br = make_principal()
        runner = Runner()
        result = await runner.run(br, g)

        assert "a" in executed
        assert "ctrl" in executed
        assert "b" not in executed  # halted before b ran

    @pytest.mark.asyncio
    async def test_halt_result_contains_completed_nodes(self):
        async def _a(br, **kw):
            return {"a": 1}

        async def _halt_ctrl(br, **kw):
            return {"action": "halt", "reason": "stop"}

        na = make_node(_a, name="a")
        nc = make_node(_halt_ctrl, name="ctrl", deps=[na.id], control=True)

        async def _b(br, **kw):
            return {"b": 2}

        nb = make_node(_b, name="b", deps=[nc.id])
        g = OpGraph(nodes={na.id: na, nc.id: nc, nb.id: nb}, roots={na.id})
        br = make_principal()
        runner = Runner()
        result = await runner.run(br, g)

        assert na.id in result
        assert nc.id in result
        assert nb.id not in result


# ---------------------------------------------------------------------------
# Route action: only specified targets proceed
# ---------------------------------------------------------------------------


class TestRouteAction:
    @pytest.mark.asyncio
    async def test_route_skips_non_targeted_nodes(self):
        executed = []

        async def _source(br, **kw):
            return {"src": 1}

        async def _route_ctrl(br, **kw):
            executed.append("ctrl")
            # route to nb only (not nc)
            return {
                "action": "route",
                "targets": [str(nb.id)],
                "reason": "routing to b only",
            }

        async def _node_b(br, **kw):
            executed.append("b")
            return {"b": 2}

        async def _node_c(br, **kw):
            executed.append("c")
            return {"c": 3}

        na = make_node(_source, name="source")
        nc_ctrl = make_node(_route_ctrl, name="ctrl", deps=[na.id], control=True)
        nb = make_node(_node_b, name="b", deps=[nc_ctrl.id])
        nc = make_node(_node_c, name="c", deps=[nc_ctrl.id])

        g = OpGraph(
            nodes={na.id: na, nc_ctrl.id: nc_ctrl, nb.id: nb, nc.id: nc},
            roots={na.id},
        )
        br = make_principal()
        runner = Runner()
        result = await runner.run(br, g)

        assert "b" in executed
        assert "c" not in executed

    @pytest.mark.asyncio
    async def test_route_targeted_node_in_results(self):
        async def _source(br, **kw):
            return {"src": 1}

        async def _route_ctrl(br, **kw):
            return {
                "action": "route",
                "targets": [str(nb.id)],
                "reason": "routing",
            }

        async def _node_b(br, **kw):
            return {"b": 2}

        async def _node_c(br, **kw):
            return {"c": 3}

        na = make_node(_source, name="source")
        nc_ctrl = make_node(_route_ctrl, name="ctrl", deps=[na.id], control=True)
        nb = make_node(_node_b, name="b", deps=[nc_ctrl.id])
        nc = make_node(_node_c, name="c", deps=[nc_ctrl.id])

        g = OpGraph(
            nodes={na.id: na, nc_ctrl.id: nc_ctrl, nb.id: nb, nc.id: nc},
            roots={na.id},
        )
        br = make_principal()
        runner = Runner()
        result = await runner.run(br, g)
        assert nb.id in result


# ---------------------------------------------------------------------------
# Spawn: missing 'morphism' key → ValueError
# ---------------------------------------------------------------------------


class TestSpawnMissingMorphism:
    @pytest.mark.asyncio
    async def test_spawn_missing_morphism_raises_value_error(self):
        async def _spawn_ctrl(br, **kw):
            return {
                "action": "spawn",
                "nodes": [{"name": "new_node"}],  # no 'morphism' key
                "reason": "spawning",
            }

        mc = make_node(_spawn_ctrl, name="ctrl", control=True)
        g = OpGraph(nodes={mc.id: mc}, roots={mc.id})
        br = make_principal()
        runner = Runner()
        with pytest.raises(ValueError, match="missing 'morphism'"):
            await runner.run(br, g)


# ---------------------------------------------------------------------------
# Spawn: unresolved dependency → ValueError
# ---------------------------------------------------------------------------


class TestSpawnUnresolvedDep:
    @pytest.mark.asyncio
    async def test_spawn_unresolved_dep_raises_value_error(self):
        mn = MorphismAdapter.wrap(_always_ok, name="node")

        async def _spawn_ctrl(br, **kw):
            return {
                "action": "spawn",
                "nodes": [
                    {"name": "n1", "morphism": mn, "deps": ["nonexistent_node"]},
                ],
                "reason": "spawning",
            }

        mc = make_node(_spawn_ctrl, name="ctrl", control=True)
        g = OpGraph(nodes={mc.id: mc}, roots={mc.id})
        br = make_principal()
        runner = Runner()
        with pytest.raises(ValueError, match="unresolved dependency"):
            await runner.run(br, g)


# ---------------------------------------------------------------------------
# Spawn: exceeding max_dynamic_nodes → ExecutionError
# ---------------------------------------------------------------------------


class TestSpawnMaxDynamicNodes:
    @pytest.mark.asyncio
    async def test_spawn_exceeds_limit_raises_execution_error(self):
        mn = MorphismAdapter.wrap(_always_ok, name="node")

        async def _spawn_ctrl(br, **kw):
            return {
                "action": "spawn",
                "nodes": [
                    {"name": f"n{i}", "morphism": mn, "deps": []} for i in range(3)
                ],
                "reason": "spawning",
            }

        mc = make_node(_spawn_ctrl, name="ctrl", control=True)
        g = OpGraph(nodes={mc.id: mc}, roots={mc.id})
        br = make_principal()
        runner = Runner(max_dynamic_nodes=2)
        with pytest.raises(ExecutionError, match="Dynamic node limit"):
            await runner.run(br, g)

    @pytest.mark.asyncio
    async def test_spawn_within_limit_succeeds(self):
        mn = MorphismAdapter.wrap(_always_ok, name="node")

        async def _spawn_ctrl(br, **kw):
            return {
                "action": "spawn",
                "nodes": [
                    {"name": f"n{i}", "morphism": mn, "deps": []} for i in range(2)
                ],
                "reason": "spawning",
            }

        mc = make_node(_spawn_ctrl, name="ctrl", control=True)
        g = OpGraph(nodes={mc.id: mc}, roots={mc.id})
        br = make_principal()
        runner = Runner(max_dynamic_nodes=5)
        result = await runner.run(br, g)
        # ctrl + 2 spawned nodes = 3
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Spawn: on_spawn event handler triggered
# ---------------------------------------------------------------------------


class TestSpawnOnSpawnHandler:
    @pytest.mark.asyncio
    async def test_spawn_emits_graph_spawn_event(self):
        mn = MorphismAdapter.wrap(_always_ok, name="node")
        spawn_events = []

        async def _spawn_ctrl(br, **kw):
            return {
                "action": "spawn",
                "nodes": [{"name": "spawned_node", "morphism": mn, "deps": []}],
                "reason": "spawning",
            }

        mc = make_node(_spawn_ctrl, name="ctrl", control=True)
        g = OpGraph(nodes={mc.id: mc}, roots={mc.id})
        br = make_principal()
        runner = Runner()

        async def on_spawn_handler(br, ctrl_node, info):
            spawn_events.append(info)

        runner.bus.subscribe("graph.spawn", on_spawn_handler)
        await runner.run(br, g)

        assert len(spawn_events) == 1
        assert spawn_events[0]["spawned_count"] == 1
        assert "spawned_node" in spawn_events[0]["names"]


# ---------------------------------------------------------------------------
# max_concurrent limiter path in _run_parallel_nodes
# ---------------------------------------------------------------------------


class TestMaxConcurrentLimiter:
    @pytest.mark.asyncio
    async def test_max_concurrent_all_nodes_execute(self):
        results = []

        async def _node(br, **kw):
            results.append("ran")
            return {"ok": True}

        mn = MorphismAdapter.wrap(_node, name="node")
        n1 = OpNode(m=mn, deps=set())
        n2 = OpNode(m=mn, deps=set())
        n3 = OpNode(m=mn, deps=set())
        g = OpGraph(
            nodes={n1.id: n1, n2.id: n2, n3.id: n3},
            roots={n1.id, n2.id, n3.id},
        )
        br = make_principal()
        runner = Runner(max_concurrent=2)
        result = await runner.run(br, g)

        assert len(result) == 3
        assert len(results) == 3


# ---------------------------------------------------------------------------
# IPU after_node raised during error cleanup is suppressed
# ---------------------------------------------------------------------------


class ErrorAfterNodeIPU(IPU):
    """IPU that raises in after_node to test error suppression."""

    def __init__(self):
        pass

    async def before_node(self, br, node):
        pass

    async def after_node(self, br, node, result, *, error=None):
        raise RuntimeError("after_node failed during cleanup")

    async def on_observation(self, obs):
        pass


class TestIPUAfterNodeErrorDuringCleanup:
    @pytest.mark.asyncio
    async def test_ipu_after_node_error_suppressed_original_error_raised(self):
        """When a node fails AND after_node also raises, original error surfaces."""

        async def _failing_node(br, **kw):
            raise ValueError("node failed")

        mn = MorphismAdapter.wrap(_failing_node, name="failing")
        node = OpNode(m=mn, deps=set())
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner(ipu=ErrorAfterNodeIPU())

        # Original ValueError should surface, IPU RuntimeError should be suppressed
        with pytest.raises(ValueError, match="node failed"):
            await runner.run(br, g)
