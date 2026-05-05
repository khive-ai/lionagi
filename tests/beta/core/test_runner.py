# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.beta.core.runner: Runner."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from lionagi._errors import AccessError, ExecutionError
from lionagi.beta.core.graph import OpGraph, OpNode
from lionagi.beta.core.ipu import LenientIPU, StrictIPU, default_invariants
from lionagi.beta.core.morphism import MorphismAdapter
from lionagi.beta.core.runner import Runner
from lionagi.beta.core.types import Principal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _always_ok(br, **kw):
    return {"result": "ok"}


async def _echo(br, **kw):
    return {"echo": kw.get("value", None)}


def make_node(
    fn=None,
    *,
    requires=frozenset(),
    provides=frozenset(),
    deps=None,
    params=None,
    control=False,
):
    fn = fn or _always_ok
    m = MorphismAdapter.wrap(
        fn, name=fn.__name__, requires=frozenset(requires), provides=frozenset(provides)
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
# Construction
# ---------------------------------------------------------------------------


class TestRunnerConstruction:
    def test_default_construction(self):
        runner = Runner()
        assert isinstance(runner.ipu, LenientIPU)
        assert runner.max_concurrent is None
        assert runner.max_dynamic_nodes == 100

    def test_max_concurrent_none_ok(self):
        runner = Runner(max_concurrent=None)
        assert runner.max_concurrent is None

    def test_max_concurrent_positive_ok(self):
        runner = Runner(max_concurrent=4)
        assert runner.max_concurrent == 4

    def test_max_concurrent_one_ok(self):
        runner = Runner(max_concurrent=1)
        assert runner.max_concurrent == 1

    def test_max_concurrent_zero_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Runner(max_concurrent=0)

    def test_max_concurrent_negative_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Runner(max_concurrent=-1)

    def test_custom_ipu_stored(self):
        ipu = StrictIPU(default_invariants())
        runner = Runner(ipu=ipu)
        assert runner.ipu is ipu

    def test_max_dynamic_nodes_stored(self):
        runner = Runner(max_dynamic_nodes=50)
        assert runner.max_dynamic_nodes == 50


# ---------------------------------------------------------------------------
# run() — basic execution
# ---------------------------------------------------------------------------


class TestRunBasic:
    @pytest.mark.asyncio
    async def test_single_node_returns_result(self):
        runner = Runner()
        br = make_principal()
        g, node = single_node_graph()
        result = await runner.run(br, g)
        assert isinstance(result, dict)
        assert node.id in result
        assert result[node.id] == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_result_key_is_uuid(self):
        runner = Runner()
        br = make_principal()
        g, node = single_node_graph()
        result = await runner.run(br, g)
        for k in result:
            assert isinstance(k, UUID)

    @pytest.mark.asyncio
    async def test_chain_both_nodes_executed(self):
        runner = Runner()
        br = make_principal()

        async def _a(br, **kw):
            return {"a": 1}

        async def _b(br, **kw):
            return {"b": 2}

        na = make_node(_a)
        nb = make_node(_b, deps=[na.id])
        g = OpGraph(nodes={na.id: na, nb.id: nb}, roots={na.id})

        result = await runner.run(br, g)
        assert na.id in result
        assert nb.id in result
        assert result[na.id] == {"a": 1}
        assert result[nb.id] == {"b": 2}

    @pytest.mark.asyncio
    async def test_non_dict_result_wrapped(self):
        async def _raw(br, **kw):
            return "string_result"

        runner = Runner()
        br = make_principal()
        g, node = single_node_graph(_raw)
        result = await runner.run(br, g)
        assert result[node.id] == {"result": "string_result"}


# ---------------------------------------------------------------------------
# run_stream() — async generator
# ---------------------------------------------------------------------------


class TestRunStream:
    @pytest.mark.asyncio
    async def test_run_stream_yields_tuples(self):
        runner = Runner()
        br = make_principal()
        g, node = single_node_graph()
        items = []
        async for item in runner.run_stream(br, g):
            items.append(item)
        assert len(items) == 1
        node_id, res = items[0]
        assert isinstance(node_id, UUID)
        assert isinstance(res, dict)

    @pytest.mark.asyncio
    async def test_run_stream_node_id_matches_node(self):
        runner = Runner()
        br = make_principal()
        g, node = single_node_graph()
        async for node_id, res in runner.run_stream(br, g):
            assert node_id == node.id


# ---------------------------------------------------------------------------
# Satisfiability check
# ---------------------------------------------------------------------------


class TestSatisfiabilityCheck:
    @pytest.mark.asyncio
    async def test_satisfiability_disabled_skips_check(self):
        runner = Runner()
        br = make_principal()
        # Node requires 'fs.read', principal has none, but check is disabled
        g, node = single_node_graph(requires={"fs.read"})
        # Without the check, policy_check still runs and will deny
        # so we also need to grant the right for the policy check
        br_with_right = br.grant("fs.read")
        result = await runner.run(br_with_right, g, check_satisfiability=False)
        assert node.id in result

    @pytest.mark.asyncio
    async def test_satisfiability_check_passes_when_rights_present(self):
        runner = Runner()
        br = make_principal()
        br_with_right = br.grant("fs.read")
        g, node = single_node_graph(requires={"fs.read"})
        result = await runner.run(br_with_right, g, check_satisfiability=True)
        assert node.id in result

    @pytest.mark.asyncio
    async def test_satisfiability_check_fails_missing_right(self):
        runner = Runner()
        br = make_principal()  # no rights
        g, node = single_node_graph(requires={"fs.read"})
        with pytest.raises(AccessError, match="satisfiability"):
            await runner.run(br, g, check_satisfiability=True)

    @pytest.mark.asyncio
    async def test_satisfiability_check_default_enabled(self):
        runner = Runner()
        br = make_principal()
        g, node = single_node_graph(requires={"net.out"})
        with pytest.raises(AccessError):
            await runner.run(br, g)  # default check_satisfiability=True


# ---------------------------------------------------------------------------
# Policy check
# ---------------------------------------------------------------------------


class TestPolicyCheck:
    @pytest.mark.asyncio
    async def test_policy_denied_raises_access_error(self):
        runner = Runner()
        br = make_principal()
        g, node = single_node_graph(requires={"fs.read"})
        with pytest.raises(AccessError, match="Policy denied"):
            await runner.run(br, g, check_satisfiability=False)

    @pytest.mark.asyncio
    async def test_policy_passes_with_granted_right(self):
        runner = Runner()
        br = make_principal().grant("fs.read")
        g, node = single_node_graph(requires={"fs.read"})
        result = await runner.run(br, g, check_satisfiability=False)
        assert node.id in result

    @pytest.mark.asyncio
    async def test_policy_no_requires_always_passes(self):
        runner = Runner()
        br = make_principal()  # no rights
        g, node = single_node_graph()  # no requires
        result = await runner.run(br, g)
        assert node.id in result


# ---------------------------------------------------------------------------
# _normalize_control_action()
# ---------------------------------------------------------------------------


class TestNormalizeControlAction:
    def _make_graph_with_ctrl(self):
        async def _ok(br, **kw):
            return {"result": "ok"}

        m = MorphismAdapter.wrap(_ok, name="test")
        na = OpNode(m=m)
        ctrl = OpNode(m=m, control=True, deps={na.id})
        g = OpGraph(nodes={na.id: na, ctrl.id: ctrl}, roots={na.id})
        return g, na, ctrl

    def test_abort_normalized_to_halt(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action({"action": "abort"}, g, ctrl)
        assert result["action"] == "halt"

    def test_halt_stays_halt(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action({"action": "halt"}, g, ctrl)
        assert result["action"] == "halt"

    def test_proceed_passes_through(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action({"action": "proceed"}, g, ctrl)
        assert result["action"] == "proceed"

    def test_skip_passes_through(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action(
            {"action": "skip", "targets": [na.id]}, g, ctrl
        )
        assert result["action"] == "skip"

    def test_result_has_targets_list(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action({"action": "halt"}, g, ctrl)
        assert "targets" in result
        assert isinstance(result["targets"], list)

    def test_result_has_reason(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action(
            {"action": "halt", "reason": "test_reason"}, g, ctrl
        )
        assert result["reason"] == "test_reason"

    def test_empty_action_defaults_to_proceed(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        result = runner._normalize_control_action({}, g, ctrl)
        assert result["action"] == "proceed"

    def test_route_becomes_skip(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        # no successors to ctrl in this graph, so skip targets = [] effectively
        result = runner._normalize_control_action(
            {"action": "route", "targets": []}, g, ctrl
        )
        assert result["action"] == "skip"

    def test_spawn_action_passthrough(self):
        runner = Runner()
        g, na, ctrl = self._make_graph_with_ctrl()
        spawn_nodes = [{"morphism": None, "name": "dyn"}]
        result = runner._normalize_control_action(
            {"action": "spawn", "nodes": spawn_nodes}, g, ctrl
        )
        assert result["action"] == "spawn"
        assert result["metadata"]["spawn_nodes"] == spawn_nodes


# ---------------------------------------------------------------------------
# _resolve_node_id()
# ---------------------------------------------------------------------------


class TestResolveNodeId:
    def _make_graph(self):
        async def _ok(br, **kw):
            return {"result": "ok"}

        m = MorphismAdapter.wrap(_ok, name="test")
        na = OpNode(m=m, params={"name": "alpha"})
        nb = OpNode(m=m, params={"_lionagi_operation_name": "beta"})
        g = OpGraph(nodes={na.id: na, nb.id: nb}, roots={na.id})
        return g, na, nb

    def test_resolve_uuid_in_graph(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        assert runner._resolve_node_id(na.id, g) == na.id

    def test_resolve_uuid_not_in_graph_returns_none(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        unknown = uuid4()
        assert runner._resolve_node_id(unknown, g) is None

    def test_resolve_str_uuid_in_graph(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        assert runner._resolve_node_id(str(na.id), g) == na.id

    def test_resolve_str_uuid_not_in_graph_returns_none(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        assert runner._resolve_node_id(str(uuid4()), g) is None

    def test_resolve_by_name_param(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        assert runner._resolve_node_id("alpha", g) == na.id

    def test_resolve_by_lionagi_operation_name_param(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        assert runner._resolve_node_id("beta", g) == nb.id

    def test_resolve_unknown_name_returns_none(self):
        runner = Runner()
        g, na, nb = self._make_graph()
        assert runner._resolve_node_id("nonexistent", g) is None


# ---------------------------------------------------------------------------
# _apply_control_action()
# ---------------------------------------------------------------------------


class TestApplyControlAction:
    def _make_two_node_graph(self):
        async def _ok(br, **kw):
            return {"result": "ok"}

        m = MorphismAdapter.wrap(_ok, name="test")
        na = OpNode(m=m)
        nb = OpNode(m=m)
        g = OpGraph(nodes={na.id: na, nb.id: nb}, roots={na.id, nb.id})
        return g, na, nb, m

    def test_skip_removes_target_from_ready(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = {na.id, nb.id}
        done = set()
        results = {}
        action = {"action": "skip", "targets": [na.id]}
        runner._apply_control_action(action, g, ready, done, results)
        assert na.id not in ready
        assert nb.id in ready

    def test_skip_non_target_stays_in_ready(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = {na.id, nb.id}
        done = set()
        results = {}
        action = {"action": "skip", "targets": [na.id]}
        runner._apply_control_action(action, g, ready, done, results)
        assert nb.id in ready

    def test_retry_removes_from_done_and_adds_to_ready(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = set()
        done = {na.id}
        results = {na.id: {"result": "ok"}}
        action = {"action": "retry", "targets": [na.id]}
        runner._apply_control_action(action, g, ready, done, results)
        assert na.id not in done
        assert na.id in ready
        assert na.id not in results

    def test_retry_only_adds_to_ready_if_deps_satisfied(self):
        runner = Runner()
        runner._total_spawned = 0

        async def _ok(br, **kw):
            return {"result": "ok"}

        m = MorphismAdapter.wrap(_ok, name="test")
        na = OpNode(m=m)
        nb = OpNode(m=m, deps={na.id})  # nb depends on na
        g = OpGraph(nodes={na.id: na, nb.id: nb}, roots={na.id})

        ready = set()
        done = {nb.id}  # nb done, na not done
        results = {nb.id: {"result": "ok"}}
        action = {"action": "retry", "targets": [nb.id]}
        runner._apply_control_action(action, g, ready, done, results)
        # nb's dep (na) is not in done, so nb should NOT be re-added to ready
        assert nb.id not in ready

    def test_spawn_adds_new_node(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = set()
        done = set()
        results = {}
        action = {
            "action": "spawn",
            "targets": [],
            "metadata": {
                "spawn_nodes": [
                    {"morphism": m, "deps": [], "name": "dyn1"},
                ]
            },
        }
        runner._apply_control_action(action, g, ready, done, results)
        assert runner._total_spawned == 1
        # new node should be in graph
        assert len(g.nodes) == 3

    def test_spawn_adds_to_ready_when_no_deps(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = set()
        done = set()
        results = {}
        action = {
            "action": "spawn",
            "targets": [],
            "metadata": {
                "spawn_nodes": [
                    {"morphism": m, "deps": [], "name": "dyn1"},
                ]
            },
        }
        runner._apply_control_action(action, g, ready, done, results)
        # The spawned node has no deps, so it should be in ready
        assert len(ready) == 1

    def test_spawn_exceeds_max_dynamic_nodes_raises(self):
        runner = Runner(max_dynamic_nodes=2)
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = set()
        done = set()
        results = {}
        action = {
            "action": "spawn",
            "targets": [],
            "metadata": {
                "spawn_nodes": [
                    {"morphism": m, "deps": [], "name": f"n{i}"}
                    for i in range(3)  # 3 exceeds limit of 2
                ]
            },
        }
        with pytest.raises(ExecutionError, match="Dynamic node limit"):
            runner._apply_control_action(action, g, ready, done, results)

    def test_spawn_empty_nodes_noop(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = set()
        done = set()
        results = {}
        action = {
            "action": "spawn",
            "targets": [],
            "metadata": {"spawn_nodes": []},
        }
        runner._apply_control_action(action, g, ready, done, results)
        assert runner._total_spawned == 0
        assert len(ready) == 0

    def test_unknown_action_no_error(self):
        runner = Runner()
        runner._total_spawned = 0
        g, na, nb, m = self._make_two_node_graph()
        ready = {na.id}
        done = set()
        results = {}
        action = {"action": "unknown_action", "targets": []}
        # Unknown actions should not raise — they just fall through
        runner._apply_control_action(action, g, ready, done, results)
        assert na.id in ready


# ---------------------------------------------------------------------------
# StrictIPU / pre() returning False
# ---------------------------------------------------------------------------


class TestStrictIPUIntegration:
    @pytest.mark.asyncio
    async def test_pre_returning_false_raises_assertion_error(self):
        class FailingPreMorphism(MorphismAdapter, kw_only=True):
            async def pre(self, br, **kw):
                return False

        m = FailingPreMorphism(name="failing_pre", _fn=_always_ok)
        node = OpNode(m=m)
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner(ipu=StrictIPU(default_invariants()))

        with pytest.raises(AssertionError, match="pre\\(\\) returned False"):
            await runner.run(br, g)

    @pytest.mark.asyncio
    async def test_post_returning_false_raises_assertion_error(self):
        class FailingPostMorphism(MorphismAdapter, kw_only=True):
            async def post(self, br, result, **kw):
                return False

        m = FailingPostMorphism(name="failing_post", _fn=_always_ok)
        node = OpNode(m=m)
        g = OpGraph(nodes={node.id: node}, roots={node.id})
        br = make_principal()
        runner = Runner()

        with pytest.raises(AssertionError, match="post\\(\\) returned False"):
            await runner.run(br, g)
