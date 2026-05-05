# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.beta.core.ipu: invariants and IPU implementations."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from lionagi.beta.core.graph import OpGraph, OpNode
from lionagi.beta.core.ipu import (
    LatencyBound,
    LenientIPU,
    PolicyGatePresent,
    ResultShape,
    StrictIPU,
    default_invariants,
)
from lionagi.beta.core.morphism import MorphismAdapter
from lionagi.beta.core.types import Observation, Principal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(fn=None, name="test", **node_kwargs):
    if fn is None:

        async def fn(br, **kw):
            return {"ok": True}

    m = MorphismAdapter.wrap(fn, name=name)
    return OpNode(m=m, **node_kwargs)


def make_node_with_baseop(**attrs):
    """Create an OpNode whose .m is a BaseOp instance (mutable), with given attributes set."""
    from lionagi.beta.core.wrappers import BaseOp

    class _Op(BaseOp):
        async def apply(self, br, **kw):
            return {"ok": True}

    op = _Op()
    for k, v in attrs.items():
        setattr(op, k, v)
    return OpNode(m=op)


def make_br():
    return Principal()


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# PolicyGatePresent
# ---------------------------------------------------------------------------


class TestPolicyGatePresent:
    def test_pre_always_true(self):
        inv = PolicyGatePresent()
        node = make_node()
        br = make_br()
        assert inv.pre(br, node) is True

    def test_post_always_true(self):
        inv = PolicyGatePresent()
        node = make_node()
        br = make_br()
        assert inv.post(br, node, {"any": "result"}) is True

    def test_name(self):
        inv = PolicyGatePresent()
        assert inv.name == "PolicyGatePresent"


# ---------------------------------------------------------------------------
# LatencyBound
# ---------------------------------------------------------------------------


class TestLatencyBound:
    def test_pre_stores_t0_and_returns_true(self):
        inv = LatencyBound()
        br = make_br()
        node = make_node()
        result = inv.pre(br, node)
        assert result is True
        assert (br.id, node.id) in inv._t0

    def test_post_no_budget_returns_true(self):
        inv = LatencyBound()
        br = make_br()
        node = make_node()
        inv.pre(br, node)
        result = inv.post(br, node, {})
        assert result is True

    def test_post_clears_t0(self):
        inv = LatencyBound()
        br = make_br()
        node = make_node()
        inv.pre(br, node)
        inv.post(br, node, {})
        assert (br.id, node.id) not in inv._t0

    def test_post_within_budget_returns_true(self):
        inv = LatencyBound()
        br = make_br()
        # Use WithTimeout as node.m so latency_budget_ms is set (MorphismAdapter is frozen)
        from lionagi.beta.core.wrappers import WithTimeout

        inner = make_node().m
        m = WithTimeout(inner, timeout_ms=10000)
        node = make_node()
        node.m = m
        inv.pre(br, node)
        result = inv.post(br, node, {})
        assert result is True

    def test_post_exceeds_budget_returns_false(self):
        inv = LatencyBound()
        br = make_br()
        # WithTimeout sets latency_budget_ms; use a tiny budget of 1 ms
        from lionagi.beta.core.wrappers import WithTimeout

        inner_m = make_node().m
        m = WithTimeout(inner_m, timeout_ms=1)
        node = make_node()
        node.m = m
        # Manually plant a t0 that is 1 second in the past
        inv._t0[(br.id, node.id)] = time.perf_counter() - 1.0
        result = inv.post(br, node, {})
        assert result is False

    def test_post_without_pre_still_returns_true_when_no_budget(self):
        inv = LatencyBound()
        br = make_br()
        node = make_node()
        # No pre called — t0 missing, no budget → True
        result = inv.post(br, node, {})
        assert result is True


# ---------------------------------------------------------------------------
# ResultShape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_pre_always_true(self):
        inv = ResultShape()
        node = make_node()
        assert inv.pre(make_br(), node) is True

    def test_post_no_schema_no_keys_returns_true(self):
        inv = ResultShape()
        node = make_node()
        assert inv.post(make_br(), node, {"anything": 1}) is True

    def test_post_result_keys_all_present(self):
        inv = ResultShape()
        node = make_node_with_baseop(result_keys={"a", "b"})
        assert inv.post(make_br(), node, {"a": 1, "b": 2, "c": 3}) is True

    def test_post_result_keys_missing_returns_false(self):
        inv = ResultShape()
        node = make_node_with_baseop(result_keys={"a", "b"})
        assert inv.post(make_br(), node, {"a": 1}) is False

    def test_post_result_keys_empty_required_returns_true(self):
        inv = ResultShape()
        node = make_node_with_baseop(result_keys=set())
        assert inv.post(make_br(), node, {}) is True

    def test_post_result_not_dict_with_keys_required_returns_false(self):
        inv = ResultShape()
        node = make_node_with_baseop(result_keys={"a"})
        assert inv.post(make_br(), node, [1, 2, 3]) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LenientIPU
# ---------------------------------------------------------------------------


class TestLenientIPU:
    def test_before_node_no_raise_on_pre_violation(self, caplog):
        class FailingInv:
            name = "FailInv"

            def pre(self, br, node):
                return False

            def post(self, br, node, result):
                return True

        ipu = LenientIPU([FailingInv()])
        br = make_br()
        node = make_node()
        with caplog.at_level(logging.WARNING):
            run(ipu.before_node(br, node))
        assert "pre-violation" in caplog.text

    def test_after_node_no_raise_on_post_violation(self, caplog):
        class FailingInv:
            name = "FailInv"

            def pre(self, br, node):
                return True

            def post(self, br, node, result):
                return False

        ipu = LenientIPU([FailingInv()])
        br = make_br()
        node = make_node()
        with caplog.at_level(logging.WARNING):
            run(ipu.after_node(br, node, {}))
        assert "post-violation" in caplog.text

    def test_after_node_skips_result_dependent_on_error(self):
        class ResultShapeInv:
            name = "ResultShape"  # matches _RESULT_DEPENDENT_NAMES

            def pre(self, br, node):
                return True

            def post(self, br, node, result):
                raise AssertionError("Should not be called")

        ipu = LenientIPU([ResultShapeInv()])
        br = make_br()
        node = make_node()
        # Should not raise even though post would raise
        run(ipu.after_node(br, node, {}, error=ValueError("oops")))

    def test_on_observation_logs_debug(self, caplog):
        ipu = LenientIPU([])
        obs = Observation(what="test_event", payload={"key": "val"})
        with caplog.at_level(logging.DEBUG, logger="lionagi.beta.core.ipu"):
            run(ipu.on_observation(obs))
        assert "test_event" in caplog.text

    def test_before_node_passes_all_invariants(self):
        calls = []

        class TrackInv:
            name = "TrackInv"

            def pre(self, br, node):
                calls.append("pre")
                return True

            def post(self, br, node, result):
                return True

        ipu = LenientIPU([TrackInv(), TrackInv()])
        run(ipu.before_node(make_br(), make_node()))
        assert calls == ["pre", "pre"]


# ---------------------------------------------------------------------------
# StrictIPU
# ---------------------------------------------------------------------------


class TestStrictIPU:
    def test_before_node_raises_on_pre_violation(self):
        class FailingInv:
            name = "FailInv"

            def pre(self, br, node):
                return False

            def post(self, br, node, result):
                return True

        ipu = StrictIPU([FailingInv()])
        with pytest.raises(AssertionError, match="pre"):
            run(ipu.before_node(make_br(), make_node()))

    def test_after_node_raises_on_post_violation(self):
        class FailingInv:
            name = "FailInv"

            def pre(self, br, node):
                return True

            def post(self, br, node, result):
                return False

        ipu = StrictIPU([FailingInv()])
        with pytest.raises(AssertionError, match="post"):
            run(ipu.after_node(make_br(), make_node(), {}))

    def test_after_node_skips_result_dependent_on_error(self):
        class ResultShapeInv:
            name = "ResultShape"

            def pre(self, br, node):
                return True

            def post(self, br, node, result):
                return False  # would fail

        ipu = StrictIPU([ResultShapeInv()])
        # Should NOT raise because error is set
        run(ipu.after_node(make_br(), make_node(), {}, error=ValueError("err")))

    def test_before_node_passes_when_all_invariants_pass(self):
        class PassInv:
            name = "PassInv"

            def pre(self, br, node):
                return True

            def post(self, br, node, result):
                return True

        ipu = StrictIPU([PassInv()])
        run(ipu.before_node(make_br(), make_node()))  # no exception

    def test_inherits_on_observation_from_lenient(self, caplog):
        ipu = StrictIPU([])
        obs = Observation(what="strict_event")
        with caplog.at_level(logging.DEBUG, logger="lionagi.beta.core.ipu"):
            run(ipu.on_observation(obs))
        assert "strict_event" in caplog.text


# ---------------------------------------------------------------------------
# default_invariants
# ---------------------------------------------------------------------------


class TestDefaultInvariants:
    def test_returns_list_of_three(self):
        invs = default_invariants()
        assert len(invs) == 3

    def test_contains_policy_gate_present(self):
        invs = default_invariants()
        names = [i.name for i in invs]
        assert "PolicyGatePresent" in names

    def test_contains_latency_bound(self):
        invs = default_invariants()
        names = [i.name for i in invs]
        assert "LatencyBound" in names

    def test_contains_result_shape(self):
        invs = default_invariants()
        names = [i.name for i in invs]
        assert "ResultShape" in names

    def test_returns_fresh_instances_each_call(self):
        inv1 = default_invariants()
        inv2 = default_invariants()
        assert inv1 is not inv2
