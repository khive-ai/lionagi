# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.core.wrappers: register, BaseOp, inherit_contract,
maybe_await, CtxSet, WithRetry, WithTimeout, OpThenPatch."""

from __future__ import annotations

import asyncio

import pytest

from lionagi.core.morphism import MorphismAdapter
from lionagi.core.types import Principal
from lionagi.core.wrappers import (
    MORPH_REGISTRY,
    BaseOp,
    CtxSet,
    OpThenPatch,
    WithRetry,
    WithTimeout,
    inherit_contract,
    maybe_await,
    register,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_adapter(fn=None, name="test", requires=(), provides=()):
    if fn is None:

        async def fn(br, **kw):
            return {"ok": True}

    return MorphismAdapter.wrap(fn, name=name, requires=requires, provides=provides)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_adds_to_registry(self):
        @register
        class MyOp(BaseOp):
            name = "test.register.myop"

        assert "test.register.myop" in MORPH_REGISTRY
        assert MORPH_REGISTRY["test.register.myop"] is MyOp

    def test_register_returns_class(self):
        class AnotherOp(BaseOp):
            name = "test.register.another"

        result = register(AnotherOp)
        assert result is AnotherOp

    def test_builtin_ops_registered(self):
        assert "ctx.set" in MORPH_REGISTRY
        assert "with.retry" in MORPH_REGISTRY
        assert "with.timeout" in MORPH_REGISTRY
        assert "op.then_patch" in MORPH_REGISTRY


# ---------------------------------------------------------------------------
# BaseOp
# ---------------------------------------------------------------------------


class TestBaseOp:
    def test_pre_returns_true(self):
        op = BaseOp()
        assert run(op.pre(None)) is True

    def test_post_returns_true(self):
        op = BaseOp()
        assert run(op.post(None, {})) is True

    def test_apply_raises_not_implemented(self):
        op = BaseOp()
        with pytest.raises(NotImplementedError):
            run(op.apply(None))

    def test_default_attributes(self):
        op = BaseOp()
        assert op.requires == frozenset()
        assert op.provides == frozenset()
        assert op.io is False
        assert op.latency_budget_ms is None
        assert op.result_keys is None
        assert op.result_schema is None
        assert op.ctx_writes is None
        assert op.result_bytes_limit is None


# ---------------------------------------------------------------------------
# inherit_contract
# ---------------------------------------------------------------------------


class TestInheritContract:
    def test_copies_requires(self):
        inner = make_adapter(requires=["fs.read"])
        target = BaseOp()
        inherit_contract(target, inner)
        assert "fs.read" in target.requires

    def test_copies_provides(self):
        inner = make_adapter(provides=["result.ready"])
        target = BaseOp()
        inherit_contract(target, inner)
        assert "result.ready" in target.provides

    def test_copies_latency_budget(self):
        class M:
            requires = frozenset()
            provides = frozenset()
            io = False
            ctx_writes = None
            result_schema = None
            result_keys = None
            result_bytes_limit = None
            latency_budget_ms = 500

        target = BaseOp()
        inherit_contract(target, M())
        assert target.latency_budget_ms == 500

    def test_copies_result_keys(self):
        class M:
            requires = frozenset()
            provides = frozenset()
            io = False
            ctx_writes = None
            result_schema = None
            result_keys = {"key1", "key2"}
            result_bytes_limit = None
            latency_budget_ms = None

        target = BaseOp()
        inherit_contract(target, M())
        assert target.result_keys == {"key1", "key2"}

    def test_none_inner_attributes_set_to_none(self):
        inner = make_adapter()
        target = BaseOp()
        inherit_contract(target, inner)
        assert target.latency_budget_ms is None
        assert target.result_keys is None


# ---------------------------------------------------------------------------
# maybe_await
# ---------------------------------------------------------------------------


class TestMaybeAwait:
    def test_awaitable_is_awaited(self):
        async def coro():
            return 42

        assert run(maybe_await(coro())) == 42

    def test_non_awaitable_returned_as_is(self):
        assert run(maybe_await(99)) == 99

    def test_none_returned_as_is(self):
        assert run(maybe_await(None)) is None

    def test_string_returned_as_is(self):
        assert run(maybe_await("hello")) == "hello"


# ---------------------------------------------------------------------------
# CtxSet
# ---------------------------------------------------------------------------


class TestCtxSet:
    def _make_br(self):
        return Principal()

    def test_pre_true_when_values_subset_of_allowed(self):
        op = CtxSet(values={"a": 1, "b": 2}, allowed_keys={"a", "b", "c"})
        assert run(op.pre(self._make_br())) is True

    def test_pre_false_when_values_not_subset(self):
        op = CtxSet(values={"a": 1, "x": 99}, allowed_keys={"a", "b"})
        assert run(op.pre(self._make_br())) is False

    def test_apply_returns_values_dict(self):
        op = CtxSet(values={"a": 1, "b": 2}, allowed_keys={"a", "b"})
        result = run(op.apply(self._make_br()))
        assert result == {"a": 1, "b": 2}

    def test_post_true_when_all_values_present(self):
        op = CtxSet(values={"a": 1}, allowed_keys={"a"})
        assert run(op.post(self._make_br(), {"a": 1})) is True

    def test_post_false_when_value_missing(self):
        op = CtxSet(values={"a": 1}, allowed_keys={"a"})
        assert run(op.post(self._make_br(), {})) is False

    def test_post_false_when_value_wrong(self):
        op = CtxSet(values={"a": 1}, allowed_keys={"a"})
        assert run(op.post(self._make_br(), {"a": 2})) is False

    def test_provides_reflects_values(self):
        op = CtxSet(values={"x": 10, "y": 20}, allowed_keys={"x", "y"})
        assert "x" in op.provides
        assert "y" in op.provides

    def test_empty_values(self):
        op = CtxSet(values={}, allowed_keys=set())
        assert run(op.pre(self._make_br())) is True
        assert run(op.apply(self._make_br())) == {}
        assert run(op.post(self._make_br(), {})) is True


# ---------------------------------------------------------------------------
# WithRetry
# ---------------------------------------------------------------------------


class TestWithRetry:
    def _make_br(self):
        return Principal()

    def test_attempts_less_than_one_raises(self):
        inner = make_adapter()
        with pytest.raises(ValueError):
            WithRetry(inner, attempts=0)

    def test_inherits_contract(self):
        inner = make_adapter(requires=["fs.read"], provides=["data"])
        op = WithRetry(inner, attempts=2)
        assert "fs.read" in op.requires
        assert "data" in op.provides

    def test_apply_succeeds_immediately(self):
        async def fn(br, **kw):
            return {"val": 1}

        inner = make_adapter(fn)
        op = WithRetry(inner, attempts=3)
        result = run(op.apply(self._make_br()))
        assert result == {"val": 1}

    def test_apply_retries_on_failure(self):
        call_count = 0

        async def fn(br, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return {"ok": True}

        inner = make_adapter(fn)
        # base_delay must be > 0 per retry() contract
        op = WithRetry(inner, attempts=3, base_delay=0.001, jitter=0.0)
        result = run(op.apply(self._make_br()))
        assert result == {"ok": True}
        assert call_count == 3

    def test_pre_delegates_to_inner(self):
        # Use a BaseOp subclass (mutable) rather than MorphismAdapter (msgspec Struct)
        pre_calls = []

        class TrackingOp(BaseOp):
            async def pre(self, br, **kw):
                pre_calls.append("called")
                return True

            async def apply(self, br, **kw):
                return {"ok": True}

        inner = TrackingOp()
        op = WithRetry(inner, attempts=2)
        run(op.pre(self._make_br()))
        assert "called" in pre_calls

    def test_post_delegates_to_inner(self):
        inner = make_adapter()
        op = WithRetry(inner, attempts=2)
        assert run(op.post(self._make_br(), {"ok": True})) is True


# ---------------------------------------------------------------------------
# WithTimeout
# ---------------------------------------------------------------------------


class TestWithTimeout:
    def _make_br(self):
        return Principal()

    def test_timeout_less_than_one_raises(self):
        inner = make_adapter()
        with pytest.raises(ValueError):
            WithTimeout(inner, timeout_ms=0)

    def test_latency_budget_set(self):
        inner = make_adapter()
        op = WithTimeout(inner, timeout_ms=500)
        assert op.latency_budget_ms == 500

    def test_inherits_contract(self):
        inner = make_adapter(requires=["net.out"], provides=["resp"])
        op = WithTimeout(inner, timeout_ms=100)
        assert "net.out" in op.requires
        assert "resp" in op.provides

    def test_apply_delegates(self):
        async def fn(br, **kw):
            return {"data": "ok"}

        inner = make_adapter(fn)
        op = WithTimeout(inner, timeout_ms=1000)
        result = run(op.apply(self._make_br()))
        assert result == {"data": "ok"}

    def test_pre_delegates(self):
        inner = make_adapter()
        op = WithTimeout(inner, timeout_ms=100)
        assert run(op.pre(self._make_br())) is True

    def test_post_delegates(self):
        inner = make_adapter()
        op = WithTimeout(inner, timeout_ms=100)
        assert run(op.post(self._make_br(), {})) is True


# ---------------------------------------------------------------------------
# OpThenPatch
# ---------------------------------------------------------------------------


class TestOpThenPatch:
    def _make_br(self):
        return Principal()

    def test_patch_dict_maps_keys(self):
        async def fn(br, **kw):
            return {"src": 42}

        inner = make_adapter(fn)
        op = OpThenPatch(inner, patch={"src": "dst"})
        result = run(op.apply(self._make_br()))
        assert "dst" in result
        assert result["dst"] == 42

    def test_patch_list_identity_mapping(self):
        async def fn(br, **kw):
            return {"x": 1, "y": 2}

        inner = make_adapter(fn)
        op = OpThenPatch(inner, patch=["x", "y"])
        result = run(op.apply(self._make_br()))
        assert result["x"] == 1
        assert result["y"] == 2

    def test_patch_preserves_original_keys(self):
        async def fn(br, **kw):
            return {"a": 10, "b": 20}

        inner = make_adapter(fn)
        op = OpThenPatch(inner, patch={"a": "c"})
        result = run(op.apply(self._make_br()))
        # original key still present
        assert "a" in result
        # mapped key also present
        assert "c" in result

    def test_ctx_writes_set_from_patch_values(self):
        inner = make_adapter()
        op = OpThenPatch(inner, patch={"src": "dst_ctx_key"})
        assert "dst_ctx_key" in op.ctx_writes

    def test_missing_src_key_not_added(self):
        async def fn(br, **kw):
            return {"other": 5}

        inner = make_adapter(fn)
        op = OpThenPatch(inner, patch={"missing": "dst"})
        result = run(op.apply(self._make_br()))
        # "dst" should not appear since "missing" not in result
        assert "dst" not in result

    def test_pre_delegates(self):
        inner = make_adapter()
        op = OpThenPatch(inner, patch={})
        assert run(op.pre(self._make_br())) is True

    def test_post_delegates(self):
        inner = make_adapter()
        op = OpThenPatch(inner, patch={})
        assert run(op.post(self._make_br(), {})) is True

    def test_inherits_contract_from_inner(self):
        inner = make_adapter(requires=["fs.read"], provides=["data"])
        op = OpThenPatch(inner, patch={"data": "ctx_data"})
        assert "fs.read" in op.requires
        assert "data" in op.provides
