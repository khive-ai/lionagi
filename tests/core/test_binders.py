# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.core.binders: _build_call_kwargs and BoundOp."""

from __future__ import annotations

import asyncio

import pytest

from lionagi.core.binders import BoundOp, _build_call_kwargs
from lionagi.core.morphism import MorphismAdapter
from lionagi.core.types import Principal

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


def make_br(ctx=None):
    p = Principal(ctx=ctx or {})
    return p


# ---------------------------------------------------------------------------
# _build_call_kwargs
# ---------------------------------------------------------------------------


class TestBuildCallKwargs:
    def test_bind_pulls_from_ctx(self):
        br = make_br(ctx={"token": "abc"})
        result = _build_call_kwargs(
            br=br,
            runtime_kw={},
            bind={"auth": "token"},
            defaults={},
        )
        assert result["auth"] == "abc"

    def test_bind_skips_missing_ctx_key(self):
        br = make_br(ctx={})
        result = _build_call_kwargs(
            br=br,
            runtime_kw={},
            bind={"auth": "token"},  # "token" not in ctx
            defaults={},
        )
        assert "auth" not in result

    def test_defaults_fill_missing_keys(self):
        br = make_br(ctx={})
        result = _build_call_kwargs(
            br=br,
            runtime_kw={},
            bind={},
            defaults={"timeout": 30},
        )
        assert result["timeout"] == 30

    def test_defaults_do_not_override_ctx_bind(self):
        br = make_br(ctx={"token": "from_ctx"})
        result = _build_call_kwargs(
            br=br,
            runtime_kw={},
            bind={"auth": "token"},
            defaults={"auth": "fallback"},
        )
        assert result["auth"] == "from_ctx"

    def test_runtime_kw_overrides_everything(self):
        br = make_br(ctx={"token": "ctx_val"})
        result = _build_call_kwargs(
            br=br,
            runtime_kw={"auth": "runtime_val"},
            bind={"auth": "token"},
            defaults={"auth": "default_val"},
        )
        assert result["auth"] == "runtime_val"

    def test_priority_order(self):
        # ctx via bind < defaults < runtime_kw
        br = make_br(ctx={"src": "ctx"})
        result = _build_call_kwargs(
            br=br,
            runtime_kw={"x": "rt"},
            bind={"x": "src"},
            defaults={"x": "def", "y": "def_y"},
        )
        # x: bind pulled "ctx" from br.ctx["src"], then runtime_kw overrides to "rt"
        assert result["x"] == "rt"
        # y: not in bind or runtime, comes from defaults
        assert result["y"] == "def_y"

    def test_empty_all_returns_empty_dict(self):
        br = make_br(ctx={})
        result = _build_call_kwargs(br=br, runtime_kw={}, bind={}, defaults={})
        assert result == {}


# ---------------------------------------------------------------------------
# BoundOp
# ---------------------------------------------------------------------------


class TestBoundOp:
    def test_init_inherits_contract(self):
        inner = make_adapter(requires=["fs.read"], provides=["data"])
        op = BoundOp(inner, bind={}, defaults={})
        assert "fs.read" in op.requires
        assert "data" in op.provides

    def test_pre_delegates_with_bound_kwargs(self):
        # MorphismAdapter is a msgspec.Struct and cannot have attributes monkey-patched.
        # Use a mutable BaseOp subclass to capture the kwargs passed by BoundOp.
        from lionagi.core.wrappers import BaseOp

        received = {}

        class CapturingOp(BaseOp):
            async def pre(self, br, **kw):
                received["pre_kw"] = dict(kw)
                return True

            async def apply(self, br, **kw):
                return {}

        inner = CapturingOp()
        br = make_br(ctx={"token": "abc"})
        op = BoundOp(inner, bind={"auth": "token"}, defaults={})
        run(op.pre(br))
        assert received.get("pre_kw", {}).get("auth") == "abc"

    def test_apply_injects_ctx_values(self):
        received_kw = {}

        async def fn(br, **kw):
            received_kw.update(kw)
            return {"ok": True}

        inner = make_adapter(fn)
        br = make_br(ctx={"api_key": "secret"})
        op = BoundOp(inner, bind={"key": "api_key"}, defaults={})
        run(op.apply(br))
        assert received_kw.get("key") == "secret"

    def test_apply_uses_defaults(self):
        received_kw = {}

        async def fn(br, **kw):
            received_kw.update(kw)
            return {}

        inner = make_adapter(fn)
        br = make_br(ctx={})
        op = BoundOp(inner, bind={}, defaults={"timeout": 60})
        run(op.apply(br))
        assert received_kw.get("timeout") == 60

    def test_apply_runtime_overrides_default(self):
        received_kw = {}

        async def fn(br, **kw):
            received_kw.update(kw)
            return {}

        inner = make_adapter(fn)
        br = make_br(ctx={})
        op = BoundOp(inner, bind={}, defaults={"x": "default"})
        run(op.apply(br, x="runtime"))
        assert received_kw.get("x") == "runtime"

    def test_post_delegates_to_inner(self):
        inner = make_adapter()
        op = BoundOp(inner, bind={}, defaults={})
        result = run(op.post(make_br(), {"ok": True}))
        assert result is True

    def test_no_bind_no_defaults(self):
        called = {}

        async def fn(br, **kw):
            called["kw"] = kw
            return {}

        inner = make_adapter(fn)
        br = make_br(ctx={"irrelevant": "data"})
        op = BoundOp(inner)  # bind=None, defaults=None
        run(op.apply(br))
        # No binding → no ctx keys injected
        assert "irrelevant" not in called.get("kw", {})
