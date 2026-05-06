# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.core.morphism: Morphism, MorphismAdapter."""

from __future__ import annotations

import asyncio

import pytest

from lionagi.core.morphism import Morphism, MorphismAdapter, MorphismLike

# ---------------------------------------------------------------------------
# Morphism
# ---------------------------------------------------------------------------


class TestMorphism:
    def test_basic_construction(self):
        m = Morphism(name="test_op")
        assert m.name == "test_op"
        assert m.requires == frozenset()
        assert m.provides == frozenset()
        assert m.required_rights is None

    def test_with_requires_and_provides(self):
        m = Morphism(
            name="op",
            requires=frozenset({"fs.read"}),
            provides=frozenset({"data.loaded"}),
        )
        assert "fs.read" in m.requires
        assert "data.loaded" in m.provides

    def test_pre_returns_true(self):
        m = Morphism(name="op")
        result = asyncio.run(m.pre(None))
        assert result is True

    def test_post_returns_true(self):
        m = Morphism(name="op")
        result = asyncio.run(m.post(None, {}))
        assert result is True

    def test_apply_raises_not_implemented(self):
        m = Morphism(name="op")
        with pytest.raises(NotImplementedError):
            asyncio.run(m.apply(None))

    def test_required_rights_custom(self):
        m = Morphism(name="op", required_rights={"admin"})
        assert m.required_rights == {"admin"}


# ---------------------------------------------------------------------------
# MorphismAdapter.wrap
# ---------------------------------------------------------------------------


class TestMorphismAdapterWrap:
    def test_wrap_single_callable(self):
        async def fn(br, **kw):
            return {"ok": True}

        adapter = MorphismAdapter.wrap(fn)
        assert adapter.name == "fn"
        assert adapter._fn is fn

    def test_wrap_callable_with_name_kwarg(self):
        async def fn(br, **kw):
            return {}

        adapter = MorphismAdapter.wrap(fn, name="custom_name")
        assert adapter.name == "custom_name"

    def test_wrap_two_positional_args(self):
        async def fn(br, **kw):
            return {}

        adapter = MorphismAdapter.wrap("my_op", fn)
        assert adapter.name == "my_op"
        assert adapter._fn is fn

    def test_wrap_fn_kwarg(self):
        async def fn(br, **kw):
            return {}

        adapter = MorphismAdapter.wrap(fn=fn, name="via_kwarg")
        assert adapter.name == "via_kwarg"

    def test_wrap_no_callable_raises_type_error(self):
        with pytest.raises(TypeError):
            MorphismAdapter.wrap()

    def test_wrap_too_many_positional_raises_type_error(self):
        async def fn(br, **kw):
            return {}

        with pytest.raises(TypeError):
            MorphismAdapter.wrap(fn, fn, fn)

    def test_wrap_with_requires_and_provides(self):
        async def fn(br, **kw):
            return {}

        adapter = MorphismAdapter.wrap(
            fn,
            requires={"fs.read"},
            provides={"data.loaded"},
        )
        assert "fs.read" in adapter.requires
        assert "data.loaded" in adapter.provides

    def test_wrap_lambda_gets_name(self):
        fn = lambda br, **kw: {"x": 1}  # noqa: E731
        adapter = MorphismAdapter.wrap(fn)
        # lambdas have name "<lambda>"
        assert adapter.name == "<lambda>"


# ---------------------------------------------------------------------------
# MorphismAdapter.from_protocol
# ---------------------------------------------------------------------------


class TestMorphismAdapterFromProtocol:
    def test_from_protocol_basic(self):
        class MyMorphism:
            name = "proto_op"
            requires = frozenset({"net.out"})

            async def apply(self, br, **kw):
                return {"done": True}

        obj = MyMorphism()
        adapter = MorphismAdapter.from_protocol(obj)
        assert adapter.name == "proto_op"
        assert "net.out" in adapter.requires
        # from_protocol stores the bound method; use == for bound method comparison
        assert adapter._fn == obj.apply

    def test_from_protocol_copies_provides(self):
        class MyMorphism:
            name = "op"
            requires = frozenset()
            provides = frozenset({"result.ready"})

            async def apply(self, br, **kw):
                return {}

        obj = MyMorphism()
        adapter = MorphismAdapter.from_protocol(obj)
        assert "result.ready" in adapter.provides

    def test_from_protocol_no_provides_defaults_empty(self):
        class MyMorphism:
            name = "op"
            requires = frozenset()

            async def apply(self, br, **kw):
                return {}

        obj = MyMorphism()
        adapter = MorphismAdapter.from_protocol(obj)
        assert adapter.provides == frozenset()


# ---------------------------------------------------------------------------
# MorphismAdapter.apply
# ---------------------------------------------------------------------------


class TestMorphismAdapterApply:
    def test_apply_async_callable(self):
        async def fn(br, **kw):
            return {"result": 42}

        adapter = MorphismAdapter.wrap(fn, name="test")
        result = asyncio.run(adapter.apply(None))
        assert result == {"result": 42}

    def test_apply_passes_kwargs(self):
        async def fn(br, **kw):
            return {"received": kw.get("x")}

        adapter = MorphismAdapter.wrap(fn, name="test")
        result = asyncio.run(adapter.apply(None, x=99))
        assert result["received"] == 99

    def test_apply_sync_callable(self):
        def fn(br, **kw):
            return {"sync": True}

        adapter = MorphismAdapter.wrap(fn, name="sync_test")
        result = asyncio.run(adapter.apply(None))
        assert result == {"sync": True}

    def test_apply_no_fn_raises_runtime_error(self):
        adapter = MorphismAdapter(name="empty")
        with pytest.raises(RuntimeError):
            asyncio.run(adapter.apply(None))
