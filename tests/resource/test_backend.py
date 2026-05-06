# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/resource/backend.py — ResourceConfig, Normalized, Calling, ResourceBackend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from lionagi._errors import ValidationError
from lionagi.service.backend import (
    Calling,
    Normalized,
    ResourceBackend,
    ResourceConfig,
    _get_schema_field_keys,
)
from lionagi.protocols.generic.event import Event, EventStatus

# ---------------------------------------------------------------------------
# _get_schema_field_keys (cache)
# ---------------------------------------------------------------------------


class TestGetSchemaFieldKeys:
    def test_returns_field_names(self):
        class SampleModel(BaseModel):
            alpha: str
            beta: int

        keys = _get_schema_field_keys(SampleModel)
        assert "alpha" in keys
        assert "beta" in keys

    def test_cached(self):
        class M(BaseModel):
            x: int

        k1 = _get_schema_field_keys(M)
        k2 = _get_schema_field_keys(M)
        assert k1 is k2


# ---------------------------------------------------------------------------
# ResourceConfig
# ---------------------------------------------------------------------------


class TestResourceConfig:
    def test_basic_creation(self):
        cfg = ResourceConfig(provider="openai", name="gpt-4")
        assert cfg.provider == "openai"
        assert cfg.name == "gpt-4"

    def test_default_timeout_and_retries(self):
        cfg = ResourceConfig(provider="openai", name="gpt-4")
        assert cfg.timeout == 300
        assert cfg.max_retries == 3

    def test_extra_kwargs_absorbed(self):
        cfg = ResourceConfig(provider="p", name="n", some_extra="yes")
        assert "some_extra" in cfg.kwargs

    def test_version_and_tags(self):
        cfg = ResourceConfig(provider="p", name="n", version="v1", tags=["a", "b"])
        assert cfg.version == "v1"
        assert "a" in cfg.tags

    def test_request_options_accepts_model_type(self):
        class MyReqModel(BaseModel):
            q: str

        cfg = ResourceConfig(provider="p", name="n", request_options=MyReqModel)
        assert cfg.request_options is MyReqModel

    def test_request_options_accepts_instance(self):
        class MyReqModel(BaseModel):
            q: str

        cfg = ResourceConfig(provider="p", name="n", request_options=MyReqModel(q="x"))
        assert cfg.request_options is MyReqModel

    def test_request_options_invalid_raises(self):
        with pytest.raises(Exception):
            ResourceConfig(provider="p", name="n", request_options="bad")

    def test_validate_payload_no_options(self):
        cfg = ResourceConfig(provider="p", name="n")
        result = cfg.validate_payload({"any": "data"})
        assert result == {"any": "data"}

    def test_validate_payload_with_model(self):
        class Req(BaseModel):
            x: int

        cfg = ResourceConfig(provider="p", name="n", request_options=Req)
        result = cfg.validate_payload({"x": 5})
        assert result["x"] == 5

    def test_validate_payload_invalid_raises(self):
        class Req(BaseModel):
            x: int

        cfg = ResourceConfig(provider="p", name="n", request_options=Req)
        with pytest.raises(ValueError):
            cfg.validate_payload({"x": "not_int"})


# ---------------------------------------------------------------------------
# Normalized
# ---------------------------------------------------------------------------


class TestNormalized:
    def test_success(self):
        n = Normalized(status="success", data="hello")
        assert n.status == "success"
        assert n.data == "hello"

    def test_error(self):
        n = Normalized(status="error", data=None, error="oops")
        assert n.status == "error"
        assert n.error == "oops"

    def test_with_serialized(self):
        n = Normalized(status="success", data={}, serialized={"raw": 1})
        assert n.serialized == {"raw": 1}

    def test_with_metadata(self):
        n = Normalized(status="success", data=None, metadata={"tokens": 10})
        assert n.metadata["tokens"] == 10


# ---------------------------------------------------------------------------
# Concrete Calling / ResourceBackend stubs for testing
# ---------------------------------------------------------------------------


class MockCalling(Calling):
    @property
    def call_args(self) -> dict:
        return {"payload": self.payload}

    async def invoke(self):
        response = await self._invoke()
        self.execution.response = response
        self.execution.status = EventStatus.COMPLETED

    def assert_is_normalized(self):
        pass  # allow in tests


class MockBackend(ResourceBackend):
    @property
    def event_type(self):
        return MockCalling

    async def call(self, **kw) -> Normalized:
        return Normalized(status="success", data="result_data")


def make_backend():
    cfg = ResourceConfig(provider="test", name="mock")
    return MockBackend(config=cfg)


def make_calling(backend=None, payload=None):
    if backend is None:
        backend = make_backend()
    return MockCalling(backend=backend, payload=payload or {"key": "val"})


# ---------------------------------------------------------------------------
# ResourceBackend properties
# ---------------------------------------------------------------------------


class TestResourceBackendProperties:
    def test_provider(self):
        b = make_backend()
        assert b.provider == "test"

    def test_name(self):
        b = make_backend()
        assert b.name == "mock"

    def test_version_none(self):
        b = make_backend()
        assert b.version is None

    def test_tags_empty(self):
        b = make_backend()
        assert b.tags == set()

    def test_normalize_response(self):
        b = make_backend()
        n = b.normalize_response("raw")
        assert n.status == "success"
        assert n.data == "raw"

    def test_normalize_chunk(self):
        b = make_backend()
        n = b.normalize_chunk({"delta": "x"})
        assert n.status == "success"

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self):
        b = make_backend()
        with pytest.raises(NotImplementedError):
            await b.stream()


# ---------------------------------------------------------------------------
# Calling
# ---------------------------------------------------------------------------


class TestCalling:
    def test_payload_stored(self):
        c = make_calling(payload={"x": 1})
        assert c.payload == {"x": 1}

    def test_call_args(self):
        c = make_calling(payload={"x": 1})
        assert c.call_args == {"payload": {"x": 1}}

    def test_stream_args_excludes_keys(self):
        class SpecialCalling(MockCalling):
            @property
            def call_args(self):
                return {"skip_payload_creation": True, "messages": []}

        b = make_backend()
        c = SpecialCalling(backend=b, payload={})
        assert "skip_payload_creation" not in c.stream_args
        assert "messages" in c.stream_args

    def test_response_unset_initially(self):
        from lionagi.ln.types._sentinel import Unset, is_sentinel

        c = make_calling()
        assert is_sentinel(c.response) or c.response is Unset

    @pytest.mark.asyncio
    async def test_invoke_calls_backend(self):
        b = make_backend()
        c = make_calling(backend=b, payload={})
        response = await c._invoke()
        assert response.status == "success"
        assert response.data == "result_data"

    def test_assert_is_normalized_passthrough(self):
        c = make_calling()
        c.assert_is_normalized()  # MockCalling overrides to pass

    def test_create_pre_invoke_hook(self):
        from lionagi.service.hooks import HookRegistry

        c = make_calling()
        registry = MagicMock(spec=HookRegistry)
        registry.hooks = []
        # Shouldn't raise
        c.create_pre_invoke_hook(registry)
        assert c._pre_invoke_hook_event is not None

    def test_create_post_invoke_hook(self):
        from lionagi.service.hooks import HookRegistry

        c = make_calling()
        registry = MagicMock(spec=HookRegistry)
        registry.hooks = []
        c.create_post_invoke_hook(registry)
        assert c._post_invoke_hook_event is not None
