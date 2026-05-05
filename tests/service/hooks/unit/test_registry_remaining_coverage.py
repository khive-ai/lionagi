# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for hook_registry.py uncovered branches."""

from __future__ import annotations

import pytest

from lionagi.service.hooks._types import HookEventTypes
from lionagi.service.hooks.hook_registry import HookRegistry, _normalize_hook_key
from tests.service.hooks.conftest import FakeEvent


class TestNormalizeHookKey:
    def test_canonical_member_passthrough(self):
        assert (
            _normalize_hook_key(HookEventTypes.PreInvocation)
            is HookEventTypes.PreInvocation
        )

    def test_alias_pre_invoke(self):
        assert _normalize_hook_key("pre_invoke") is HookEventTypes.PreInvocation

    def test_alias_post_invoke(self):
        assert _normalize_hook_key("post_invoke") is HookEventTypes.PostInvocation

    def test_alias_pre_event_create(self):
        assert _normalize_hook_key("pre_event_create") is HookEventTypes.PreEventCreate

    def test_alias_pre_event_create_hook(self):
        assert (
            _normalize_hook_key("pre_event_create_hook")
            is HookEventTypes.PreEventCreate
        )

    def test_alias_error_handling(self):
        assert _normalize_hook_key("error_handling") is HookEventTypes.ErrorHandling

    def test_unknown_string_tries_enum_value_fallback(self):
        # "pre_invocation" is the actual enum value for PreInvocation
        result = _normalize_hook_key("pre_invocation")
        assert result is HookEventTypes.PreInvocation

    def test_unrecognized_string_returns_unchanged(self):
        result = _normalize_hook_key("totally_unknown_key")
        assert result == "totally_unknown_key"

    def test_non_string_non_enum_passthrough(self):
        result = _normalize_hook_key(42)
        assert result == 42


class TestErrorHandlerSetter:
    def test_registers_callable_as_error_handler(self):
        registry = HookRegistry()

        async def my_handler(ev, **kw):
            pass

        registry.error_handler(my_handler)
        assert HookEventTypes.ErrorHandling in registry._hooks
        assert registry._hooks[HookEventTypes.ErrorHandling] is my_handler

    def test_overwrite_emits_warning(self):
        registry = HookRegistry()

        async def h1(ev, **kw):
            pass

        async def h2(ev, **kw):
            pass

        registry.error_handler(h1)
        with pytest.warns(UserWarning):
            registry.error_handler(h2)
        assert registry._hooks[HookEventTypes.ErrorHandling] is h2


class TestCallStreamHandlerPath:
    @pytest.mark.anyio
    async def test_call_with_stream_handler(self):
        results = []

        async def my_handler(ev, ct, chunk, **kw):
            results.append((ct, chunk))
            return "handled"

        registry = HookRegistry(stream_handlers={"text": my_handler})
        result = await registry.handle_streaming_chunk(
            "text", "hello chunk", exit=False
        )
        assert result[0] == "handled"
        assert results == [(("text", "hello chunk"))]

    @pytest.mark.anyio
    async def test_call_stream_handler_missing_raises(self):
        registry = HookRegistry()
        with pytest.raises(RuntimeError, match="No stream handler registered"):
            await registry._call_stream_handler("unknown_type", "chunk", None)

    @pytest.mark.anyio
    async def test_can_handle_stream_type_true(self):
        async def h(ev, ct, chunk, **kw):
            pass

        registry = HookRegistry(stream_handlers={"mytype": h})
        assert registry._can_handle(ct_="mytype") is True

    @pytest.mark.anyio
    async def test_can_handle_stream_type_missing_false(self):
        registry = HookRegistry()
        assert registry._can_handle(ct_="missing") is False


class TestErrorHandlingHook:
    @pytest.mark.anyio
    async def test_error_handling_success(self):
        async def eh(ev, **kw):
            return "recovered"

        registry = HookRegistry()
        registry.error_handler(eh)
        event = FakeEvent()
        result, should_exit, status = await registry.error_handling(event, exit=False)
        assert result == "recovered"
        assert should_exit is False

    @pytest.mark.anyio
    async def test_error_handling_exception_returns_failed(self):
        async def eh(ev, **kw):
            raise ValueError("unhandled")

        registry = HookRegistry()
        registry.error_handler(eh)
        event = FakeEvent()
        from lionagi.protocols.types import EventStatus

        result, should_exit, status = await registry.error_handling(event, exit=False)
        assert isinstance(result, ValueError)
        assert status == EventStatus.FAILED

    @pytest.mark.anyio
    async def test_error_handling_cancellation(self, patch_cancellation):
        CancelledExc = patch_cancellation

        async def eh(ev, **kw):
            raise CancelledExc("cancelled")

        registry = HookRegistry()
        registry.error_handler(eh)
        event = FakeEvent()
        from lionagi.protocols.types import EventStatus

        result, should_exit, status = await registry.error_handling(event, exit=False)
        assert should_exit is True
        assert status == EventStatus.CANCELLED


class TestCallDispatchErrorHandling:
    @pytest.mark.anyio
    async def test_call_dispatches_error_handling(self):
        received = []

        async def eh(ev, **kw):
            received.append(ev)
            return "ok"

        registry = HookRegistry()
        registry.error_handler(eh)
        event = FakeEvent(eid="test-eid", created_at=999.0)
        (result, should_exit, status), meta = await registry.call(
            event,
            hook_type=HookEventTypes.ErrorHandling,
        )
        assert result == "ok"
        assert meta["event_id"] == str(event.id)
        assert meta["event_created_at"] == event.created_at
        assert received[0] is event


class TestCanHandle:
    def test_can_handle_hook_type_registered(self):
        async def h(ev, **kw):
            pass

        registry = HookRegistry(hooks={HookEventTypes.PreInvocation: h})
        assert registry._can_handle(ht_=HookEventTypes.PreInvocation) is True

    def test_can_handle_hook_type_not_registered(self):
        registry = HookRegistry()
        assert registry._can_handle(ht_=HookEventTypes.PreInvocation) is False

    def test_can_handle_neither_returns_false(self):
        registry = HookRegistry()
        assert registry._can_handle() is False
