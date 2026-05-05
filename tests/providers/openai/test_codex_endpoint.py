# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.openai.codex.endpoint.

Covers CodexCLIEndpoint: handlers validation, setter, update_handlers,
copy_runtime_state_to, _runtime_handlers, create_payload, stream(), _call().
All tests are deterministic — stream_codex_cli is fully mocked.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_endpoint(**kwargs):
    from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

    return CodexCLIEndpoint(**kwargs)


# ---------------------------------------------------------------------------
# _validate_handlers — standalone function tests
# ---------------------------------------------------------------------------


class TestValidateHandlers:
    def test_raises_if_not_dict(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_handlers(["on_text"])  # type: ignore[arg-type]

    def test_raises_on_invalid_key(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="Invalid handler key"):
            _validate_handlers({"on_unknown_event": lambda x: x})

    def test_raises_on_non_callable_value(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="callable or None"):
            _validate_handlers({"on_text": "not-a-callable"})

    def test_accepts_valid_callable(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        # Should not raise
        _validate_handlers({"on_text": lambda chunk: None})

    def test_accepts_none_values(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        _validate_handlers({"on_text": None, "on_final": None})


# ---------------------------------------------------------------------------
# CodexCLIEndpoint — init and default handlers
# ---------------------------------------------------------------------------


class TestCodexCLIEndpointInit:
    def test_default_handlers_all_none(self):
        ep = _make_endpoint()
        for v in ep.codex_handlers.values():
            assert v is None

    def test_handlers_keys_are_standard(self):
        ep = _make_endpoint()
        assert set(ep.codex_handlers.keys()) == {
            "on_text",
            "on_tool_use",
            "on_tool_result",
            "on_final",
        }

    def test_init_with_codex_handlers_kwarg(self):
        handler = lambda chunk: None
        ep = _make_endpoint(codex_handlers={"on_text": handler})
        assert ep.codex_handlers["on_text"] is handler

    def test_init_with_invalid_handlers_raises(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        with pytest.raises(ValueError):
            CodexCLIEndpoint(codex_handlers={"bad_key": lambda x: x})


# ---------------------------------------------------------------------------
# codex_handlers setter
# ---------------------------------------------------------------------------


class TestCodexHandlersSetter:
    def test_setter_valid_dict(self):
        ep = _make_endpoint()
        handler = lambda c: None
        ep.codex_handlers = {"on_text": handler}
        assert ep.codex_handlers["on_text"] is handler
        # Unspecified keys reset to None
        assert ep.codex_handlers["on_final"] is None

    def test_setter_invalid_key_raises(self):
        ep = _make_endpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            ep.codex_handlers = {"on_bad_key": lambda c: None}

    def test_setter_resets_previous_handlers(self):
        ep = _make_endpoint(codex_handlers={"on_text": lambda c: None})
        ep.codex_handlers = {}
        assert ep.codex_handlers["on_text"] is None


# ---------------------------------------------------------------------------
# update_handlers
# ---------------------------------------------------------------------------


class TestUpdateHandlers:
    def test_update_merges_new_handler(self):
        ep = _make_endpoint()
        handler = lambda c: None
        ep.update_handlers(on_final=handler)
        assert ep.codex_handlers["on_final"] is handler
        assert ep.codex_handlers["on_text"] is None

    def test_update_invalid_key_raises(self):
        ep = _make_endpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            ep.update_handlers(on_nonexistent=lambda c: None)


# ---------------------------------------------------------------------------
# copy_runtime_state_to
# ---------------------------------------------------------------------------


class TestCopyRuntimeStateTo:
    def test_copies_to_same_type(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        src = _make_endpoint(codex_handlers={"on_text": lambda c: None})
        dst = _make_endpoint()
        src.copy_runtime_state_to(dst)
        assert dst.codex_handlers["on_text"] is src.codex_handlers["on_text"]

    def test_ignores_different_type(self):
        src = _make_endpoint(codex_handlers={"on_text": lambda c: None})
        # A plain object — should not raise, just silently skip
        src.copy_runtime_state_to(object())

    def test_copy_is_independent(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        src = _make_endpoint()
        dst = _make_endpoint()
        src.copy_runtime_state_to(dst)
        # Mutating src after copy should not affect dst
        src.update_handlers(on_text=lambda c: None)
        assert dst.codex_handlers["on_text"] is None


# ---------------------------------------------------------------------------
# _runtime_handlers
# ---------------------------------------------------------------------------


class TestRuntimeHandlers:
    def test_call_kwarg_overrides_instance_handler(self):
        ep = _make_endpoint()
        instance_handler = lambda c: "instance"
        call_handler = lambda c: "call"
        ep.update_handlers(on_text=instance_handler)
        kwargs = {"on_text": call_handler}
        runtime = ep._runtime_handlers(kwargs)
        assert runtime["on_text"] is call_handler
        # kwargs dict consumed
        assert "on_text" not in kwargs

    def test_none_values_filtered_out(self):
        ep = _make_endpoint()
        runtime = ep._runtime_handlers({})
        # All None by default — result dict should be empty
        assert runtime == {}

    def test_non_handler_kwargs_not_consumed(self):
        ep = _make_endpoint()
        kwargs = {"model": "codex-mini", "on_text": lambda c: None}
        runtime = ep._runtime_handlers(kwargs)
        # "model" should remain, "on_text" was consumed
        assert "model" in kwargs
        assert "on_text" not in kwargs


# ---------------------------------------------------------------------------
# create_payload
# ---------------------------------------------------------------------------


class TestCreatePayload:
    def test_from_dict_with_prompt(self):
        ep = _make_endpoint()
        result, headers = ep.create_payload({"prompt": "Fix the bug"})
        assert "request" in result
        assert headers == {}
        req = result["request"]
        assert req.prompt == "Fix the bug"

    def test_from_codex_code_request_object(self):
        from lionagi.providers.openai.codex.models import CodexCodeRequest

        ep = _make_endpoint()
        req_obj = CodexCodeRequest(prompt="Explain async/await")
        result, headers = ep.create_payload(req_obj)
        assert result["request"].prompt == "Explain async/await"

    def test_payload_keys_filtered_to_model_fields(self):
        ep = _make_endpoint()
        result, _ = ep.create_payload({"prompt": "hello", "unknown_field": "ignored"})
        req = result["request"]
        # The request object should exist and not have unknown_field
        assert not hasattr(req, "unknown_field")


# ---------------------------------------------------------------------------
# stream() — mocked stream_codex_cli
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_text_chunk(self):
        from unittest.mock import patch

        from lionagi.providers.openai.codex.models import CodexChunk
        from lionagi.service.types.stream_chunk import StreamChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield CodexChunk(
                raw={"type": "text", "text": "hello"}, type="text", text="hello"
            )

        with patch(
            "lionagi.providers.openai.codex.endpoint.stream_codex_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "text"
        assert chunks[0].content == "hello"

    @pytest.mark.asyncio
    async def test_stream_yields_tool_use_chunk(self):
        from unittest.mock import patch

        from lionagi.providers.openai.codex.models import CodexChunk
        from lionagi.service.types.stream_chunk import StreamChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield CodexChunk(
                raw={"type": "tool_use", "name": "Read"},
                type="tool_use",
                tool_use={"name": "Read", "id": "tool-1", "input": {"path": "/tmp/x"}},
            )

        with patch(
            "lionagi.providers.openai.codex.endpoint.stream_codex_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "tool_use"
        assert chunks[0].tool_name == "Read"

    @pytest.mark.asyncio
    async def test_stream_skips_codex_session_items(self):
        from unittest.mock import patch

        from lionagi.providers.openai.codex.models import CodexChunk, CodexSession

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield CodexSession()  # should be skipped
            yield CodexChunk(raw={"type": "text", "text": "hi"}, type="text", text="hi")

        with patch(
            "lionagi.providers.openai.codex.endpoint.stream_codex_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_yields_tool_result_chunk(self):
        from unittest.mock import patch

        from lionagi.providers.openai.codex.models import CodexChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield CodexChunk(
                raw={"type": "tool_result"},
                type="tool_result",
                tool_result={
                    "tool_use_id": "tool-1",
                    "content": "file contents",
                    "is_error": False,
                },
            )

        with patch(
            "lionagi.providers.openai.codex.endpoint.stream_codex_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "tool_result"
        assert chunks[0].tool_output == "file contents"


# ---------------------------------------------------------------------------
# _call() — mocked stream_codex_cli
# ---------------------------------------------------------------------------


class TestCall:
    @pytest.mark.asyncio
    async def test_call_returns_session_dict(self):
        from unittest.mock import patch

        from lionagi.providers.openai.codex.models import CodexChunk, CodexSession

        ep = _make_endpoint()

        async def fake_stream(request_obj, session=None, **handlers):
            if session is not None:
                # Simulate a text chunk being appended to session
                chunk = CodexChunk(
                    raw={"type": "text", "text": "result text"},
                    type="text",
                    text="result text",
                )
                session.chunks.append(chunk)
            yield {"type": "done"}

        payload, _ = ep.create_payload({"prompt": "Do something"})

        with patch(
            "lionagi.providers.openai.codex.endpoint.stream_codex_cli",
            side_effect=fake_stream,
        ):
            result = await ep._call(payload, {})

        assert isinstance(result, dict)
        # session_id field should be present (even if None)
        assert "session_id" in result

    @pytest.mark.asyncio
    async def test_call_populates_result_from_text_chunks(self):
        from unittest.mock import patch

        from lionagi.providers.openai.codex.models import CodexChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, session=None, **handlers):
            if session is not None:
                for txt in ["Hello", " World"]:
                    session.chunks.append(
                        CodexChunk(
                            raw={"type": "text", "text": txt}, type="text", text=txt
                        )
                    )
            yield {"type": "done"}

        payload, _ = ep.create_payload({"prompt": "hello"})

        with patch(
            "lionagi.providers.openai.codex.endpoint.stream_codex_cli",
            side_effect=fake_stream,
        ):
            result = await ep._call(payload, {})

        # result should contain joined text
        assert "Hello" in result["result"] or " World" in result["result"]
