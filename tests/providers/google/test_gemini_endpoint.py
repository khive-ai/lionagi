# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.google.gemini_code.endpoint.

Covers GeminiCLIEndpoint: handlers validation, setter, update_handlers,
copy_runtime_state_to, _runtime_handlers, create_payload, stream(), _call().
All tests are deterministic — stream_gemini_cli is fully mocked.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_endpoint(**kwargs):
    from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

    return GeminiCLIEndpoint(**kwargs)


# ---------------------------------------------------------------------------
# _validate_handlers — standalone function tests
# ---------------------------------------------------------------------------


class TestGeminiValidateHandlers:
    def test_raises_if_not_dict(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_handlers(["on_text"])  # type: ignore[arg-type]

    def test_raises_on_invalid_key(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="Invalid handler key"):
            _validate_handlers({"on_unknown_event": lambda x: x})

    def test_raises_on_non_callable_value(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="callable or None"):
            _validate_handlers({"on_text": 42})

    def test_accepts_valid_callable(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        _validate_handlers({"on_text": lambda chunk: None})

    def test_accepts_none_values(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        _validate_handlers({"on_text": None, "on_final": None})


# ---------------------------------------------------------------------------
# GeminiCLIEndpoint — init and default handlers
# ---------------------------------------------------------------------------


class TestGeminiCLIEndpointInit:
    def test_default_handlers_all_none(self):
        ep = _make_endpoint()
        for v in ep.gemini_handlers.values():
            assert v is None

    def test_handlers_keys_are_standard(self):
        ep = _make_endpoint()
        assert set(ep.gemini_handlers.keys()) == {
            "on_text",
            "on_tool_use",
            "on_tool_result",
            "on_final",
        }

    def test_init_with_gemini_handlers_kwarg(self):
        handler = lambda chunk: None
        ep = _make_endpoint(gemini_handlers={"on_text": handler})
        assert ep.gemini_handlers["on_text"] is handler

    def test_init_with_invalid_handlers_raises(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        with pytest.raises(ValueError):
            GeminiCLIEndpoint(gemini_handlers={"bad_key": lambda x: x})


# ---------------------------------------------------------------------------
# gemini_handlers setter
# ---------------------------------------------------------------------------


class TestGeminiHandlersSetter:
    def test_setter_valid_dict(self):
        ep = _make_endpoint()
        handler = lambda c: None
        ep.gemini_handlers = {"on_text": handler}
        assert ep.gemini_handlers["on_text"] is handler
        assert ep.gemini_handlers["on_final"] is None

    def test_setter_invalid_key_raises(self):
        ep = _make_endpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            ep.gemini_handlers = {"on_bad_key": lambda c: None}

    def test_setter_resets_previous_handlers(self):
        ep = _make_endpoint(gemini_handlers={"on_text": lambda c: None})
        ep.gemini_handlers = {}
        assert ep.gemini_handlers["on_text"] is None


# ---------------------------------------------------------------------------
# update_handlers
# ---------------------------------------------------------------------------


class TestGeminiUpdateHandlers:
    def test_update_merges_new_handler(self):
        ep = _make_endpoint()
        handler = lambda c: None
        ep.update_handlers(on_final=handler)
        assert ep.gemini_handlers["on_final"] is handler
        assert ep.gemini_handlers["on_text"] is None

    def test_update_invalid_key_raises(self):
        ep = _make_endpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            ep.update_handlers(on_nonexistent=lambda c: None)


# ---------------------------------------------------------------------------
# copy_runtime_state_to
# ---------------------------------------------------------------------------


class TestGeminiCopyRuntimeStateTo:
    def test_copies_to_same_type(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        src = _make_endpoint(gemini_handlers={"on_text": lambda c: None})
        dst = _make_endpoint()
        src.copy_runtime_state_to(dst)
        assert dst.gemini_handlers["on_text"] is src.gemini_handlers["on_text"]

    def test_ignores_different_type(self):
        src = _make_endpoint(gemini_handlers={"on_text": lambda c: None})
        # Should silently skip non-GeminiCLIEndpoint targets
        src.copy_runtime_state_to(object())

    def test_copy_is_independent(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        src = _make_endpoint()
        dst = _make_endpoint()
        src.copy_runtime_state_to(dst)
        src.update_handlers(on_text=lambda c: None)
        assert dst.gemini_handlers["on_text"] is None


# ---------------------------------------------------------------------------
# _runtime_handlers
# ---------------------------------------------------------------------------


class TestGeminiRuntimeHandlers:
    def test_call_kwarg_overrides_instance_handler(self):
        ep = _make_endpoint()
        instance_handler = lambda c: "instance"
        call_handler = lambda c: "call"
        ep.update_handlers(on_text=instance_handler)
        kwargs = {"on_text": call_handler}
        runtime = ep._runtime_handlers(kwargs)
        assert runtime["on_text"] is call_handler
        assert "on_text" not in kwargs

    def test_none_values_filtered_out(self):
        ep = _make_endpoint()
        runtime = ep._runtime_handlers({})
        assert runtime == {}

    def test_non_handler_kwargs_not_consumed(self):
        ep = _make_endpoint()
        kwargs = {"model": "gemini-2.5-pro", "on_text": lambda c: None}
        runtime = ep._runtime_handlers(kwargs)
        assert "model" in kwargs
        assert "on_text" not in kwargs


# ---------------------------------------------------------------------------
# create_payload
# ---------------------------------------------------------------------------


class TestGeminiCreatePayload:
    def test_from_dict_with_prompt(self):
        ep = _make_endpoint()
        result, headers = ep.create_payload({"prompt": "Explain async/await"})
        assert "request" in result
        assert headers == {}
        assert result["request"].prompt == "Explain async/await"

    def test_from_gemini_code_request_object(self):
        from lionagi.providers.google.gemini_code.models import GeminiCodeRequest

        ep = _make_endpoint()
        req_obj = GeminiCodeRequest(prompt="Write a hello world program")
        result, headers = ep.create_payload(req_obj)
        assert result["request"].prompt == "Write a hello world program"

    def test_payload_extracts_messages_to_prompt(self):
        ep = _make_endpoint()
        result, _ = ep.create_payload(
            {"messages": [{"role": "user", "content": "Hello"}]}
        )
        # messages converted to prompt by GeminiCodeRequest model_validator
        assert result["request"].prompt == "Hello"


# ---------------------------------------------------------------------------
# stream() — mocked stream_gemini_cli
# ---------------------------------------------------------------------------


class TestGeminiStream:
    @pytest.mark.asyncio
    async def test_stream_yields_text_chunk(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import GeminiChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield GeminiChunk(
                raw={"type": "text", "text": "hello"},
                type="text",
                text="hello",
                is_delta=False,
            )

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "text"
        assert chunks[0].content == "hello"

    @pytest.mark.asyncio
    async def test_stream_passes_is_delta_flag(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import GeminiChunk
        from lionagi.service.types.stream_chunk import StreamChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield GeminiChunk(
                raw={"type": "text", "text": "frag"},
                type="text",
                text="frag",
                is_delta=True,
            )

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert chunks[0].is_delta is True

    @pytest.mark.asyncio
    async def test_stream_yields_tool_use_chunk(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import GeminiChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield GeminiChunk(
                raw={"type": "tool_use"},
                type="tool_use",
                tool_use={"name": "Bash", "id": "t1", "input": {"command": "ls"}},
            )

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert chunks[0].type == "tool_use"
        assert chunks[0].tool_name == "Bash"

    @pytest.mark.asyncio
    async def test_stream_skips_gemini_session_items(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import (
            GeminiChunk,
            GeminiSession,
        )

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield GeminiSession()
            yield GeminiChunk(
                raw={"type": "text", "text": "ok"},
                type="text",
                text="ok",
            )

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_yields_tool_result_chunk(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import GeminiChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, **handlers):
            yield GeminiChunk(
                raw={"type": "tool_result"},
                type="tool_result",
                tool_result={
                    "tool_use_id": "t1",
                    "content": "ls output",
                    "is_error": False,
                },
            )

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            chunks = []
            async for chunk in ep.stream({"prompt": "test"}):
                chunks.append(chunk)

        assert chunks[0].type == "tool_result"
        assert chunks[0].tool_output == "ls output"


# ---------------------------------------------------------------------------
# _call() — mocked stream_gemini_cli with delta accumulation
# ---------------------------------------------------------------------------


class TestGeminiCall:
    @pytest.mark.asyncio
    async def test_call_returns_session_dict(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import (
            GeminiChunk,
            GeminiSession,
        )

        ep = _make_endpoint()

        async def fake_stream(request_obj, session=None, **handlers):
            if session is not None:
                session.chunks.append(
                    GeminiChunk(
                        raw={"type": "text", "text": "result"},
                        type="text",
                        text="result",
                        is_delta=False,
                    )
                )
            yield {"type": "done"}

        payload, _ = ep.create_payload({"prompt": "Do something"})

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            result = await ep._call(payload, {})

        assert isinstance(result, dict)
        assert "session_id" in result

    @pytest.mark.asyncio
    async def test_call_accumulates_delta_fragments(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import GeminiChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, session=None, **handlers):
            if session is not None:
                # Three delta fragments
                for frag in ["Hello", " ", "World"]:
                    session.chunks.append(
                        GeminiChunk(
                            raw={"type": "text", "text": frag},
                            type="text",
                            text=frag,
                            is_delta=True,
                        )
                    )
            yield {"type": "done"}

        payload, _ = ep.create_payload({"prompt": "greet"})

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            result = await ep._call(payload, {})

        # Delta fragments should be concatenated
        assert "Hello World" in result["result"]

    @pytest.mark.asyncio
    async def test_call_mixes_delta_and_non_delta(self):
        from unittest.mock import patch

        from lionagi.providers.google.gemini_code.models import GeminiChunk

        ep = _make_endpoint()

        async def fake_stream(request_obj, session=None, **handlers):
            if session is not None:
                # Delta run
                for frag in ["line1-", "part2"]:
                    session.chunks.append(
                        GeminiChunk(
                            raw={"type": "text", "text": frag},
                            type="text",
                            text=frag,
                            is_delta=True,
                        )
                    )
                # Non-delta
                session.chunks.append(
                    GeminiChunk(
                        raw={"type": "text", "text": "standalone"},
                        type="text",
                        text="standalone",
                        is_delta=False,
                    )
                )
            yield {"type": "done"}

        payload, _ = ep.create_payload({"prompt": "mix"})

        with patch(
            "lionagi.providers.google.gemini_code.endpoint.stream_gemini_cli",
            side_effect=fake_stream,
        ):
            result = await ep._call(payload, {})

        # Both the delta-joined fragment and standalone should appear
        assert "line1-part2" in result["result"]
        assert "standalone" in result["result"]
