# Copyright (c) 2025-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage-targeted tests for provider endpoint runtime configuration.

Covers non-stream handler/config management methods for:
  - AG2GroupChatEndpoint
  - CodexCLIEndpoint
  - ClaudeCodeCLIEndpoint
  - GeminiCLIEndpoint
  - AG2NlipEndpoint
  - _event_to_chunk (AG2GroupChat)
  - _copy_runtime_value helpers
"""

from copy import deepcopy

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(*_, **__):
    return None


def _other_noop(*_, **__):
    return None


# ---------------------------------------------------------------------------
# AG2 GroupChat: _copy_runtime_value
# ---------------------------------------------------------------------------


class TestCopyRuntimeValue:
    def test_deepcopy_list(self):
        from lionagi.providers.ag2.groupchat.endpoint import _copy_runtime_value

        original = [{"name": "agent_a"}, {"name": "agent_b"}]
        result = _copy_runtime_value(original)
        assert result == original
        assert result is not original

    def test_deepcopy_dict(self):
        from lionagi.providers.ag2.groupchat.endpoint import _copy_runtime_value

        original = {"key": "value", "nested": {"x": 1}}
        result = _copy_runtime_value(original)
        assert result == original
        assert result is not original

    def test_deepcopy_falsy_bool(self):
        from lionagi.providers.ag2.groupchat.endpoint import _copy_runtime_value

        result = _copy_runtime_value(False)
        assert result is False

    def test_returns_original_when_deepcopy_fails(self):
        from lionagi.providers.ag2.groupchat.endpoint import _copy_runtime_value

        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("cannot deepcopy")

        obj = Uncopyable()
        result = _copy_runtime_value(obj)
        assert result is obj


# ---------------------------------------------------------------------------
# AG2 GroupChat: _validate_handlers
# ---------------------------------------------------------------------------


class TestAG2GroupChatValidateHandlers:
    def test_accepts_valid_callable(self):
        from lionagi.providers.ag2.groupchat.endpoint import _validate_handlers

        _validate_handlers({"on_text": _noop, "on_tool_use": None})

    def test_rejects_non_dict(self):
        from lionagi.providers.ag2.groupchat.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="dictionary"):
            _validate_handlers(["on_text"])  # type: ignore

    def test_rejects_invalid_key(self):
        from lionagi.providers.ag2.groupchat.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="Invalid handler key"):
            _validate_handlers({"on_bogus": _noop})

    def test_rejects_non_callable_value(self):
        from lionagi.providers.ag2.groupchat.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="callable or None"):
            _validate_handlers({"on_text": 42})  # type: ignore

    def test_accepts_all_none_values(self):
        from lionagi.providers.ag2.groupchat.endpoint import _validate_handlers

        _validate_handlers({"on_text": None, "on_tool_use": None})

    def test_accepts_empty_dict(self):
        from lionagi.providers.ag2.groupchat.endpoint import _validate_handlers

        _validate_handlers({})  # should not raise


# ---------------------------------------------------------------------------
# AG2 GroupChat: AG2GroupChatEndpoint init + handler accessors
# ---------------------------------------------------------------------------


class TestAG2GroupChatEndpointInit:
    def test_default_init_no_handlers(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint
        from lionagi.providers.ag2.groupchat.models import AG2_HANDLER_PARAMS

        endpoint = AG2GroupChatEndpoint()
        # All handler slots present, all None
        for key in AG2_HANDLER_PARAMS:
            assert key in endpoint.ag2_handlers
            assert endpoint.ag2_handlers[key] is None

    def test_init_with_ag2_handlers_kwarg(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint(ag2_handlers={"on_text": _noop})
        assert endpoint.ag2_handlers["on_text"] is _noop

    def test_init_handlers_do_not_leak_into_config_kwargs(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint(
            ag2_handlers={"on_text": _noop},
            agent_configs=[{"name": "A"}],
            llm_config=False,
        )
        assert "ag2_handlers" not in endpoint.config.kwargs
        assert "agent_configs" not in endpoint.config.kwargs
        assert "llm_config" not in endpoint.config.kwargs

    def test_init_with_invalid_handler_raises(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        with pytest.raises(ValueError):
            AG2GroupChatEndpoint(ag2_handlers={"on_bogus": _noop})

    def test_ag2_handlers_setter_replaces_all(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint
        from lionagi.providers.ag2.groupchat.models import AG2_HANDLER_PARAMS

        endpoint = AG2GroupChatEndpoint(ag2_handlers={"on_text": _noop})
        endpoint.ag2_handlers = {"on_tool_use": _other_noop}
        # old on_text should be None now
        assert endpoint.ag2_handlers["on_text"] is None
        assert endpoint.ag2_handlers["on_tool_use"] is _other_noop

    def test_ag2_handlers_setter_rejects_invalid_key(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            endpoint.ag2_handlers = {"bad_key": _noop}

    def test_update_handlers_merges(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint(ag2_handlers={"on_text": _noop})
        endpoint.update_handlers(on_tool_use=_other_noop)
        assert endpoint.ag2_handlers["on_text"] is _noop
        assert endpoint.ag2_handlers["on_tool_use"] is _other_noop

    def test_update_handlers_rejects_invalid_key(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            endpoint.update_handlers(on_fake=_noop)


# ---------------------------------------------------------------------------
# AG2 GroupChat: _runtime_handlers and _runtime_config
# ---------------------------------------------------------------------------


class TestAG2GroupChatRuntimeHandlers:
    def test_runtime_handlers_uses_stored_handlers(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint(ag2_handlers={"on_text": _noop})
        result = endpoint._runtime_handlers({})
        assert result == {"on_text": _noop}

    def test_runtime_handlers_call_kwargs_override_stored(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint(ag2_handlers={"on_text": _noop})
        kwargs = {"on_text": _other_noop}
        result = endpoint._runtime_handlers(kwargs)
        assert result["on_text"] is _other_noop
        # kwargs consumed
        assert "on_text" not in kwargs

    def test_runtime_handlers_omits_none_values(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint()
        result = endpoint._runtime_handlers({})
        assert result == {}

    def test_runtime_config_extracts_known_keys(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint()
        kwargs = {
            "agent_configs": [{"name": "A"}],
            "llm_config": {"model": "gpt-4"},
            "unrelated": "value",
        }
        result = endpoint._runtime_config(kwargs)
        assert result == {
            "agent_configs": [{"name": "A"}],
            "llm_config": {"model": "gpt-4"},
        }
        # known keys removed from kwargs, unrelated preserved
        assert "agent_configs" not in kwargs
        assert "llm_config" not in kwargs
        assert "unrelated" in kwargs

    def test_runtime_config_returns_empty_when_no_keys(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        endpoint = AG2GroupChatEndpoint()
        kwargs = {"on_text": _noop}
        result = endpoint._runtime_config(kwargs)
        assert result == {}
        assert "on_text" in kwargs  # untouched


# ---------------------------------------------------------------------------
# AG2 GroupChat: copy_runtime_state_to
# ---------------------------------------------------------------------------


class TestAG2GroupChatCopyRuntimeState:
    def test_copies_to_same_type(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        src = AG2GroupChatEndpoint(
            ag2_handlers={"on_text": _noop},
            agent_configs=[{"name": "A"}],
            llm_config={"model": "gpt"},
            tool_registry={"t": _noop},
        )
        dst = AG2GroupChatEndpoint()
        src.copy_runtime_state_to(dst)

        assert dst.ag2_handlers["on_text"] is _noop
        assert dst._agent_configs == [{"name": "A"}]
        assert dst._llm_config == {"model": "gpt"}
        assert dst._tool_registry == {"t": _noop}

    def test_copies_code_executor_by_reference(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        executor = object()
        src = AG2GroupChatEndpoint()
        src._code_executor = executor
        dst = AG2GroupChatEndpoint()
        src.copy_runtime_state_to(dst)
        assert dst._code_executor is executor

    def test_ignores_different_type(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint

        src = AG2GroupChatEndpoint(ag2_handlers={"on_text": _noop})
        # Pass something that is not an AG2GroupChatEndpoint — should not raise
        src.copy_runtime_state_to("not_an_endpoint")


# ---------------------------------------------------------------------------
# AG2 GroupChat: create_payload
# ---------------------------------------------------------------------------


class TestAG2GroupChatCreatePayload:
    def test_create_payload_from_dict(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint
        from lionagi.providers.ag2.groupchat.models import AG2GroupChatRequest

        endpoint = AG2GroupChatEndpoint()
        payload, headers = endpoint.create_payload(
            {"prompt": "hello", "agents": [{"name": "A"}]}
        )
        assert isinstance(payload["request"], AG2GroupChatRequest)
        assert payload["request"].prompt == "hello"
        assert headers == {}

    def test_create_payload_from_model(self):
        from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint
        from lionagi.providers.ag2.groupchat.models import AG2GroupChatRequest

        endpoint = AG2GroupChatEndpoint()
        req = AG2GroupChatRequest(prompt="world", agents=[{"name": "B"}])
        payload, headers = endpoint.create_payload(req)
        assert isinstance(payload["request"], AG2GroupChatRequest)
        assert payload["request"].prompt == "world"


# ---------------------------------------------------------------------------
# AG2 GroupChat: _event_to_chunk with fake event classes
# ---------------------------------------------------------------------------


class _FakeInner:
    """Generic inner content for wrapped events."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _FakeFunctionCall:
    def __init__(self, name=None, arguments=None):
        self.function = _FakeFunction(name=name, arguments=arguments)


class _FakeToolResponse:
    def __init__(self, content=None, tool_call_id=None):
        self.content = content
        self.tool_call_id = tool_call_id


# ---------------------------------------------------------------------------
# Fake autogen event classes — used to mock autogen.events.agent_events
# so _event_to_chunk tests run without the autogen package installed.
# _event_to_chunk does isinstance() checks, so the fake classes must be
# the exact types that are patched into sys.modules.
# ---------------------------------------------------------------------------


class _FakeTextEvent:
    pass


class _FakeGroupChatRunChatEvent:
    pass


class _FakeSelectSpeakerEvent:
    pass


class _FakeToolCallEvent:
    pass


class _FakeToolResponseEvent:
    pass


class _FakeRunCompletionEvent:
    pass


class _FakeTerminationEvent:
    pass


class _FakeErrorEvent:
    pass


def _make_fake_autogen_module():
    """Build a fake `autogen.events.agent_events` module."""
    import sys
    import types

    agent_events = types.ModuleType("autogen.events.agent_events")
    agent_events.TextEvent = _FakeTextEvent
    agent_events.GroupChatRunChatEvent = _FakeGroupChatRunChatEvent
    agent_events.SelectSpeakerEvent = _FakeSelectSpeakerEvent
    agent_events.ToolCallEvent = _FakeToolCallEvent
    agent_events.ToolResponseEvent = _FakeToolResponseEvent
    agent_events.RunCompletionEvent = _FakeRunCompletionEvent
    agent_events.TerminationEvent = _FakeTerminationEvent
    agent_events.ErrorEvent = _FakeErrorEvent

    autogen_events = types.ModuleType("autogen.events")
    autogen_events.agent_events = agent_events

    autogen = types.ModuleType("autogen")
    autogen.events = autogen_events

    return autogen, autogen_events, agent_events


@pytest.fixture
def fake_autogen(monkeypatch):
    """Inject fake autogen modules and reload endpoint so isinstance checks work."""
    import importlib
    import sys

    autogen_mod, events_mod, agent_events_mod = _make_fake_autogen_module()
    monkeypatch.setitem(sys.modules, "autogen", autogen_mod)
    monkeypatch.setitem(sys.modules, "autogen.events", events_mod)
    monkeypatch.setitem(sys.modules, "autogen.events.agent_events", agent_events_mod)

    # Re-import _event_to_chunk so it picks up the patched modules at call time.
    # The function uses lazy imports inside the body, so no reload needed —
    # sys.modules patch is sufficient.
    yield agent_events_mod


class TestEventToChunk:
    def _make_event(self, event_class, **inner_kwargs):
        """Create an instance of event_class with a .content attribute."""
        event = event_class()
        inner = _FakeInner(**inner_kwargs)
        event.content = inner
        return event

    def test_text_event_produces_text_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(_FakeTextEvent, content="hello", sender="AgentA")
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "text"
        assert chunk.content == "hello"
        assert chunk.metadata["agent"] == "AgentA"

    def test_text_event_no_inner_content_fallback(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(
            _FakeTextEvent, content="fallback_str", sender="unknown"
        )
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "text"

    def test_text_event_content_none_uses_str(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = _FakeTextEvent()
        event.content = None  # inner is None; fallback = str(event)
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "text"

    def test_groupchat_run_chat_event_produces_system_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(_FakeGroupChatRunChatEvent, speaker="AgentB")
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "system"
        assert "AgentB" in chunk.content
        assert chunk.metadata["event"] == "speaker_turn"

    def test_select_speaker_event_with_agents(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        class _FakeAgent:
            name = "CandidateAgent"

        event = self._make_event(_FakeSelectSpeakerEvent, agents=[_FakeAgent()])
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "system"
        assert "CandidateAgent" in chunk.content

    def test_select_speaker_event_no_agents(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(_FakeSelectSpeakerEvent, agents=[])
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert "?" in chunk.content

    def test_tool_call_event_produces_tool_use_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        fc = _FakeFunctionCall(name="my_tool", arguments={"arg": "val"})
        event = self._make_event(_FakeToolCallEvent, tool_calls=[fc], sender="AgentC")
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "tool_use"
        assert chunk.tool_name == "my_tool"
        assert chunk.tool_input == {"arg": "val"}
        assert chunk.metadata["agent"] == "AgentC"

    def test_tool_call_event_empty_tool_calls(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(_FakeToolCallEvent, tool_calls=[], sender="AgentC")
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "tool_use"
        assert chunk.tool_name is None

    def test_tool_response_event_produces_tool_result_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        tr = _FakeToolResponse(content="result_value", tool_call_id="tc-1")
        event = self._make_event(
            _FakeToolResponseEvent, tool_responses=[tr], sender="AgentD"
        )
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "tool_result"
        assert chunk.tool_output == "result_value"
        assert chunk.metadata["tool_call_id"] == "tc-1"

    def test_run_completion_event_produces_result_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(
            _FakeRunCompletionEvent,
            summary="All done",
            last_speaker="AgentE",
            cost=0.01,
        )
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "result"
        assert chunk.content == "All done"
        assert chunk.metadata["last_speaker"] == "AgentE"

    def test_run_completion_event_no_summary_fallback(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(
            _FakeRunCompletionEvent, summary=None, last_speaker=None, cost=None
        )
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "result"
        assert "complete" in chunk.content.lower()

    def test_termination_event_produces_system_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(_FakeTerminationEvent, reason="max rounds reached")
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "system"
        assert chunk.content == "max rounds reached"
        assert chunk.metadata["event"] == "termination"

    def test_error_event_produces_error_chunk(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        event = self._make_event(_FakeErrorEvent, error=RuntimeError("boom"))
        chunk = _event_to_chunk(event)
        assert chunk is not None
        assert chunk.type == "result"
        assert chunk.is_error is True
        assert "boom" in chunk.content

    def test_unknown_event_returns_none(self, fake_autogen):
        from lionagi.providers.ag2.groupchat.endpoint import _event_to_chunk

        class _UnknownEvent:
            content = None

        chunk = _event_to_chunk(_UnknownEvent())
        assert chunk is None


# ---------------------------------------------------------------------------
# CodexCLIEndpoint
# ---------------------------------------------------------------------------


class TestCodexValidateHandlers:
    def test_accepts_valid_keys(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        _validate_handlers({"on_text": _noop, "on_final": None})

    def test_rejects_invalid_key(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="Invalid handler key"):
            _validate_handlers({"on_bad": _noop})

    def test_rejects_non_dict(self):
        from lionagi.providers.openai.codex.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="dictionary"):
            _validate_handlers(None)  # type: ignore


class TestCodexCLIEndpointInit:
    def test_default_init(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint()
        assert endpoint._codex_handlers["on_text"] is None
        assert endpoint._codex_handlers["on_final"] is None

    def test_init_with_handlers(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint(codex_handlers={"on_text": _noop})
        assert endpoint._codex_handlers["on_text"] is _noop

    def test_handlers_do_not_leak_to_config(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint(codex_handlers={"on_text": _noop})
        assert "codex_handlers" not in endpoint.config.kwargs

    def test_update_handlers_merges(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint(codex_handlers={"on_text": _noop})
        endpoint.update_handlers(on_tool_use=_other_noop)
        assert endpoint._codex_handlers["on_text"] is _noop
        assert endpoint._codex_handlers["on_tool_use"] is _other_noop

    def test_update_handlers_rejects_invalid(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            endpoint.update_handlers(on_fake=_noop)

    def test_runtime_handlers_merges_call_kwargs(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint(codex_handlers={"on_text": _noop})
        kwargs = {"on_text": _other_noop, "extra": "value"}
        result = endpoint._runtime_handlers(kwargs)
        assert result["on_text"] is _other_noop
        assert "on_text" not in kwargs
        assert kwargs.get("extra") == "value"

    def test_runtime_handlers_omits_none(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        endpoint = CodexCLIEndpoint()
        result = endpoint._runtime_handlers({})
        assert result == {}

    def test_copy_runtime_state_to_same_type(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        src = CodexCLIEndpoint(codex_handlers={"on_text": _noop})
        dst = CodexCLIEndpoint()
        src.copy_runtime_state_to(dst)
        assert dst._codex_handlers["on_text"] is _noop

    def test_copy_runtime_state_to_different_type_ignored(self):
        from lionagi.providers.openai.codex.endpoint import CodexCLIEndpoint

        src = CodexCLIEndpoint(codex_handlers={"on_text": _noop})
        # Should not raise
        src.copy_runtime_state_to("not_a_codex_endpoint")


# ---------------------------------------------------------------------------
# ClaudeCodeCLIEndpoint
# ---------------------------------------------------------------------------


class TestClaudeCodeCLIEndpointInit:
    def test_default_init(self):
        from lionagi.providers.anthropic.claude_code.endpoint import (
            ClaudeCodeCLIEndpoint,
        )

        endpoint = ClaudeCodeCLIEndpoint()
        assert endpoint._claude_handlers["on_text"] is None
        assert endpoint._claude_handlers["on_thinking"] is None
        assert endpoint._claude_handlers["on_system"] is None

    def test_init_with_claude_handlers(self):
        from lionagi.providers.anthropic.claude_code.endpoint import (
            ClaudeCodeCLIEndpoint,
        )

        endpoint = ClaudeCodeCLIEndpoint(claude_handlers={"on_text": _noop})
        assert endpoint._claude_handlers["on_text"] is _noop

    def test_handlers_do_not_leak_to_config(self):
        from lionagi.providers.anthropic.claude_code.endpoint import (
            ClaudeCodeCLIEndpoint,
        )

        endpoint = ClaudeCodeCLIEndpoint(claude_handlers={"on_text": _noop})
        assert "claude_handlers" not in endpoint.config.kwargs

    def test_runtime_handlers_merges_call_kwargs(self):
        from lionagi.providers.anthropic.claude_code.endpoint import (
            ClaudeCodeCLIEndpoint,
        )

        endpoint = ClaudeCodeCLIEndpoint(claude_handlers={"on_thinking": _noop})
        kwargs = {"on_thinking": _other_noop, "other": "value"}
        result = endpoint._runtime_handlers(kwargs)
        assert result["on_thinking"] is _other_noop
        assert "on_thinking" not in kwargs
        assert kwargs.get("other") == "value"

    def test_runtime_handlers_omits_none(self):
        from lionagi.providers.anthropic.claude_code.endpoint import (
            ClaudeCodeCLIEndpoint,
        )

        endpoint = ClaudeCodeCLIEndpoint()
        result = endpoint._runtime_handlers({})
        assert result == {}


# ---------------------------------------------------------------------------
# GeminiCLIEndpoint
# ---------------------------------------------------------------------------


class TestGeminiValidateHandlers:
    def test_accepts_valid_keys(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        _validate_handlers({"on_text": _noop, "on_final": None})

    def test_rejects_invalid_key(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="Invalid handler key"):
            _validate_handlers({"on_stream": _noop})

    def test_rejects_non_dict(self):
        from lionagi.providers.google.gemini_code.endpoint import _validate_handlers

        with pytest.raises(ValueError, match="dictionary"):
            _validate_handlers([])  # type: ignore


class TestGeminiCLIEndpointInit:
    def test_default_init(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint()
        assert endpoint._gemini_handlers["on_text"] is None

    def test_init_with_handlers(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        assert endpoint._gemini_handlers["on_text"] is _noop

    def test_handlers_do_not_leak_to_config(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        assert "gemini_handlers" not in endpoint.config.kwargs

    def test_gemini_handlers_property_returns_dict(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        assert isinstance(endpoint.gemini_handlers, dict)
        assert endpoint.gemini_handlers["on_text"] is _noop

    def test_gemini_handlers_setter_replaces_all(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        endpoint.gemini_handlers = {"on_tool_use": _other_noop}
        assert endpoint.gemini_handlers["on_text"] is None
        assert endpoint.gemini_handlers["on_tool_use"] is _other_noop

    def test_update_handlers_merges(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        endpoint.update_handlers(on_final=_other_noop)
        assert endpoint.gemini_handlers["on_text"] is _noop
        assert endpoint.gemini_handlers["on_final"] is _other_noop

    def test_update_handlers_rejects_invalid(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint()
        with pytest.raises(ValueError, match="Invalid handler key"):
            endpoint.update_handlers(on_bad=_noop)

    def test_copy_runtime_state_to_same_type(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        src = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        dst = GeminiCLIEndpoint()
        src.copy_runtime_state_to(dst)
        assert dst.gemini_handlers["on_text"] is _noop

    def test_copy_runtime_state_to_different_type_ignored(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        src = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        src.copy_runtime_state_to("not_gemini")  # Should not raise

    def test_runtime_handlers_merges_call_kwargs(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint(gemini_handlers={"on_text": _noop})
        kwargs = {"on_text": _other_noop, "unrelated": 42}
        result = endpoint._runtime_handlers(kwargs)
        assert result["on_text"] is _other_noop
        assert "on_text" not in kwargs
        assert kwargs["unrelated"] == 42

    def test_runtime_handlers_omits_none(self):
        from lionagi.providers.google.gemini_code.endpoint import GeminiCLIEndpoint

        endpoint = GeminiCLIEndpoint()
        result = endpoint._runtime_handlers({})
        assert result == {}


# ---------------------------------------------------------------------------
# AG2NlipEndpoint: init, copy_runtime_state_to, as_agent_config
# ---------------------------------------------------------------------------


class TestAG2NlipEndpointInit:
    def test_default_init(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint()
        assert endpoint._url == ""
        assert endpoint._timeout == 60.0
        assert endpoint._max_retries == 3
        assert endpoint._agent_name == "remote"
        assert endpoint._silent is None

    def test_init_with_kwargs(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint(
            url="https://remote.test",
            agent_name="ExpertAgent",
            timeout=10.0,
            max_retries=2,
            silent=True,
        )
        assert endpoint._url == "https://remote.test"
        assert endpoint._agent_name == "ExpertAgent"
        assert endpoint._timeout == 10.0
        assert endpoint._max_retries == 2
        assert endpoint._silent is True

    def test_nlip_kwargs_do_not_leak_to_config(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint(url="https://remote.test", agent_name="X")
        assert "url" not in endpoint.config.kwargs
        assert "agent_name" not in endpoint.config.kwargs


class TestAG2NlipCopyRuntimeState:
    def test_copies_all_fields(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        src = AG2NlipEndpoint(
            url="https://remote.test",
            agent_name="SrcAgent",
            timeout=15.0,
            max_retries=5,
            silent=False,
        )
        dst = AG2NlipEndpoint()
        src.copy_runtime_state_to(dst)

        assert dst._url == "https://remote.test"
        assert dst._agent_name == "SrcAgent"
        assert dst._timeout == 15.0
        assert dst._max_retries == 5
        assert dst._silent is False

    def test_ignores_different_type(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        src = AG2NlipEndpoint(url="https://remote.test")
        src.copy_runtime_state_to("not_nlip")  # Should not raise


class TestAG2NlipAsAgentConfig:
    def test_basic_config(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint(
            url="https://remote.test",
            agent_name="RemoteExpert",
            timeout=12.0,
            max_retries=2,
            silent=True,
        )
        config = endpoint.as_agent_config(
            role="remote researcher",
            description="Remote specialist",
        )
        assert config["nlip_url"] == "https://remote.test"
        assert config["name"] == "RemoteExpert"
        assert config["role"] == "remote researcher"
        assert config["nlip_timeout"] == 12.0
        assert config["nlip_max_retries"] == 2
        assert config["nlip_silent"] is True

    def test_overrides_url_and_name(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint(
            url="https://default.test",
            agent_name="DefaultAgent",
        )
        config = endpoint.as_agent_config(
            url="https://override.test",
            name="OverriddenAgent",
        )
        assert config["nlip_url"] == "https://override.test"
        assert config["name"] == "OverriddenAgent"

    def test_raises_when_no_url(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint()  # _url == ""
        with pytest.raises(ValueError, match="url"):
            endpoint.as_agent_config()

    def test_includes_context_variables_and_client_tools(self):
        from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint

        endpoint = AG2NlipEndpoint(url="https://remote.test")
        config = endpoint.as_agent_config(
            context_variables={"topic": "ai"},
            client_tools=[{"name": "search"}],
        )
        assert config["context_variables"] == {"topic": "ai"}
        assert config["nlip_client_tools"] == [{"name": "search"}]
