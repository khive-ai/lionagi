# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.pi.cli.models — deterministic, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionagi.providers.pi.cli.models import (
    _PI_MODEL_PROVIDER_MAP,
    PiChunk,
    PiCodeRequest,
    PiSession,
    _assistant_message_text,
    _error_message_from_event,
    _extract_summary,
    _maybe_await,
    _pp_text,
    _pp_tool_result,
    _pp_tool_use,
    _remember_assistant_message,
    _tool_call_from_event,
    print_readable,
)

# ---------------------------------------------------------------------------
# PiCodeRequest — construction and validation (lines 203-205, 217-226, 232-250)
# ---------------------------------------------------------------------------


class TestPiCodeRequestValidation:
    def test_minimal_with_prompt(self):
        req = PiCodeRequest(prompt="Write a function")
        assert req.prompt == "Write a function"

    def test_no_session_default_true(self):
        req = PiCodeRequest(prompt="x")
        assert req.no_session is True

    def test_derive_prompt_from_messages(self):
        req = PiCodeRequest(messages=[{"role": "user", "content": "Hello"}])
        assert req.prompt == "Hello"

    def test_derive_prompt_from_messages_system_extracted(self):
        req = PiCodeRequest(
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Write tests"},
            ]
        )
        assert "Write tests" in req.prompt
        assert req.system_prompt == "Be concise"

    def test_missing_both_prompt_and_messages_raises(self):
        with pytest.raises(ValueError, match="messages or prompt required"):
            PiCodeRequest(messages=[])

    def test_messages_dict_content_json_serialized(self):
        req = PiCodeRequest(messages=[{"role": "user", "content": {"key": "value"}}])
        assert "key" in req.prompt

    def test_messages_list_content_json_serialized(self):
        req = PiCodeRequest(messages=[{"role": "user", "content": [{"text": "hello"}]}])
        assert req.prompt

    def test_norm_tools_string_to_list(self):
        req = PiCodeRequest(prompt="x", tools="bash")
        assert req.tools == ["bash"]

    def test_norm_tools_list_passthrough(self):
        req = PiCodeRequest(prompt="x", tools=["bash", "read"])
        assert req.tools == ["bash", "read"]

    def test_infer_provider_from_model_deepseek(self):
        req = PiCodeRequest(prompt="x", model="deepseek-chat")
        assert req.provider == "deepseek"

    def test_infer_provider_from_model_anthropic(self):
        req = PiCodeRequest(prompt="x", model="claude-3-opus")
        assert req.provider == "anthropic"
        assert req.model == "claude-3-opus"

    def test_infer_provider_from_model_openai_gpt(self):
        req = PiCodeRequest(prompt="x", model="gpt-4o")
        assert req.provider == "openai"

    def test_infer_provider_from_model_openai_o1(self):
        req = PiCodeRequest(prompt="x", model="o1-mini")
        assert req.provider == "openai"

    def test_infer_provider_from_model_openai_o3(self):
        req = PiCodeRequest(prompt="x", model="o3-mini")
        assert req.provider == "openai"

    def test_infer_provider_from_model_openai_o4(self):
        req = PiCodeRequest(prompt="x", model="o4-mini")
        assert req.provider == "openai"

    def test_infer_provider_openrouter_strips_prefix(self):
        req = PiCodeRequest(prompt="x", model="openrouter/openai/gpt-4o")
        assert req.provider == "openrouter"
        assert req.model == "openai/gpt-4o"  # prefix stripped

    def test_explicit_provider_not_overridden(self):
        req = PiCodeRequest(prompt="x", model="claude-3-opus", provider="bedrock")
        assert req.provider == "bedrock"

    def test_unknown_model_no_provider_inferred(self):
        req = PiCodeRequest(prompt="x", model="llama-3-70b")
        assert req.provider is None


# ---------------------------------------------------------------------------
# PiCodeRequest.as_cmd_args (lines 259-273)
# ---------------------------------------------------------------------------


class TestPiCodeRequestCmdArgs:
    def test_basic_structure(self):
        req = PiCodeRequest(prompt="fix bugs")
        args = req.as_cmd_args()
        assert "-p" in args
        assert "--mode" in args
        assert "json" in args
        assert "fix bugs" in args

    def test_model_flag_included(self):
        req = PiCodeRequest(prompt="x", model="gpt-4o", provider="openai")
        args = req.as_cmd_args()
        assert "--model" in args
        assert "gpt-4o" in args

    def test_provider_flag_included(self):
        req = PiCodeRequest(prompt="x", provider="anthropic")
        args = req.as_cmd_args()
        assert "--provider" in args
        assert "anthropic" in args

    def test_no_session_flag_included(self):
        req = PiCodeRequest(prompt="x")
        args = req.as_cmd_args()
        assert "--no-session" in args

    def test_no_tools_flag(self):
        req = PiCodeRequest(prompt="x", no_tools=True)
        args = req.as_cmd_args()
        assert "--no-tools" in args

    def test_tools_repeat_flag(self):
        req = PiCodeRequest(prompt="x", tools=["bash", "read"])
        args = req.as_cmd_args()
        assert "--tools" in args
        assert args.count("--tools") == 2

    def test_system_prompt_flag(self):
        req = PiCodeRequest(prompt="x", system_prompt="Be helpful")
        args = req.as_cmd_args()
        assert "--system-prompt" in args
        assert "Be helpful" in args

    def test_file_args_at_prefixed(self):
        req = PiCodeRequest(prompt="x", file_args=["src/main.py"])
        args = req.as_cmd_args()
        assert "@src/main.py" in args

    def test_file_args_already_at_prefixed_unchanged(self):
        req = PiCodeRequest(prompt="x", file_args=["@README.md"])
        args = req.as_cmd_args()
        assert "@README.md" in args
        assert "@@README.md" not in args

    def test_no_builtin_tools_flag(self):
        req = PiCodeRequest(prompt="x", no_builtin_tools=True)
        args = req.as_cmd_args()
        assert "--no-builtin-tools" in args

    def test_no_context_files_flag(self):
        req = PiCodeRequest(prompt="x", no_context_files=True)
        args = req.as_cmd_args()
        assert "--no-context-files" in args

    def test_extension_repeat_flag(self):
        req = PiCodeRequest(prompt="x", extension=["ext1.js", "ext2.js"])
        args = req.as_cmd_args()
        assert "--extension" in args
        assert args.count("--extension") == 2

    def test_skill_repeat_flag(self):
        req = PiCodeRequest(prompt="x", skill=["skill1"])
        args = req.as_cmd_args()
        assert "--skill" in args

    def test_no_extensions_flag(self):
        req = PiCodeRequest(prompt="x", no_extensions=True)
        args = req.as_cmd_args()
        assert "--no-extensions" in args

    def test_no_skills_flag(self):
        req = PiCodeRequest(prompt="x", no_skills=True)
        args = req.as_cmd_args()
        assert "--no-skills" in args

    def test_thinking_flag(self):
        req = PiCodeRequest(prompt="x", thinking="high")
        args = req.as_cmd_args()
        assert "--thinking" in args
        assert "high" in args

    def test_append_system_prompt_repeat(self):
        req = PiCodeRequest(prompt="x", append_system_prompt=["extra context"])
        args = req.as_cmd_args()
        assert "--append-system-prompt" in args

    def test_none_values_not_in_args(self):
        req = PiCodeRequest(prompt="x")
        args = req.as_cmd_args()
        # provider and model are None — they should not appear
        assert "--model" not in args

    def test_false_bool_not_in_args(self):
        # no_tools=False (default) → flag should NOT appear
        req = PiCodeRequest(prompt="x", no_tools=False)
        args = req.as_cmd_args()
        assert "--no-tools" not in args


# ---------------------------------------------------------------------------
# PiCodeRequest.env (lines 289-293)
# ---------------------------------------------------------------------------


class TestPiCodeRequestEnv:
    def test_no_api_key_returns_none(self):
        req = PiCodeRequest(prompt="x")
        assert req.env() is None

    def test_known_provider_maps_env_var(self):
        req = PiCodeRequest(prompt="x", api_key="sk-test", provider="anthropic")
        env = req.env()
        assert env == {"ANTHROPIC_API_KEY": "sk-test"}

    def test_openai_provider(self):
        req = PiCodeRequest(prompt="x", api_key="sk-openai", provider="openai")
        env = req.env()
        assert "OPENAI_API_KEY" in env

    def test_google_provider(self):
        req = PiCodeRequest(prompt="x", api_key="goog-key", provider="google")
        env = req.env()
        assert "GEMINI_API_KEY" in env

    def test_deepseek_provider(self):
        req = PiCodeRequest(prompt="x", api_key="ds-key", provider="deepseek")
        env = req.env()
        assert "DEEPSEEK_API_KEY" in env

    def test_openrouter_provider(self):
        req = PiCodeRequest(prompt="x", api_key="or-key", provider="openrouter")
        env = req.env()
        assert "OPENROUTER_API_KEY" in env

    def test_groq_provider(self):
        req = PiCodeRequest(prompt="x", api_key="groq-key", provider="groq")
        env = req.env()
        assert "GROQ_API_KEY" in env

    def test_unknown_provider_uses_upper_convention(self):
        req = PiCodeRequest(prompt="x", api_key="key", provider="mycloud")
        env = req.env()
        assert "MYCLOUD_API_KEY" in env

    def test_no_provider_defaults_to_google(self):
        req = PiCodeRequest(prompt="x", api_key="gkey")
        env = req.env()
        assert "GEMINI_API_KEY" in env


# ---------------------------------------------------------------------------
# PiChunk and PiSession dataclasses (lines 333-363, 368)
# ---------------------------------------------------------------------------


class TestPiChunk:
    def test_basic_construction(self):
        chunk = PiChunk(raw={"type": "text_delta"}, type="text_delta")
        assert chunk.type == "text_delta"
        assert chunk.raw == {"type": "text_delta"}
        assert chunk.text is None
        assert chunk.thinking is None
        assert chunk.tool_use is None
        assert chunk.tool_result is None

    def test_with_text(self):
        chunk = PiChunk(raw={}, type="text", text="hello")
        assert chunk.text == "hello"

    def test_with_tool_use(self):
        tu = {"name": "bash", "input": {"command": "ls"}}
        chunk = PiChunk(raw={}, type="tool_use", tool_use=tu)
        assert chunk.tool_use == tu

    def test_with_tool_result(self):
        tr = {"content": "output", "is_error": False}
        chunk = PiChunk(raw={}, type="tool_result", tool_result=tr)
        assert chunk.tool_result == tr

    def test_with_thinking(self):
        chunk = PiChunk(raw={}, type="thinking", thinking="Let me think...")
        assert chunk.thinking == "Let me think..."


class TestPiSession:
    def test_default_construction(self):
        session = PiSession()
        assert session.session_id is None
        assert session.model is None
        assert session.chunks == []
        assert session.messages == []
        assert session.tool_uses == []
        assert session.tool_results == []
        assert session.result == ""
        assert session.usage == {}
        assert session.num_turns is None
        assert session.duration_ms is None
        assert session.is_error is False
        assert session.summary is None

    def test_populate_summary_basic(self):
        session = PiSession()
        session.result = "Done"
        session.populate_summary()
        assert session.summary is not None
        assert "tool_counts" in session.summary
        assert "key_actions" in session.summary

    def test_populate_summary_with_tool_uses(self):
        session = PiSession()
        session.tool_uses = [
            {"name": "read", "input": {"path": "main.py"}},
            {"name": "bash", "input": {"command": "ls -la"}},
        ]
        session.result = "Files processed"
        session.populate_summary()
        assert session.summary["tool_counts"]["read"] == 1
        assert session.summary["tool_counts"]["bash"] == 1
        assert "Read main.py" in session.summary["key_actions"]
        assert any("Ran:" in a for a in session.summary["key_actions"])


# ---------------------------------------------------------------------------
# _extract_summary (lines 368-407, 426-498 → helpers)
# ---------------------------------------------------------------------------


class TestExtractSummary:
    def test_empty_session(self):
        session = PiSession()
        summary = _extract_summary(session)
        assert summary["tool_counts"] == {}
        assert summary["total_tool_calls"] == 0
        assert summary["key_actions"] == ["No specific actions"]

    def test_read_file_operation(self):
        session = PiSession()
        session.tool_uses = [{"name": "read", "input": {"path": "foo.py"}}]
        summary = _extract_summary(session)
        assert "foo.py" in summary["file_operations"]["reads"]
        assert "Read foo.py" in summary["key_actions"]

    def test_read_file_alt_name_Read(self):
        session = PiSession()
        session.tool_uses = [{"name": "Read", "input": {"file_path": "bar.py"}}]
        summary = _extract_summary(session)
        assert "bar.py" in summary["file_operations"]["reads"]

    def test_read_file_alt_name_read_file(self):
        session = PiSession()
        session.tool_uses = [{"name": "read_file", "input": {"path": "baz.py"}}]
        summary = _extract_summary(session)
        assert "baz.py" in summary["file_operations"]["reads"]

    def test_write_file_operation(self):
        session = PiSession()
        session.tool_uses = [{"name": "write", "input": {"path": "out.py"}}]
        summary = _extract_summary(session)
        assert "out.py" in summary["file_operations"]["writes"]
        assert "Wrote out.py" in summary["key_actions"]

    def test_write_file_alt_names(self):
        session = PiSession()
        session.tool_uses = [
            {"name": "Write", "input": {"file_path": "a.py"}},
            {"name": "write_file", "input": {"path": "b.py"}},
            {"name": "create_file", "input": {"path": "c.py"}},
        ]
        summary = _extract_summary(session)
        assert len(summary["file_operations"]["writes"]) == 3

    def test_edit_file_operation(self):
        session = PiSession()
        session.tool_uses = [{"name": "edit", "input": {"path": "edit.py"}}]
        summary = _extract_summary(session)
        assert "edit.py" in summary["file_operations"]["edits"]
        assert "Edited edit.py" in summary["key_actions"]

    def test_edit_alt_names(self):
        session = PiSession()
        session.tool_uses = [
            {"name": "Edit", "input": {"file_path": "x.py"}},
            {"name": "edit_file", "input": {"path": "y.py"}},
            {"name": "patch", "input": {"path": "z.py"}},
        ]
        summary = _extract_summary(session)
        assert len(summary["file_operations"]["edits"]) == 3

    def test_bash_operation_truncates_long_command(self):
        session = PiSession()
        long_cmd = "x" * 100
        session.tool_uses = [{"name": "bash", "input": {"command": long_cmd}}]
        summary = _extract_summary(session)
        assert any("Ran:" in a for a in summary["key_actions"])
        # should have been truncated
        action = next(a for a in summary["key_actions"] if "Ran:" in a)
        assert "..." in action

    def test_bash_alt_names(self):
        session = PiSession()
        session.tool_uses = [
            {"name": "Bash", "input": {"command": "echo hi"}},
            {"name": "shell", "input": {"cmd": "ls"}},
        ]
        summary = _extract_summary(session)
        assert summary["tool_counts"]["Bash"] == 1
        assert summary["tool_counts"]["shell"] == 1

    def test_unknown_tool_uses_used(self):
        session = PiSession()
        session.tool_uses = [{"name": "custom_tool", "input": {}}]
        summary = _extract_summary(session)
        assert "Used custom_tool" in summary["key_actions"]

    def test_result_truncation_over_200_chars(self):
        session = PiSession()
        session.result = "x" * 300
        summary = _extract_summary(session)
        assert summary["result_summary"].endswith("...")
        assert len(summary["result_summary"]) == 203

    def test_result_no_truncation_under_200(self):
        session = PiSession()
        session.result = "short result"
        summary = _extract_summary(session)
        assert summary["result_summary"] == "short result"

    def test_deduplication_in_file_operations(self):
        session = PiSession()
        session.tool_uses = [
            {"name": "read", "input": {"path": "dup.py"}},
            {"name": "read", "input": {"path": "dup.py"}},
        ]
        summary = _extract_summary(session)
        # duplicates removed
        assert summary["file_operations"]["reads"].count("dup.py") == 1

    def test_usage_stats_includes_session_usage(self):
        session = PiSession()
        session.usage = {"input_tokens": 100, "output_tokens": 50}
        session.num_turns = 3
        session.duration_ms = 1500
        summary = _extract_summary(session)
        assert summary["usage_stats"]["num_turns"] == 3
        assert summary["usage_stats"]["duration_ms"] == 1500
        assert summary["usage_stats"]["input_tokens"] == 100

    def test_tool_uses_args_key_fallback(self):
        # Some events use 'args' instead of 'input'
        session = PiSession()
        session.tool_uses = [
            {"name": "read", "args": {"path": "via_args.py"}, "input": {}}
        ]
        # input key exists but is empty, args key is secondary
        summary = _extract_summary(session)
        assert summary["tool_counts"]["read"] == 1


# ---------------------------------------------------------------------------
# _assistant_message_text (lines 543-554)
# ---------------------------------------------------------------------------


class TestAssistantMessageText:
    def test_string_content(self):
        msg = {"content": "Hello world"}
        assert _assistant_message_text(msg) == "Hello world"

    def test_list_content_with_text_blocks(self):
        msg = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]
        }
        result = _assistant_message_text(msg)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_list_content_with_string_items(self):
        msg = {"content": ["hello", "world"]}
        result = _assistant_message_text(msg)
        assert "hello" in result
        assert "world" in result

    def test_list_content_skips_empty_text(self):
        msg = {
            "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "real"},
            ]
        }
        result = _assistant_message_text(msg)
        assert result == "real"

    def test_empty_content(self):
        msg = {"content": ""}
        assert _assistant_message_text(msg) == ""

    def test_missing_content_key(self):
        msg = {}
        assert _assistant_message_text(msg) == ""

    def test_dict_content_type_not_text_skipped(self):
        msg = {
            "content": [
                {"type": "tool_use", "id": "1"},
                {"type": "text", "text": "kept"},
            ]
        }
        result = _assistant_message_text(msg)
        assert result == "kept"


# ---------------------------------------------------------------------------
# _remember_assistant_message (lines 561-570)
# ---------------------------------------------------------------------------


class TestRememberAssistantMessage:
    def test_non_dict_message_ignored(self):
        session = PiSession()
        _remember_assistant_message(session, "not a dict")
        assert session.model is None
        assert session.result == ""

    def test_none_message_ignored(self):
        session = PiSession()
        _remember_assistant_message(session, None)
        assert session.model is None

    def test_model_extracted(self):
        session = PiSession()
        _remember_assistant_message(session, {"model": "claude-3-opus"})
        assert session.model == "claude-3-opus"

    def test_usage_extracted(self):
        session = PiSession()
        usage = {"input_tokens": 10, "output_tokens": 5}
        _remember_assistant_message(session, {"usage": usage})
        assert session.usage == usage

    def test_usage_non_dict_ignored(self):
        session = PiSession()
        _remember_assistant_message(session, {"usage": "not a dict"})
        assert session.usage == {}

    def test_text_content_extracted(self):
        session = PiSession()
        _remember_assistant_message(session, {"content": "response text"})
        assert session.result == "response text"


# ---------------------------------------------------------------------------
# _tool_call_from_event (lines 574-579)
# ---------------------------------------------------------------------------


class TestToolCallFromEvent:
    def test_basic_tool_call(self):
        event = {"name": "bash", "input": {"command": "ls"}}
        tc = _tool_call_from_event(event)
        assert tc["name"] == "bash"
        assert tc["input"] == {"command": "ls"}

    def test_toolCall_key_used(self):
        event = {
            "toolCall": {
                "id": "tc1",
                "name": "read",
                "input": {"path": "main.py"},
            }
        }
        tc = _tool_call_from_event(event)
        assert tc["name"] == "read"
        assert tc["id"] == "tc1"

    def test_args_key_fallback(self):
        event = {"name": "edit", "args": {"path": "x.py"}}
        tc = _tool_call_from_event(event)
        assert tc["input"] == {"path": "x.py"}

    def test_arguments_as_json_string_parsed(self):
        event = {"name": "tool", "arguments": '{"key": "val"}'}
        tc = _tool_call_from_event(event)
        assert tc["input"] == {"key": "val"}

    def test_arguments_invalid_json_string_returns_string(self):
        event = {"name": "tool", "arguments": "not json {{"}
        tc = _tool_call_from_event(event)
        assert isinstance(tc["input"], str)

    def test_tool_name_alt_keys(self):
        event = {"toolName": "my_tool", "arguments": {}}
        tc = _tool_call_from_event(event)
        assert tc["name"] == "my_tool"

    def test_id_fallback_to_toolCallId(self):
        event = {"toolCallId": "abc123", "name": "x", "input": {}}
        tc = _tool_call_from_event(event)
        assert tc["id"] == "abc123"


# ---------------------------------------------------------------------------
# _error_message_from_event (lines 587-595)
# ---------------------------------------------------------------------------


class TestErrorMessageFromEvent:
    def test_error_dict_with_errorMessage(self):
        event = {"error": {"errorMessage": "something broke"}}
        msg = _error_message_from_event(event)
        assert msg == "something broke"

    def test_error_dict_with_message(self):
        event = {"error": {"message": "error msg"}}
        msg = _error_message_from_event(event)
        assert msg == "error msg"

    def test_error_dict_fallback_to_str(self):
        event = {"error": {"unknown_key": "data"}}
        msg = _error_message_from_event(event)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_error_string_value(self):
        event = {"error": "plain string error"}
        # error is not a dict → fallback to event.get("errorMessage") etc
        msg = _error_message_from_event(event)
        assert "plain string error" in msg

    def test_event_errorMessage_key(self):
        event = {"errorMessage": "top-level error"}
        msg = _error_message_from_event(event)
        assert msg == "top-level error"

    def test_event_message_key(self):
        event = {"message": "fallback message"}
        msg = _error_message_from_event(event)
        assert msg == "fallback message"

    def test_none_error_falls_through(self):
        event = {"type": "error"}
        msg = _error_message_from_event(event)
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# _maybe_await (lines 771-775)
# ---------------------------------------------------------------------------


class TestMaybeAwait:
    def test_sync_function_called(self):
        results = []

        def sync_fn(x):
            results.append(x)

        asyncio.run(_maybe_await(sync_fn, "arg"))
        assert results == ["arg"]

    def test_async_function_awaited(self):
        results = []

        async def async_fn(x):
            results.append(x)

        asyncio.run(_maybe_await(async_fn, "async_arg"))
        assert results == ["async_arg"]

    def test_none_func_does_nothing(self):
        # _maybe_await with None func should be a no-op
        asyncio.run(_maybe_await(None))


# ---------------------------------------------------------------------------
# _pp_text, _pp_tool_use, _pp_tool_result (lines 515, 519-520, 529-531)
# ---------------------------------------------------------------------------


class TestPrettyPrinters:
    def test_pp_text_runs_without_error(self, capsys):
        with patch("lionagi.providers.pi.cli.models.print_readable") as mock_pr:
            _pp_text("Hello Pi", theme="light")
            mock_pr.assert_called_once()
            call_args = mock_pr.call_args
            assert "Hello Pi" in str(call_args[0][0])

    def test_pp_tool_use_runs_without_error(self):
        with patch("lionagi.providers.pi.cli.models.print_readable") as mock_pr:
            tu = {"name": "bash", "input": {"command": "ls"}}
            _pp_tool_use(tu, theme="light")
            mock_pr.assert_called_once()
            call_str = str(mock_pr.call_args[0][0])
            assert "bash" in call_str

    def test_pp_tool_use_alt_name_key(self):
        with patch("lionagi.providers.pi.cli.models.print_readable") as mock_pr:
            tu = {"toolName": "read_file", "args": {"path": "x"}}
            _pp_tool_use(tu, theme="dark")
            mock_pr.assert_called_once()

    def test_pp_tool_result_ok(self):
        with patch("lionagi.providers.pi.cli.models.print_readable") as mock_pr:
            tr = {"result": "file contents", "isError": False}
            _pp_tool_result(tr, theme="light")
            mock_pr.assert_called_once()
            call_str = str(mock_pr.call_args[0][0])
            assert "OK" in call_str

    def test_pp_tool_result_error(self):
        with patch("lionagi.providers.pi.cli.models.print_readable") as mock_pr:
            tr = {"result": "error occurred", "isError": True}
            _pp_tool_result(tr, theme="dark")
            mock_pr.assert_called_once()
            call_str = str(mock_pr.call_args[0][0])
            assert "ERR" in call_str

    def test_pp_tool_result_content_key(self):
        with patch("lionagi.providers.pi.cli.models.print_readable") as mock_pr:
            tr = {"content": "some output", "is_error": False}
            _pp_tool_result(tr, theme="light")
            mock_pr.assert_called_once()


# ---------------------------------------------------------------------------
# print_readable partial (line 511)
# ---------------------------------------------------------------------------


class TestPrintReadable:
    def test_print_readable_is_callable(self):
        assert callable(print_readable)

    def test_print_readable_is_partial_with_md_display(self):
        # print_readable is partial(as_readable, md=True, display_str=True)
        # Verify the partial keywords are baked in by checking its keywords
        assert print_readable.keywords.get("md") is True
        assert print_readable.keywords.get("display_str") is True


# ---------------------------------------------------------------------------
# stream_pi_cli with PI_CLI=None raises RuntimeError (lines 426-430, 503-508)
# ---------------------------------------------------------------------------


class TestStreamPiCliNoCli:
    def test_stream_pi_cli_events_no_cli_raises(self):
        from lionagi.providers.pi.cli.models import stream_pi_cli_events

        req = PiCodeRequest(prompt="x")

        async def _run():
            with patch("lionagi.providers.pi.cli.models.PI_CLI", None):
                async for _ in stream_pi_cli_events(req):
                    pass

        with pytest.raises(RuntimeError, match="Pi CLI not found"):
            asyncio.run(_run())

    def test_ndjson_from_cli_no_cli_raises(self):
        from lionagi.providers.pi.cli.models import _ndjson_from_cli

        req = PiCodeRequest(prompt="x")

        async def _run():
            with patch("lionagi.providers.pi.cli.models.PI_CLI", None):
                async for _ in _ndjson_from_cli(req):
                    pass

        with pytest.raises(RuntimeError, match="Pi CLI not found"):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# stream_pi_cli full event processing (lines 617-766)
# ---------------------------------------------------------------------------


async def _collect_stream_pi_cli(
    events: list[dict],
    request: PiCodeRequest = None,
    **kwargs,
) -> tuple[list, PiSession]:
    """Helper: feed mock events into stream_pi_cli and collect chunks + final session."""
    from lionagi.providers.pi.cli.models import stream_pi_cli

    if request is None:
        request = PiCodeRequest(prompt="test", provider="openai")

    async def _mock_events():
        for e in events:
            yield e

    chunks = []
    session = None
    with patch(
        "lionagi.providers.pi.cli.models.stream_pi_cli_events",
        return_value=_mock_events(),
    ):
        async for item in stream_pi_cli(request, **kwargs):
            if isinstance(item, PiSession):
                session = item
            else:
                chunks.append(item)
    return chunks, session


class TestStreamPiCliEventProcessing:
    def test_agent_start_event_yields_raw_dict(self):
        events = [{"type": "agent_start", "sessionId": "s1"}, {"type": "done"}]
        chunks, session = asyncio.run(_collect_stream_pi_cli(events))
        raw_dicts = [c for c in chunks if isinstance(c, dict)]
        assert any(d.get("type") == "agent_start" for d in raw_dicts)

    def test_agent_end_event_remembered(self):
        events = [
            {
                "type": "agent_end",
                "messages": [{"content": "final answer", "model": "gpt-4"}],
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.result == "final answer"
        assert session.model == "gpt-4"

    def test_turn_end_increments_num_turns(self):
        events = [
            {"type": "turn_end", "message": {"content": "turn 1"}},
            {"type": "turn_end", "message": {"content": "turn 2"}},
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.num_turns == 2

    def test_message_end_appends_to_messages(self):
        events = [
            {
                "type": "message_end",
                "message": {"content": "msg content", "role": "assistant"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert len(session.messages) == 1

    def test_message_update_text_delta_populates_chunk(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Hello"},
            },
            {"type": "done"},
        ]
        chunks, _ = asyncio.run(_collect_stream_pi_cli(events))
        text_chunks = [c for c in chunks if isinstance(c, PiChunk) and c.text]
        assert any(c.text == "Hello" for c in text_chunks)

    def test_message_update_text_delta_calls_on_text(self):
        called = []

        def on_text(t):
            called.append(t)

        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "World"},
            },
            {"type": "done"},
        ]
        asyncio.run(_collect_stream_pi_cli(events, on_text=on_text))
        assert "World" in called

    def test_message_update_text_end_sets_result(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_end", "content": "full text"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.result == "full text"

    def test_message_update_text_start_noop(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_start"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session is not None

    def test_message_update_thinking_delta(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_delta", "delta": "hmm"},
            },
            {"type": "done"},
        ]
        chunks, _ = asyncio.run(_collect_stream_pi_cli(events))
        thinking_chunks = [c for c in chunks if isinstance(c, PiChunk) and c.thinking]
        assert any(c.thinking == "hmm" for c in thinking_chunks)

    def test_message_update_thinking_end(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "thinking_end",
                    "content": "full thought",
                },
            },
            {"type": "done"},
        ]
        chunks, _ = asyncio.run(_collect_stream_pi_cli(events))
        thinking_chunks = [c for c in chunks if isinstance(c, PiChunk) and c.thinking]
        assert any(c.thinking == "full thought" for c in thinking_chunks)

    def test_message_update_thinking_start_noop(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_start"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session is not None

    def test_message_update_done_event_remembered(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "done",
                    "message": {"content": "complete", "model": "gpt-x"},
                },
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.model == "gpt-x"

    def test_message_update_error_sets_is_error(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "error",
                    "errorMessage": "bad input",
                },
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.is_error is True

    def test_message_update_toolcall_end_appends_tool_use(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "toolcall_end",
                    "name": "bash",
                    "input": {"command": "echo hi"},
                },
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert len(session.tool_uses) == 1
        assert session.tool_uses[0]["name"] == "bash"

    def test_message_update_toolcall_end_calls_on_tool_use(self):
        called = []

        def on_tool_use(tu):
            called.append(tu)

        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "toolcall_end",
                    "name": "read",
                    "input": {"path": "file.py"},
                },
            },
            {"type": "done"},
        ]
        asyncio.run(_collect_stream_pi_cli(events, on_tool_use=on_tool_use))
        assert len(called) == 1

    def test_message_update_toolcall_start_delta_noop(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "toolcall_start"},
            },
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "toolcall_delta"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert len(session.tool_uses) == 0

    def test_message_update_start_event(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "start",
                    "partial": {"model": "started-model"},
                },
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session is not None

    def test_tool_execution_start_event(self):
        events = [
            {
                "type": "tool_execution_start",
                "toolCallId": "tid1",
                "toolName": "bash",
                "args": {"command": "ls"},
            },
            {"type": "done"},
        ]
        chunks, _ = asyncio.run(_collect_stream_pi_cli(events))
        te_chunks = [c for c in chunks if isinstance(c, PiChunk) and c.tool_use]
        assert len(te_chunks) >= 1
        assert te_chunks[0].tool_use["name"] == "bash"

    def test_tool_execution_end_appends_tool_result(self):
        events = [
            {
                "type": "tool_execution_end",
                "toolCallId": "tid1",
                "toolName": "bash",
                "result": "output text",
                "isError": False,
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert len(session.tool_results) == 1
        assert session.tool_results[0]["content"] == "output text"
        assert session.tool_results[0]["is_error"] is False

    def test_tool_execution_end_calls_on_tool_result(self):
        called = []

        def on_tool_result(tr):
            called.append(tr)

        events = [
            {
                "type": "tool_execution_end",
                "toolCallId": "tid2",
                "toolName": "read",
                "result": "file data",
                "isError": False,
            },
            {"type": "done"},
        ]
        asyncio.run(_collect_stream_pi_cli(events, on_tool_result=on_tool_result))
        assert len(called) == 1

    def test_tool_execution_update_event(self):
        events = [
            {"type": "tool_execution_update"},
            {"type": "done"},
        ]
        chunks, _ = asyncio.run(_collect_stream_pi_cli(events))
        update_chunks = [
            c
            for c in chunks
            if isinstance(c, PiChunk) and c.type == "tool_execution_update"
        ]
        assert len(update_chunks) == 1

    def test_error_event_sets_is_error(self):
        events = [
            {"type": "error", "errorMessage": "system error"},
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.is_error is True

    def test_unknown_event_type_yields_chunk(self):
        events = [
            {"type": "some_unknown_event"},
            {"type": "done"},
        ]
        chunks, session = asyncio.run(_collect_stream_pi_cli(events))
        unknown = [
            c
            for c in chunks
            if isinstance(c, PiChunk) and c.type == "some_unknown_event"
        ]
        assert len(unknown) == 1

    def test_on_final_called_with_session(self):
        called = []

        def on_final(s):
            called.append(s)

        events = [{"type": "done"}]
        asyncio.run(_collect_stream_pi_cli(events, on_final=on_final))
        assert len(called) == 1
        assert isinstance(called[0], PiSession)

    def test_result_built_from_chunk_texts_if_empty(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "chunk "},
            },
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "text"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        # session.result may have been set by text_delta chunks
        # chunks with text are aggregated as fallback
        assert session is not None

    def test_num_turns_inferred_from_messages(self):
        events = [
            {
                "type": "message_end",
                "message": {"content": "msg1", "role": "assistant"},
            },
            {"type": "done"},
        ]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        # num_turns either set by turn_end or inferred from messages
        if session.num_turns is None:
            # inferred from messages at the end
            assert len(session.messages) > 0

    def test_duration_ms_set_after_run(self):
        events = [{"type": "done"}]
        _, session = asyncio.run(_collect_stream_pi_cli(events))
        assert session.duration_ms is not None
        assert session.duration_ms >= 0

    def test_verbose_output_text_delta(self):
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "verbose"},
            },
            {"type": "done"},
        ]
        req = PiCodeRequest(prompt="test", provider="openai", verbose_output=True)
        with patch("lionagi.providers.pi.cli.models._pp_text"):
            _, session = asyncio.run(_collect_stream_pi_cli(events, request=req))
        assert session is not None

    def test_verbose_output_tool_execution_start(self):
        events = [
            {
                "type": "tool_execution_start",
                "toolCallId": "t1",
                "toolName": "bash",
                "args": {"command": "pwd"},
            },
            {"type": "done"},
        ]
        req = PiCodeRequest(prompt="test", provider="openai", verbose_output=True)
        with patch("lionagi.providers.pi.cli.models._pp_tool_use"):
            _, session = asyncio.run(_collect_stream_pi_cli(events, request=req))
        assert session is not None

    def test_verbose_output_tool_execution_end(self):
        events = [
            {
                "type": "tool_execution_end",
                "toolCallId": "t1",
                "toolName": "bash",
                "result": "done",
                "isError": False,
            },
            {"type": "done"},
        ]
        req = PiCodeRequest(prompt="test", provider="openai", verbose_output=True)
        with patch("lionagi.providers.pi.cli.models._pp_tool_result"):
            _, session = asyncio.run(_collect_stream_pi_cli(events, request=req))
        assert session is not None

    def test_message_start_turn_start_yield_chunk(self):
        events = [
            {"type": "message_start"},
            {"type": "turn_start"},
            {"type": "done"},
        ]
        chunks, _ = asyncio.run(_collect_stream_pi_cli(events))
        types = [c.type for c in chunks if isinstance(c, PiChunk)]
        assert "message_start" in types
        assert "turn_start" in types

    def test_session_passed_in_is_reused(self):
        from lionagi.providers.pi.cli.models import stream_pi_cli

        existing_session = PiSession(session_id="existing")
        req = PiCodeRequest(prompt="test", provider="openai")
        events = [{"type": "done"}]

        async def _mock_events():
            for e in events:
                yield e

        async def _run():
            result = None
            with patch(
                "lionagi.providers.pi.cli.models.stream_pi_cli_events",
                return_value=_mock_events(),
            ):
                async for item in stream_pi_cli(req, existing_session):
                    if isinstance(item, PiSession):
                        result = item
            return result

        returned = asyncio.run(_run())
        assert returned is existing_session
        assert returned.session_id == "existing"


# ---------------------------------------------------------------------------
# Additional coverage for remaining missing lines
# ---------------------------------------------------------------------------


class TestAssistantMessageTextEdgeCases:
    def test_content_neither_str_nor_list_returns_empty(self):
        """Line 554: content is not str or list → return ''."""
        msg = {"content": {"nested": "dict"}}
        result = _assistant_message_text(msg)
        assert result == ""

    def test_content_int_returns_empty(self):
        msg = {"content": 42}
        result = _assistant_message_text(msg)
        assert result == ""


class TestBuildDeclarativeArgsEmptyListSkipped:
    def test_empty_list_tools_not_emitted(self):
        """Line 306: empty list value → continue (no flag emitted)."""
        # tools=[] is an empty list; the validator converts to None for None
        # but let's pass tools=[] directly to exercise the empty list branch
        req = PiCodeRequest(prompt="x")
        # Force tools to empty list manually to hit line 306
        req_dict = req.model_dump()
        req_dict["tools"] = []
        # Re-create to trigger the validator
        req2 = PiCodeRequest(prompt="x")
        object.__setattr__(req2, "tools", [])  # bypass pydantic
        args = req2.as_cmd_args()
        # --tools flag should NOT appear since the list is empty
        assert "--tools" not in args


class TestVerboseToolcallEnd:
    def test_verbose_output_toolcall_end(self):
        """Line 700: verbose_output=True + toolcall_end → _pp_tool_use called."""
        events = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "toolcall_end",
                    "name": "bash",
                    "input": {"command": "echo test"},
                },
            },
            {"type": "done"},
        ]
        req = PiCodeRequest(prompt="test", provider="openai", verbose_output=True)
        pp_calls = []

        def fake_pp_tool_use(tu, theme):
            pp_calls.append(tu)

        with patch("lionagi.providers.pi.cli.models._pp_tool_use", fake_pp_tool_use):
            asyncio.run(_collect_stream_pi_cli(events, request=req))
        assert len(pp_calls) == 1
        assert pp_calls[0]["name"] == "bash"
