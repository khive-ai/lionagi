# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.google.gemini_code.models."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from lionagi.providers.google.gemini_code.models import (
    GeminiChunk,
    GeminiCodeRequest,
    GeminiSession,
)


class TestGeminiCodeRequestValidation:
    def test_minimal_with_prompt(self):
        req = GeminiCodeRequest(prompt="Explain async/await")
        assert req.prompt == "Explain async/await"
        assert req.model == "gemini-2.5-pro"
        assert req.sandbox is True

    def test_derive_prompt_from_messages(self):
        req = GeminiCodeRequest(
            messages=[
                {"role": "user", "content": "Write a hello world"},
            ]
        )
        assert req.prompt == "Write a hello world"

    def test_system_message_extracted(self):
        req = GeminiCodeRequest(
            messages=[
                {"role": "system", "content": "You are a Python expert"},
                {"role": "user", "content": "Write a class"},
            ]
        )
        assert req.system_prompt == "You are a Python expert"
        assert "Write a class" in req.prompt

    def test_no_messages_or_prompt_raises(self):
        with pytest.raises(ValueError, match="messages or prompt required"):
            GeminiCodeRequest(messages=[])

    def test_dict_list_content_serialized(self):
        req = GeminiCodeRequest(
            messages=[
                {"role": "user", "content": {"key": "value"}},
            ]
        )
        assert isinstance(req.prompt, str)
        assert len(req.prompt) > 0

    def test_yolo_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            req = GeminiCodeRequest(prompt="x", yolo=True)
            assert any("yolo" in str(warning.message).lower() for warning in w)

    def test_no_sandbox_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            req = GeminiCodeRequest(prompt="x", sandbox=False)
            assert any("sandbox" in str(warning.message).lower() for warning in w)

    def test_defaults(self):
        req = GeminiCodeRequest(prompt="x")
        assert req.yolo is False
        assert req.sandbox is True
        assert req.debug is False
        assert req.verbose_output is False
        assert req.include_directories == []
        assert req.mcp_tools == []


class TestGeminiCodeRequestCwd:
    def test_cwd_no_ws_returns_repo(self, tmp_path):
        req = GeminiCodeRequest(prompt="x", repo=tmp_path)
        assert req.cwd() == tmp_path

    def test_cwd_relative_ws(self, tmp_path):
        (tmp_path / "workspace").mkdir()
        req = GeminiCodeRequest(prompt="x", repo=tmp_path, ws="workspace")
        assert req.cwd() == (tmp_path / "workspace").resolve()

    def test_cwd_absolute_ws_raises(self, tmp_path):
        req = GeminiCodeRequest(prompt="x", repo=tmp_path, ws="/absolute/path")
        with pytest.raises(ValueError, match="relative"):
            req.cwd()

    def test_cwd_dotdot_raises(self, tmp_path):
        req = GeminiCodeRequest(prompt="x", repo=tmp_path, ws="../escape")
        with pytest.raises(ValueError, match="Directory traversal"):
            req.cwd()


class TestGeminiCodeRequestCmdArgs:
    def test_basic_structure(self):
        req = GeminiCodeRequest(prompt="review this code")
        args = req.as_cmd_args()
        assert "-p" in args
        assert "review this code" in args
        assert "--output-format" in args
        assert "stream-json" in args

    def test_model_included(self):
        req = GeminiCodeRequest(prompt="x", model="gemini-2.5-flash")
        args = req.as_cmd_args()
        assert "-m" in args
        assert "gemini-2.5-flash" in args

    def test_yolo_flag(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            req = GeminiCodeRequest(prompt="x", yolo=True)
        args = req.as_cmd_args()
        assert "--yolo" in args

    def test_approval_mode_flag(self):
        req = GeminiCodeRequest(prompt="x", approval_mode="auto_edit")
        args = req.as_cmd_args()
        assert "--approval-mode" in args
        assert "auto_edit" in args

    def test_debug_flag(self):
        req = GeminiCodeRequest(prompt="x", debug=True)
        args = req.as_cmd_args()
        assert "--debug" in args

    def test_no_sandbox_flag(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            req = GeminiCodeRequest(prompt="x", sandbox=False)
        args = req.as_cmd_args()
        assert "--no-sandbox" in args

    def test_sandbox_true_no_flag(self):
        req = GeminiCodeRequest(prompt="x", sandbox=True)
        args = req.as_cmd_args()
        assert "--no-sandbox" not in args

    def test_include_directories(self):
        req = GeminiCodeRequest(prompt="x", include_directories=["/src", "/tests"])
        args = req.as_cmd_args()
        assert "--include-directories" in args
        assert "/src" in args
        assert "/tests" in args

    def test_no_model_not_included_when_none(self):
        req = GeminiCodeRequest(prompt="x", model=None)
        args = req.as_cmd_args()
        assert "-m" not in args


class TestGeminiChunkAndSession:
    def test_chunk_creation(self):
        raw = {"type": "text", "text": "hello"}
        chunk = GeminiChunk(raw=raw, type="text", text="hello")
        assert chunk.type == "text"
        assert chunk.text == "hello"
        assert chunk.tool_use is None
        assert chunk.is_delta is False

    def test_session_defaults(self):
        session = GeminiSession()
        assert session.session_id is None
        assert session.result == ""
        assert session.is_error is False
        assert session.chunks == []
        assert session.summary is None

    def test_populate_summary_with_tools(self):
        session = GeminiSession(
            result="Completed",
            tool_uses=[
                {"name": "Read", "id": "r1", "input": {"path": "main.py"}},
                {"name": "Write", "id": "w1", "input": {"path": "output.py"}},
                {"name": "Edit", "id": "e1", "input": {"path": "edit.py"}},
                {
                    "name": "run_shell_command",
                    "id": "s1",
                    "input": {"command": "pytest -v"},
                },
            ],
        )
        session.populate_summary()
        s = session.summary
        assert s is not None
        assert s["total_tool_calls"] == 4
        assert len(s["file_operations"]["reads"]) == 1
        assert len(s["file_operations"]["writes"]) == 1
        assert len(s["file_operations"]["edits"]) == 1
        assert any("pytest" in a for a in s["key_actions"])

    def test_populate_summary_mcp_tool(self):
        session = GeminiSession(
            tool_uses=[
                {"name": "mcp_search", "id": "m1", "input": {"query": "python"}},
            ],
        )
        session.populate_summary()
        s = session.summary
        assert any("MCP" in a for a in s["key_actions"])

    def test_populate_summary_deduplicates_key_actions(self):
        session = GeminiSession(
            tool_uses=[
                {"name": "Read", "id": "r1", "input": {"path": "file.py"}},
                {"name": "Read", "id": "r2", "input": {"path": "file.py"}},
            ],
        )
        session.populate_summary()
        # key_actions are deduplicated
        actions = session.summary["key_actions"]
        assert len(actions) == len(set(actions))

    def test_populate_summary_empty_session(self):
        session = GeminiSession()
        session.populate_summary()
        assert session.summary is not None
        assert session.summary["total_tool_calls"] == 0
        assert session.summary["key_actions"] == ["No specific actions"]
