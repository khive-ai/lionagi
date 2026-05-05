# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.anthropic.claude_code.models."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from lionagi.providers.anthropic.claude_code.models import (
    ClaudeChunk,
    ClaudeCodeRequest,
    ClaudeSession,
)


class TestClaudeCodeRequestValidation:
    def test_minimal_with_prompt(self):
        req = ClaudeCodeRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.model == "sonnet"
        assert req.output_format == "stream-json"

    def test_derive_prompt_from_messages(self):
        req = ClaudeCodeRequest(
            messages=[
                {"role": "user", "content": "What is Python?"},
            ]
        )
        assert req.prompt == "What is Python?"
        assert req.continue_conversation is False

    def test_derive_prompt_with_resume(self):
        req = ClaudeCodeRequest(
            resume="session-abc",
            messages=[{"role": "user", "content": "continue this"}],
        )
        assert req.prompt == "continue this"
        # resume clears continue_conversation per _check_constraints
        assert req.continue_conversation is False
        assert req.resume == "session-abc"

    def test_derive_prompt_with_continue_conversation(self):
        req = ClaudeCodeRequest(
            continue_conversation=True,
            messages=[{"role": "user", "content": "keep going"}],
        )
        assert req.prompt == "keep going"

    def test_system_message_becomes_append_system_prompt(self):
        req = ClaudeCodeRequest(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Tell me a joke"},
            ]
        )
        assert "Tell me a joke" in req.prompt
        assert req.append_system_prompt == "Be concise."

    def test_system_message_with_resume_becomes_system_prompt(self):
        req = ClaudeCodeRequest(
            resume="sess-1",
            messages=[
                {"role": "system", "content": "System context."},
                {"role": "user", "content": "hello"},
            ],
        )
        assert req.system_prompt == "System context."

    def test_no_messages_or_prompt_raises(self):
        with pytest.raises(ValueError, match="messages may not be empty"):
            ClaudeCodeRequest(messages=[])

    def test_fork_session_without_resume_raises(self):
        with pytest.raises(ValueError, match="--fork-session requires"):
            ClaudeCodeRequest(prompt="x", fork_session=True)

    def test_fork_session_with_resume_ok(self):
        req = ClaudeCodeRequest(prompt="x", fork_session=True, resume="sess-1")
        assert req.fork_session is True

    def test_skip_permissions_and_bypass_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ClaudeCodeRequest(
                prompt="x",
                allow_dangerously_skip_permissions=True,
                permission_mode="bypassPermissions",
            )

    def test_system_prompt_and_file_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ClaudeCodeRequest(
                prompt="x",
                system_prompt="inline",
                system_prompt_file="/path/to/file",
            )

    def test_norm_perm_dangerous_alias(self):
        req = ClaudeCodeRequest(
            prompt="x",
            permission_mode="dangerously-skip-permissions",
        )
        assert req.permission_mode == "bypassPermissions"

    def test_norm_add_dir_string_to_list(self):
        req = ClaudeCodeRequest(prompt="x", add_dir="/some/dir")
        assert req.add_dir == ["/some/dir"]

    def test_resume_clears_continue_conversation(self):
        req = ClaudeCodeRequest(
            prompt="x",
            resume="sess",
            continue_conversation=True,
        )
        assert req.resume == "sess"
        assert req.continue_conversation is False


class TestClaudeCodeRequestCwd:
    def test_cwd_no_ws_returns_repo(self, tmp_path):
        req = ClaudeCodeRequest(prompt="x", repo=tmp_path)
        assert req.cwd() == tmp_path

    def test_cwd_relative_ws(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        req = ClaudeCodeRequest(prompt="x", repo=tmp_path, ws="subdir")
        assert req.cwd() == (tmp_path / "subdir").resolve()

    def test_cwd_absolute_ws_raises(self, tmp_path):
        req = ClaudeCodeRequest(prompt="x", repo=tmp_path, ws="/absolute/path")
        with pytest.raises(ValueError, match="relative"):
            req.cwd()

    def test_cwd_dotdot_ws_raises(self, tmp_path):
        req = ClaudeCodeRequest(prompt="x", repo=tmp_path, ws="../escape")
        with pytest.raises(ValueError, match="Directory traversal"):
            req.cwd()


class TestClaudeCodeRequestCmdArgs:
    def test_basic_args_structure(self):
        req = ClaudeCodeRequest(prompt="fix the bug")
        args = req.as_cmd_args()
        assert "-p" in args
        assert "fix the bug" in args
        assert "--output-format" in args
        assert "stream-json" in args
        assert "--verbose" in args

    def test_model_included(self):
        req = ClaudeCodeRequest(prompt="x", model="opus")
        args = req.as_cmd_args()
        assert "--model" in args
        assert "opus" in args

    def test_continue_flag(self):
        req = ClaudeCodeRequest(
            continue_conversation=True,
            messages=[{"role": "user", "content": "continue"}],
        )
        args = req.as_cmd_args()
        assert "--continue" in args

    def test_resume_flag(self):
        req = ClaudeCodeRequest(prompt="x", resume="sess-123")
        args = req.as_cmd_args()
        assert "--resume" in args
        assert "sess-123" in args

    def test_permission_mode_bypass(self):
        req = ClaudeCodeRequest(
            prompt="x",
            permission_mode="bypassPermissions",
        )
        args = req.as_cmd_args()
        assert "--dangerously-skip-permissions" in args

    def test_permission_mode_other(self):
        req = ClaudeCodeRequest(prompt="x", permission_mode="acceptEdits")
        args = req.as_cmd_args()
        assert "--permission-mode" in args
        assert "acceptEdits" in args

    def test_max_turns_offset(self):
        req = ClaudeCodeRequest(prompt="x", max_turns=4)
        args = req.as_cmd_args()
        idx = args.index("--max-turns")
        assert args[idx + 1] == "5"  # +1 offset

    def test_worktree_true(self):
        req = ClaudeCodeRequest(prompt="x", worktree=True)
        args = req.as_cmd_args()
        assert "--worktree" in args

    def test_worktree_string(self):
        req = ClaudeCodeRequest(prompt="x", worktree="my-branch")
        args = req.as_cmd_args()
        assert "--worktree" in args
        assert "my-branch" in args

    def test_debug_true(self):
        req = ClaudeCodeRequest(prompt="x", debug=True)
        args = req.as_cmd_args()
        assert "--debug" in args

    def test_debug_string(self):
        req = ClaudeCodeRequest(prompt="x", debug="all")
        args = req.as_cmd_args()
        assert "--debug" in args
        assert "all" in args

    def test_legacy_mcp_servers(self):
        req = ClaudeCodeRequest(
            prompt="x",
            mcp_servers={"myserver": {"command": "npx", "args": ["-y", "mcp-server"]}},
        )
        args = req.as_cmd_args()
        assert "--mcp-config" in args

    def test_add_dir_list(self):
        req = ClaudeCodeRequest(prompt="x", add_dir=["/a", "/b"])
        args = req.as_cmd_args()
        assert "--add-dir" in args

    def test_chrome_bool_pair_true(self):
        req = ClaudeCodeRequest(prompt="x", chrome=True)
        args = req.as_cmd_args()
        assert "--chrome" in args

    def test_chrome_bool_pair_false(self):
        req = ClaudeCodeRequest(prompt="x", chrome=False)
        args = req.as_cmd_args()
        assert "--no-chrome" in args

    def test_allowed_tools_list_args(self):
        req = ClaudeCodeRequest(prompt="x", allowed_tools=["Read", "Write"])
        args = req.as_cmd_args()
        assert "--allowedTools" in args
        assert "Read" in args
        assert "Write" in args

    def test_agents_json_value(self):
        req = ClaudeCodeRequest(prompt="x", agents={"bot": {"model": "sonnet"}})
        args = req.as_cmd_args()
        assert "--agents" in args


class TestClaudeChunkAndSession:
    def test_claude_chunk_creation(self):
        raw = {"type": "text", "text": "hello", "session_id": "sess-1"}
        chunk = ClaudeChunk(raw=raw, type="text", text="hello")
        assert chunk.type == "text"
        assert chunk.text == "hello"
        assert chunk.thinking is None
        assert chunk.tool_use is None
        assert chunk.tool_result is None

    def test_claude_session_defaults(self):
        session = ClaudeSession()
        assert session.session_id is None
        assert session.result == ""
        assert session.is_error is False
        assert session.chunks == []
        assert session.messages == []
        assert session.tool_uses == []
        assert session.tool_results == []
        assert session.usage == {}
        assert session.summary is None

    def test_populate_summary_runs(self):
        session = ClaudeSession(
            result="done",
            tool_uses=[
                {"name": "Read", "id": "1", "input": {"path": "/file.py"}},
                {"name": "Bash", "id": "2", "input": {"command": "ls -la"}},
            ],
        )
        session.populate_summary()
        assert session.summary is not None
        assert "tool_counts" in session.summary
        assert session.summary["total_tool_calls"] == 2
