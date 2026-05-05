# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.providers.openai.codex.models."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from lionagi.providers.openai.codex.models import (
    CodexChunk,
    CodexCodeRequest,
    CodexSession,
)


class TestCodexCodeRequestValidation:
    def test_minimal_with_prompt(self):
        req = CodexCodeRequest(prompt="Write a function")
        assert req.prompt == "Write a function"
        assert req.model == "gpt-5.3-codex"

    def test_derive_prompt_from_messages(self):
        req = CodexCodeRequest(
            messages=[
                {"role": "user", "content": "Refactor this code"},
            ]
        )
        assert req.prompt == "Refactor this code"

    def test_derive_prompt_from_messages_system_extracted(self):
        req = CodexCodeRequest(
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Write tests"},
            ]
        )
        assert "Write tests" in req.prompt
        assert req.system_prompt == "Be concise"

    def test_no_messages_or_prompt_raises(self):
        with pytest.raises(ValueError, match="messages or prompt required"):
            CodexCodeRequest(messages=[])

    def test_clamp_max_to_xhigh_reasoning_effort(self):
        req = CodexCodeRequest(prompt="x", reasoning_effort="max")
        assert req.reasoning_effort == "xhigh"

    def test_clamp_max_to_xhigh_plan_mode(self):
        req = CodexCodeRequest(prompt="x", plan_mode_reasoning_effort="max")
        assert req.plan_mode_reasoning_effort == "xhigh"

    def test_valid_reasoning_effort_passthrough(self):
        req = CodexCodeRequest(prompt="x", reasoning_effort="high")
        assert req.reasoning_effort == "high"

    def test_norm_add_dir_string_to_list(self):
        req = CodexCodeRequest(prompt="x", add_dir="/some/dir")
        assert req.add_dir == ["/some/dir"]

    def test_norm_add_dir_list_passthrough(self):
        req = CodexCodeRequest(prompt="x", add_dir=["/a", "/b"])
        assert req.add_dir == ["/a", "/b"]

    def test_bypass_approvals_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            req = CodexCodeRequest(prompt="x", bypass_approvals=True)
            assert any("bypass_approvals" in str(warning.message) for warning in w)


class TestCodexCodeRequestCwd:
    def test_cwd_no_ws_returns_repo(self, tmp_path):
        req = CodexCodeRequest(prompt="x", repo=tmp_path)
        assert req.cwd() == tmp_path

    def test_cwd_relative_ws(self, tmp_path):
        (tmp_path / "work").mkdir()
        req = CodexCodeRequest(prompt="x", repo=tmp_path, ws="work")
        assert req.cwd() == (tmp_path / "work").resolve()

    def test_cwd_absolute_ws_raises(self, tmp_path):
        req = CodexCodeRequest(prompt="x", repo=tmp_path, ws="/absolute")
        with pytest.raises(ValueError, match="relative"):
            req.cwd()

    def test_cwd_dotdot_raises(self, tmp_path):
        req = CodexCodeRequest(prompt="x", repo=tmp_path, ws="../escape")
        with pytest.raises(ValueError, match="Directory traversal"):
            req.cwd()


class TestCodexCodeRequestCmdArgs:
    def test_basic_structure(self, tmp_path):
        req = CodexCodeRequest(prompt="fix bugs", repo=tmp_path)
        args = req.as_cmd_args()
        assert "exec" in args
        assert "--json" in args
        assert "--" in args
        assert "fix bugs" in args
        assert "-C" in args

    def test_model_flag(self, tmp_path):
        req = CodexCodeRequest(prompt="x", model="gpt-4o", repo=tmp_path)
        args = req.as_cmd_args()
        assert "-m" in args
        assert "gpt-4o" in args

    def test_search_enabled(self, tmp_path):
        req = CodexCodeRequest(prompt="x", search=True, repo=tmp_path)
        args = req.as_cmd_args()
        assert "--enable" in args
        assert "tool_search" in args

    def test_search_disabled(self, tmp_path):
        req = CodexCodeRequest(prompt="x", search=False, repo=tmp_path)
        args = req.as_cmd_args()
        assert "--disable" in args
        assert "tool_search" in args

    def test_bypass_approvals(self, tmp_path):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            req = CodexCodeRequest(prompt="x", bypass_approvals=True, repo=tmp_path)
        args = req.as_cmd_args()
        assert "--dangerously-bypass-approvals-and-sandbox" in args

    def test_full_auto(self, tmp_path):
        req = CodexCodeRequest(prompt="x", full_auto=True, repo=tmp_path)
        args = req.as_cmd_args()
        assert "--full-auto" in args

    def test_ask_for_approval(self, tmp_path):
        req = CodexCodeRequest(prompt="x", ask_for_approval="on-request", repo=tmp_path)
        args = req.as_cmd_args()
        assert "-a" in args
        assert "on-request" in args

    def test_sandbox_mode(self, tmp_path):
        req = CodexCodeRequest(prompt="x", sandbox="read-only", repo=tmp_path)
        args = req.as_cmd_args()
        assert "-s" in args
        assert "read-only" in args

    def test_system_prompt_as_config_override(self, tmp_path):
        req = CodexCodeRequest(prompt="x", system_prompt="Be precise", repo=tmp_path)
        args = req.as_cmd_args()
        assert "-c" in args
        assert "developer_instructions=Be precise" in args

    def test_reasoning_effort_as_config(self, tmp_path):
        req = CodexCodeRequest(prompt="x", reasoning_effort="high", repo=tmp_path)
        args = req.as_cmd_args()
        assert "reasoning_effort=high" in args

    def test_images(self, tmp_path):
        req = CodexCodeRequest(
            prompt="x", images=["img1.png", "img2.png"], repo=tmp_path
        )
        args = req.as_cmd_args()
        assert "-i" in args
        assert "img1.png" in args
        assert "img2.png" in args

    def test_config_overrides(self, tmp_path):
        req = CodexCodeRequest(
            prompt="x", config_overrides={"key": "val"}, repo=tmp_path
        )
        args = req.as_cmd_args()
        assert "key=val" in args

    def test_config_overrides_dict_serialized(self, tmp_path):
        req = CodexCodeRequest(
            prompt="x",
            config_overrides={"nested": {"a": 1}},
            repo=tmp_path,
        )
        args = req.as_cmd_args()
        c_idx = [i for i, v in enumerate(args) if v == "-c"]
        # Should have at least one -c for the nested config
        assert len(c_idx) >= 1

    def test_oss_flag(self, tmp_path):
        req = CodexCodeRequest(prompt="x", oss=True, repo=tmp_path)
        args = req.as_cmd_args()
        assert "--oss" in args

    def test_skip_git_repo_check(self, tmp_path):
        req = CodexCodeRequest(prompt="x", skip_git_repo_check=True, repo=tmp_path)
        args = req.as_cmd_args()
        assert "--skip-git-repo-check" in args

    def test_add_dir_repeat(self, tmp_path):
        req = CodexCodeRequest(prompt="x", add_dir=["/a", "/b"], repo=tmp_path)
        args = req.as_cmd_args()
        assert "--add-dir" in args
        assert "/a" in args
        assert "/b" in args


class TestCodexChunkAndSession:
    def test_chunk_creation(self):
        raw = {"type": "text_delta", "text": "partial"}
        chunk = CodexChunk(raw=raw, type="text_delta", text="partial")
        assert chunk.type == "text_delta"
        assert chunk.text == "partial"
        assert chunk.tool_use is None
        assert chunk.tool_result is None

    def test_session_defaults(self):
        session = CodexSession()
        assert session.session_id is None
        assert session.result == ""
        assert session.is_error is False
        assert session.chunks == []
        assert session.messages == []
        assert session.tool_uses == []
        assert session.tool_results == []

    def test_populate_summary(self):
        session = CodexSession(
            result="Task done",
            tool_uses=[
                {"name": "Read", "id": "t1", "input": {"path": "file.py"}},
                {"name": "Write", "id": "t2", "input": {"path": "out.py"}},
                {"name": "Bash", "id": "t3", "input": {"command": "pytest"}},
            ],
        )
        session.populate_summary()
        assert session.summary is not None
        s = session.summary
        assert s["total_tool_calls"] == 3
        assert "Read" in s["tool_counts"]
        assert len(s["file_operations"]["reads"]) == 1
        assert len(s["file_operations"]["writes"]) == 1
        assert any("pytest" in a for a in s["key_actions"])

    def test_populate_summary_empty(self):
        session = CodexSession()
        session.populate_summary()
        assert session.summary is not None
        assert session.summary["total_tool_calls"] == 0
