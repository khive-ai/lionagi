# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI stream internals: _ndjson_from_cli, stream_*_cli_events,
_maybe_await, _pp_* display helpers, and _extract_summary across all three
CLI providers: Codex, Claude Code, and Gemini.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Subprocess mock helpers
# ---------------------------------------------------------------------------


class FakeStreamReader:
    """Minimal asyncio StreamReader stand-in.

    The ``n`` argument is optional so it can stand in for both
    ``StreamReader.read(n)`` (stdout loop) and ``StreamReader.read()``
    (stderr drain in error paths).
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, n: int = -1) -> bytes:
        if not self._chunks:
            return b""
        if n == -1:
            # Drain all remaining chunks at once (mirrors StreamReader.read())
            data = b"".join(self._chunks)
            self._chunks.clear()
            return data
        return self._chunks.pop(0)


class FakeProcess:
    """Minimal asyncio Process stand-in."""

    def __init__(
        self,
        stdout_chunks: list[bytes],
        returncode: int = 0,
        stderr_data: bytes = b"",
    ) -> None:
        # Append sentinel empty bytes so the read-loop terminates cleanly
        self.stdout = FakeStreamReader(list(stdout_chunks) + [b""])
        self.stderr = FakeStreamReader([stderr_data, b""])
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _ndjson_bytes(*objs: dict) -> bytes:
    """Encode a sequence of dicts as NDJSON bytes."""
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


# ===========================================================================
# CODEX
# ===========================================================================

import lionagi.providers.openai.codex.models as codex_module
from lionagi.providers.openai.codex.models import CodexCodeRequest, CodexSession
from lionagi.providers.openai.codex.models import (
    _extract_summary as codex_extract_summary,
)
from lionagi.providers.openai.codex.models import _maybe_await as codex_maybe_await
from lionagi.providers.openai.codex.models import (
    _ndjson_from_cli as codex_ndjson_from_cli,
)
from lionagi.providers.openai.codex.models import _pp_final as codex_pp_final
from lionagi.providers.openai.codex.models import _pp_text as codex_pp_text
from lionagi.providers.openai.codex.models import (
    _pp_tool_result as codex_pp_tool_result,
)
from lionagi.providers.openai.codex.models import _pp_tool_use as codex_pp_tool_use
from lionagi.providers.openai.codex.models import stream_codex_cli_events


class TestCodexExtractSummary:
    def _sess(self, tool_uses: list[dict], result: str = "") -> CodexSession:
        s = CodexSession()
        s.tool_uses = tool_uses
        s.result = result
        return s

    def test_empty_tool_uses_returns_no_actions(self):
        s = self._sess([])
        out = codex_extract_summary(s)
        assert out["key_actions"] == ["No specific actions"]
        assert out["total_tool_calls"] == 0
        assert out["file_operations"] == {"reads": [], "writes": [], "edits": []}

    def test_read_file_tool(self):
        tu = {"name": "read_file", "id": "r1", "input": {"path": "/a/b.py"}}
        out = codex_extract_summary(self._sess([tu]))
        assert "Read /a/b.py" in out["key_actions"]
        assert "/a/b.py" in out["file_operations"]["reads"]

    def test_read_aliases(self):
        for name in ("Read", "read"):
            tu = {"name": name, "id": "x", "input": {"path": "/x.py"}}
            out = codex_extract_summary(self._sess([tu]))
            assert "/x.py" in out["file_operations"]["reads"]

    def test_write_tool(self):
        tu = {"name": "write_file", "id": "w1", "input": {"path": "/out.txt"}}
        out = codex_extract_summary(self._sess([tu]))
        assert "Wrote /out.txt" in out["key_actions"]
        assert "/out.txt" in out["file_operations"]["writes"]

    def test_write_aliases(self):
        for name in ("create_file", "Write", "write"):
            tu = {"name": name, "id": "x", "input": {"path": "/f.txt"}}
            out = codex_extract_summary(self._sess([tu]))
            assert "/f.txt" in out["file_operations"]["writes"]

    def test_edit_tool(self):
        tu = {"name": "edit_file", "id": "e1", "input": {"path": "/m.py"}}
        out = codex_extract_summary(self._sess([tu]))
        assert "Edited /m.py" in out["key_actions"]
        assert "/m.py" in out["file_operations"]["edits"]

    def test_edit_aliases(self):
        for name in ("patch", "Edit", "edit"):
            tu = {"name": name, "id": "x", "input": {"path": "/p.py"}}
            out = codex_extract_summary(self._sess([tu]))
            assert "/p.py" in out["file_operations"]["edits"]

    def test_bash_tool_short_command(self):
        tu = {"name": "bash", "id": "b1", "input": {"command": "ls -la"}}
        out = codex_extract_summary(self._sess([tu]))
        assert "Ran: ls -la" in out["key_actions"]

    def test_bash_tool_long_command_truncated(self):
        long_cmd = "x" * 100
        tu = {"name": "Bash", "id": "b2", "input": {"command": long_cmd}}
        out = codex_extract_summary(self._sess([tu]))
        action = next(a for a in out["key_actions"] if a.startswith("Ran:"))
        assert action.endswith("...")
        # truncated at 50 chars + "..."
        assert len(action) == len("Ran: ") + 50 + 3

    def test_bash_aliases(self):
        for name in ("shell", "terminal", "run_shell_command", "Bash"):
            tu = {"name": name, "id": "x", "input": {"command": "pwd"}}
            out = codex_extract_summary(self._sess([tu]))
            assert any(a.startswith("Ran:") for a in out["key_actions"])

    def test_mcp_tool_with_mcp_double_underscore(self):
        tu = {"name": "mcp__khive__memory", "id": "m1", "input": {}}
        out = codex_extract_summary(self._sess([tu]))
        assert any("MCP" in a for a in out["key_actions"])

    def test_mcp_tool_with_mcp_single_underscore(self):
        tu = {"name": "mcp_some_op", "id": "m2", "input": {}}
        out = codex_extract_summary(self._sess([tu]))
        assert any("MCP" in a for a in out["key_actions"])

    def test_unknown_tool(self):
        tu = {"name": "fancy_tool", "id": "u1", "input": {}}
        out = codex_extract_summary(self._sess([tu]))
        assert "Used fancy_tool" in out["key_actions"]

    def test_deduplication_of_key_actions(self):
        tu1 = {"name": "read_file", "id": "r1", "input": {"path": "/same.py"}}
        tu2 = {"name": "read_file", "id": "r2", "input": {"path": "/same.py"}}
        out = codex_extract_summary(self._sess([tu1, tu2]))
        assert out["key_actions"].count("Read /same.py") == 1
        assert out["file_operations"]["reads"].count("/same.py") == 1

    def test_deduplication_of_file_paths(self):
        tu1 = {"name": "write_file", "id": "w1", "input": {"path": "/dup.py"}}
        tu2 = {"name": "write_file", "id": "w2", "input": {"path": "/dup.py"}}
        out = codex_extract_summary(self._sess([tu1, tu2]))
        assert out["file_operations"]["writes"].count("/dup.py") == 1

    def test_result_truncation(self):
        long_result = "x" * 300
        out = codex_extract_summary(self._sess([], result=long_result))
        assert out["result_summary"].endswith("...")
        assert len(out["result_summary"]) == 203  # 200 + len("...")

    def test_result_not_truncated_when_short(self):
        out = codex_extract_summary(self._sess([], result="short"))
        assert out["result_summary"] == "short"

    def test_total_tool_calls_count(self):
        tu_list = [
            {"name": "read_file", "id": f"r{i}", "input": {"path": f"/f{i}.py"}}
            for i in range(4)
        ]
        out = codex_extract_summary(self._sess(tu_list))
        assert out["total_tool_calls"] == 4

    def test_tool_counts_per_name(self):
        tu_list = [
            {"name": "read_file", "id": "r1", "input": {"path": "/a.py"}},
            {"name": "read_file", "id": "r2", "input": {"path": "/b.py"}},
            {"name": "bash", "id": "b1", "input": {"command": "echo hi"}},
        ]
        out = codex_extract_summary(self._sess(tu_list))
        assert out["tool_counts"]["read_file"] == 2
        assert out["tool_counts"]["bash"] == 1

    def test_populate_summary_sets_attribute(self):
        s = CodexSession()
        s.populate_summary()
        assert isinstance(s.summary, dict)
        assert "key_actions" in s.summary


class TestCodexMaybeAwait:
    @pytest.mark.asyncio
    async def test_sync_callback_called_with_args(self):
        called = []

        def sync_cb(x, y):
            called.append((x, y))

        await codex_maybe_await(sync_cb, 1, 2)
        assert called == [(1, 2)]

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self):
        called = []

        async def async_cb(x):
            called.append(x)

        await codex_maybe_await(async_cb, "hello")
        assert called == ["hello"]

    @pytest.mark.asyncio
    async def test_none_func_no_error(self):
        # Should not raise
        await codex_maybe_await(None, "ignored")

    @pytest.mark.asyncio
    async def test_kwargs_forwarded(self):
        received = {}

        def kw_cb(**kw):
            received.update(kw)

        await codex_maybe_await(kw_cb, key="val")
        assert received == {"key": "val"}


class TestCodexPpFunctions:
    """Verify _pp_* helpers run without raising exceptions."""

    def test_pp_text_no_raise(self):
        codex_pp_text("hello world", theme="light")

    def test_pp_text_dark_theme(self):
        codex_pp_text("hello world", theme="dark")

    def test_pp_tool_use_no_raise(self):
        tu = {"name": "read_file", "id": "r1", "input": {"path": "/a.py"}}
        codex_pp_tool_use(tu, theme="light")

    def test_pp_tool_use_empty_input(self):
        tu = {"name": "unknown", "id": "", "input": {}}
        codex_pp_tool_use(tu, theme="light")

    def test_pp_tool_result_no_raise(self):
        tr = {"content": "output data", "is_error": False}
        codex_pp_tool_result(tr, theme="light")

    def test_pp_tool_result_error(self):
        tr = {"content": "err msg", "is_error": True}
        codex_pp_tool_result(tr, theme="light")

    def test_pp_final_no_raise(self):
        s = CodexSession()
        s.result = "done"
        s.total_cost_usd = 0.0012
        s.num_turns = 3
        s.duration_ms = 500
        s.usage = {"input_tokens": 100, "output_tokens": 50}
        codex_pp_final(s, theme="light")

    def test_pp_final_no_cost(self):
        s = CodexSession()
        s.total_cost_usd = None
        codex_pp_final(s, theme="light")


class TestCodexNdjsonFromCli:
    @pytest.mark.asyncio
    async def test_valid_ndjson_stream(self, monkeypatch, tmp_path):
        objs = [{"type": "text", "text": "hello"}, {"type": "result", "result": "done"}]
        fake_proc = FakeProcess([_ndjson_bytes(*objs)])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(codex_module, "CODEX_CLI", "codex")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        collected = []
        async for obj in codex_ndjson_from_cli(req):
            collected.append(obj)

        assert collected == objs

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_runtime_error(self, monkeypatch, tmp_path):
        fake_proc = FakeProcess([], returncode=1, stderr_data=b"something failed")

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(codex_module, "CODEX_CLI", "codex")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        with pytest.raises(RuntimeError, match="something failed"):
            async for _ in codex_ndjson_from_cli(req):
                pass

    @pytest.mark.asyncio
    async def test_invalid_json_tail_skipped(self, monkeypatch, tmp_path, caplog):
        # Valid chunk then invalid tail
        valid = json.dumps({"type": "text", "text": "hi"}).encode()
        invalid_tail = b"THIS IS NOT JSON"

        fake_proc = FakeProcess([valid + b"\n", invalid_tail])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(codex_module, "CODEX_CLI", "codex")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        import logging

        with caplog.at_level(logging.ERROR, logger="codex-cli"):
            collected = []
            async for obj in codex_ndjson_from_cli(req):
                collected.append(obj)

        assert collected == [{"type": "text", "text": "hi"}]

    @pytest.mark.asyncio
    async def test_cli_not_installed_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_module, "CODEX_CLI", None)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        with pytest.raises(RuntimeError, match="Codex CLI not found"):
            async for _ in codex_ndjson_from_cli(req):
                pass

    @pytest.mark.asyncio
    async def test_multiple_chunks_assembled(self, monkeypatch, tmp_path):
        """Objects split across multiple read() calls are assembled correctly."""
        obj = {"type": "text", "text": "world"}
        encoded = json.dumps(obj).encode() + b"\n"
        # Split the single JSON object across two read() calls
        mid = len(encoded) // 2
        fake_proc = FakeProcess([encoded[:mid], encoded[mid:]])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(codex_module, "CODEX_CLI", "codex")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        collected = []
        async for obj_ in codex_ndjson_from_cli(req):
            collected.append(obj_)

        assert collected == [obj]


class TestStreamCodexCliEvents:
    @pytest.mark.asyncio
    async def test_yields_objects_then_done(self, monkeypatch, tmp_path):
        events = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

        async def fake_ndjson(request):
            for e in events:
                yield e

        monkeypatch.setattr(codex_module, "CODEX_CLI", "codex")
        monkeypatch.setattr(codex_module, "_ndjson_from_cli", fake_ndjson)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        collected = []
        async for obj in stream_codex_cli_events(req):
            collected.append(obj)

        assert collected[:-1] == events
        assert collected[-1] == {"type": "done"}

    @pytest.mark.asyncio
    async def test_cli_not_available_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_module, "CODEX_CLI", None)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        with pytest.raises(RuntimeError, match="Codex CLI not found"):
            async for _ in stream_codex_cli_events(req):
                pass

    @pytest.mark.asyncio
    async def test_done_is_always_last(self, monkeypatch, tmp_path):
        async def fake_ndjson(request):
            yield {"type": "system"}

        monkeypatch.setattr(codex_module, "CODEX_CLI", "codex")
        monkeypatch.setattr(codex_module, "_ndjson_from_cli", fake_ndjson)

        req = CodexCodeRequest(prompt="test", repo=tmp_path, skip_git_repo_check=True)
        result = []
        async for obj in stream_codex_cli_events(req):
            result.append(obj)

        assert result[-1]["type"] == "done"


# ===========================================================================
# CLAUDE CODE
# ===========================================================================

import lionagi.providers.anthropic.claude_code.models as claude_module
from lionagi.providers.anthropic.claude_code.models import (
    ClaudeCodeRequest,
    ClaudeSession,
)
from lionagi.providers.anthropic.claude_code.models import (
    _extract_summary as claude_extract_summary,
)
from lionagi.providers.anthropic.claude_code.models import (
    _maybe_await as claude_maybe_await,
)
from lionagi.providers.anthropic.claude_code.models import (
    _ndjson_from_cli as claude_ndjson_from_cli,
)
from lionagi.providers.anthropic.claude_code.models import _pp_assistant_text
from lionagi.providers.anthropic.claude_code.models import _pp_final as claude_pp_final
from lionagi.providers.anthropic.claude_code.models import _pp_system, _pp_thinking
from lionagi.providers.anthropic.claude_code.models import (
    _pp_tool_result as claude_pp_tool_result,
)
from lionagi.providers.anthropic.claude_code.models import (
    _pp_tool_use as claude_pp_tool_use,
)
from lionagi.providers.anthropic.claude_code.models import stream_cc_cli_events


class TestClaudeExtractSummary:
    def _sess(self, tool_uses: list[dict], result: str = "") -> ClaudeSession:
        s = ClaudeSession()
        s.tool_uses = tool_uses
        s.result = result
        return s

    def test_empty_tool_uses_returns_no_actions_detected(self):
        out = claude_extract_summary(self._sess([]))
        assert out["key_actions"] == ["No specific actions detected"]
        assert out["total_tool_calls"] == 0

    def test_read_tool(self):
        tu = {"name": "Read", "id": "r1", "input": {"file_path": "/a/b.py"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Read /a/b.py" in out["key_actions"]
        assert "/a/b.py" in out["file_operations"]["reads"]

    def test_read_lowercase(self):
        tu = {"name": "read", "id": "r1", "input": {"file_path": "/x.py"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "/x.py" in out["file_operations"]["reads"]

    def test_write_tool(self):
        tu = {"name": "Write", "id": "w1", "input": {"file_path": "/out.txt"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Wrote /out.txt" in out["key_actions"]
        assert "/out.txt" in out["file_operations"]["writes"]

    def test_edit_tool(self):
        tu = {"name": "Edit", "id": "e1", "input": {"file_path": "/m.py"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Edited /m.py" in out["key_actions"]
        assert "/m.py" in out["file_operations"]["edits"]

    def test_multiedit_tool(self):
        tu = {"name": "MultiEdit", "id": "e2", "input": {"file_path": "/n.py"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "/n.py" in out["file_operations"]["edits"]

    def test_bash_tool(self):
        tu = {"name": "Bash", "id": "b1", "input": {"command": "ls -la"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Ran: ls -la" in out["key_actions"]

    def test_bash_long_command_truncated(self):
        long_cmd = "y" * 100
        tu = {"name": "bash", "id": "b2", "input": {"command": long_cmd}}
        out = claude_extract_summary(self._sess([tu]))
        action = next(a for a in out["key_actions"] if a.startswith("Ran:"))
        assert action.endswith("...")

    def test_glob_tool(self):
        tu = {"name": "Glob", "id": "g1", "input": {"pattern": "**/*.py"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Searched files: **/*.py" in out["key_actions"]

    def test_grep_tool(self):
        tu = {"name": "Grep", "id": "gr1", "input": {"pattern": "def test_"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Searched content: def test_" in out["key_actions"]

    def test_task_agent_tool(self):
        tu = {"name": "Task", "id": "t1", "input": {"description": "run analysis"}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Spawned agent: run analysis" in out["key_actions"]

    def test_mcp_tool(self):
        tu = {"name": "mcp__khive__memory", "id": "m1", "input": {}}
        out = claude_extract_summary(self._sess([tu]))
        assert any("MCP" in a for a in out["key_actions"])

    def test_todo_write_tool(self):
        tu = {
            "name": "TodoWrite",
            "id": "td1",
            "input": {"todos": [{"id": "1"}, {"id": "2"}]},
        }
        out = claude_extract_summary(self._sess([tu]))
        assert "Created 2 todos" in out["key_actions"]

    def test_unknown_tool(self):
        tu = {"name": "exotic_tool", "id": "u1", "input": {}}
        out = claude_extract_summary(self._sess([tu]))
        assert "Used exotic_tool" in out["key_actions"]

    def test_deduplication_of_key_actions(self):
        tu1 = {"name": "Read", "id": "r1", "input": {"file_path": "/same.py"}}
        tu2 = {"name": "Read", "id": "r2", "input": {"file_path": "/same.py"}}
        out = claude_extract_summary(self._sess([tu1, tu2]))
        assert out["key_actions"].count("Read /same.py") == 1

    def test_deduplication_of_file_paths(self):
        tu1 = {"name": "Write", "id": "w1", "input": {"file_path": "/dup.py"}}
        tu2 = {"name": "Write", "id": "w2", "input": {"file_path": "/dup.py"}}
        out = claude_extract_summary(self._sess([tu1, tu2]))
        assert out["file_operations"]["writes"].count("/dup.py") == 1

    def test_result_truncation(self):
        long_result = "z" * 300
        out = claude_extract_summary(self._sess([], result=long_result))
        assert out["result_summary"].endswith("...")

    def test_usage_stats_keys_present(self):
        s = self._sess([])
        s.total_cost_usd = 0.05
        s.num_turns = 2
        s.duration_ms = 1000
        s.duration_api_ms = 800
        out = claude_extract_summary(s)
        assert "total_cost_usd" in out["usage_stats"]
        assert "duration_api_ms" in out["usage_stats"]

    def test_populate_summary_sets_attribute(self):
        s = ClaudeSession()
        s.populate_summary()
        assert isinstance(s.summary, dict)


class TestClaudeMaybeAwait:
    @pytest.mark.asyncio
    async def test_sync_callback(self):
        results = []
        await claude_maybe_await(lambda x: results.append(x), 42)
        assert results == [42]

    @pytest.mark.asyncio
    async def test_async_callback(self):
        results = []

        async def acb(x):
            results.append(x)

        await claude_maybe_await(acb, "val")
        assert results == ["val"]

    @pytest.mark.asyncio
    async def test_none_func(self):
        await claude_maybe_await(None, "ignored")


class TestClaudePpFunctions:
    def test_pp_system_no_raise(self):
        _pp_system(
            {"session_id": "s1", "model": "claude-3", "tools": ["Read", "Write"]},
            theme="light",
        )

    def test_pp_system_many_tools(self):
        tools = [f"tool_{i}" for i in range(12)]
        _pp_system({"session_id": "s1", "model": "m", "tools": tools}, theme="light")

    def test_pp_thinking_no_raise(self):
        _pp_thinking("some deep thought", theme="light")

    def test_pp_assistant_text_no_raise(self):
        _pp_assistant_text("hello from claude", theme="light")

    def test_pp_tool_use_no_raise(self):
        tu = {"name": "Read", "id": "r1", "input": {"file_path": "/a.py"}}
        claude_pp_tool_use(tu, theme="light")

    def test_pp_tool_result_no_raise(self):
        tr = {"tool_use_id": "r1", "content": "file contents", "is_error": False}
        claude_pp_tool_result(tr, theme="light")

    def test_pp_tool_result_error(self):
        tr = {"tool_use_id": "r1", "content": "error msg", "is_error": True}
        claude_pp_tool_result(tr, theme="light")

    def test_pp_final_no_raise(self):
        s = ClaudeSession()
        s.result = "done"
        s.total_cost_usd = 0.001
        s.num_turns = 1
        s.duration_ms = 200
        s.duration_api_ms = 180
        s.usage = {"input_tokens": 50, "output_tokens": 20}
        claude_pp_final(s, theme="light")

    def test_pp_final_no_cost(self):
        s = ClaudeSession()
        s.total_cost_usd = None
        claude_pp_final(s, theme="light")


class TestClaudeNdjsonFromCli:
    @pytest.mark.asyncio
    async def test_valid_ndjson_stream(self, monkeypatch, tmp_path):
        objs = [
            {"type": "system", "session_id": "abc"},
            {"type": "result", "result": "ok"},
        ]
        fake_proc = FakeProcess([_ndjson_bytes(*objs)])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(claude_module, "CLAUDE_CLI", "claude")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path)
        collected = []
        async for obj in claude_ndjson_from_cli(req):
            collected.append(obj)

        assert collected == objs

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises(self, monkeypatch, tmp_path):
        fake_proc = FakeProcess([], returncode=1, stderr_data=b"claude failed")

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(claude_module, "CLAUDE_CLI", "claude")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path)
        with pytest.raises(RuntimeError, match="claude failed"):
            async for _ in claude_ndjson_from_cli(req):
                pass

    @pytest.mark.asyncio
    async def test_workspace_created(self, monkeypatch, tmp_path):
        """_ndjson_from_cli creates workspace directory via request.cwd()."""
        ws_dir = tmp_path / "subdir"
        assert not ws_dir.exists()

        fake_proc = FakeProcess([])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(claude_module, "CLAUDE_CLI", "claude")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path, ws="subdir")
        async for _ in claude_ndjson_from_cli(req):
            pass

        assert ws_dir.exists()

    @pytest.mark.asyncio
    async def test_cli_not_installed_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(claude_module, "CLAUDE_CLI", None)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path)
        # stream_cc_cli_events guards the call; _ndjson_from_cli would use
        # CLAUDE_CLI directly — patch it to None to get the guard in
        # stream_cc_cli_events
        with pytest.raises(RuntimeError, match="Claude CLI binary not found"):
            async for _ in stream_cc_cli_events(req):
                pass


class TestStreamCcCliEvents:
    @pytest.mark.asyncio
    async def test_yields_objects_then_done(self, monkeypatch, tmp_path):
        events = [{"type": "system"}, {"type": "assistant"}]

        async def fake_ndjson(request):
            for e in events:
                yield e

        monkeypatch.setattr(claude_module, "CLAUDE_CLI", "claude")
        monkeypatch.setattr(claude_module, "_ndjson_from_cli", fake_ndjson)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path)
        collected = []
        async for obj in stream_cc_cli_events(req):
            collected.append(obj)

        assert collected[:-1] == events
        assert collected[-1] == {"type": "done"}

    @pytest.mark.asyncio
    async def test_done_always_last(self, monkeypatch, tmp_path):
        async def fake_ndjson(request):
            yield {"type": "result", "result": "hi"}

        monkeypatch.setattr(claude_module, "CLAUDE_CLI", "claude")
        monkeypatch.setattr(claude_module, "_ndjson_from_cli", fake_ndjson)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path)
        result = []
        async for obj in stream_cc_cli_events(req):
            result.append(obj)

        assert result[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_empty_inner_stream_still_emits_done(self, monkeypatch, tmp_path):
        async def fake_ndjson(request):
            return
            yield  # make it an async generator

        monkeypatch.setattr(claude_module, "CLAUDE_CLI", "claude")
        monkeypatch.setattr(claude_module, "_ndjson_from_cli", fake_ndjson)

        req = ClaudeCodeRequest(prompt="test", repo=tmp_path)
        result = []
        async for obj in stream_cc_cli_events(req):
            result.append(obj)

        assert result == [{"type": "done"}]


# ===========================================================================
# GEMINI
# ===========================================================================

import lionagi.providers.google.gemini_code.models as gemini_module
from lionagi.providers.google.gemini_code.models import GeminiCodeRequest, GeminiSession
from lionagi.providers.google.gemini_code.models import (
    _extract_summary as gemini_extract_summary,
)
from lionagi.providers.google.gemini_code.models import (
    _maybe_await as gemini_maybe_await,
)
from lionagi.providers.google.gemini_code.models import (
    _ndjson_from_cli as gemini_ndjson_from_cli,
)
from lionagi.providers.google.gemini_code.models import _pp_final as gemini_pp_final
from lionagi.providers.google.gemini_code.models import _pp_text as gemini_pp_text
from lionagi.providers.google.gemini_code.models import (
    _pp_tool_result as gemini_pp_tool_result,
)
from lionagi.providers.google.gemini_code.models import (
    _pp_tool_use as gemini_pp_tool_use,
)
from lionagi.providers.google.gemini_code.models import stream_gemini_cli_events


class TestGeminiExtractSummary:
    def _sess(self, tool_uses: list[dict], result: str = "") -> GeminiSession:
        s = GeminiSession()
        s.tool_uses = tool_uses
        s.result = result
        return s

    def test_empty_tool_uses_returns_no_actions(self):
        out = gemini_extract_summary(self._sess([]))
        assert out["key_actions"] == ["No specific actions"]
        assert out["total_tool_calls"] == 0

    def test_read_file_tool(self):
        tu = {"name": "read_file", "id": "r1", "input": {"path": "/a.py"}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "Read /a.py" in out["key_actions"]
        assert "/a.py" in out["file_operations"]["reads"]

    def test_read_alias(self):
        tu = {"name": "Read", "id": "r1", "input": {"file_path": "/b.py"}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "/b.py" in out["file_operations"]["reads"]

    def test_write_tool(self):
        tu = {"name": "write_file", "id": "w1", "input": {"path": "/out.txt"}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "Wrote /out.txt" in out["key_actions"]

    def test_write_alias(self):
        tu = {"name": "Write", "id": "w1", "input": {"path": "/f.txt"}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "/f.txt" in out["file_operations"]["writes"]

    def test_edit_tool(self):
        tu = {"name": "edit_file", "id": "e1", "input": {"path": "/m.py"}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "Edited /m.py" in out["key_actions"]

    def test_edit_alias(self):
        tu = {"name": "Edit", "id": "e1", "input": {"path": "/n.py"}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "/n.py" in out["file_operations"]["edits"]

    def test_bash_aliases(self):
        for name in ("run_shell_command", "shell", "Bash"):
            tu = {"name": name, "id": "b1", "input": {"command": "pwd"}}
            out = gemini_extract_summary(self._sess([tu]))
            assert any(a.startswith("Ran:") for a in out["key_actions"])

    def test_bash_long_command_truncated(self):
        tu = {"name": "Bash", "id": "b1", "input": {"command": "z" * 100}}
        out = gemini_extract_summary(self._sess([tu]))
        action = next(a for a in out["key_actions"] if a.startswith("Ran:"))
        assert action.endswith("...")

    def test_mcp_tool_single_underscore(self):
        tu = {"name": "mcp_some_op", "id": "m1", "input": {}}
        out = gemini_extract_summary(self._sess([tu]))
        assert any("MCP" in a for a in out["key_actions"])

    def test_unknown_tool(self):
        tu = {"name": "fancy_tool", "id": "u1", "input": {}}
        out = gemini_extract_summary(self._sess([tu]))
        assert "Used fancy_tool" in out["key_actions"]

    def test_deduplication(self):
        tu1 = {"name": "read_file", "id": "r1", "input": {"path": "/same.py"}}
        tu2 = {"name": "read_file", "id": "r2", "input": {"path": "/same.py"}}
        out = gemini_extract_summary(self._sess([tu1, tu2]))
        assert out["key_actions"].count("Read /same.py") == 1
        assert out["file_operations"]["reads"].count("/same.py") == 1

    def test_result_truncation(self):
        long_result = "q" * 300
        out = gemini_extract_summary(self._sess([], result=long_result))
        assert out["result_summary"].endswith("...")

    def test_populate_summary_sets_attribute(self):
        s = GeminiSession()
        s.populate_summary()
        assert isinstance(s.summary, dict)


class TestGeminiMaybeAwait:
    @pytest.mark.asyncio
    async def test_sync_callback(self):
        results = []
        await gemini_maybe_await(lambda x: results.append(x), "x")
        assert results == ["x"]

    @pytest.mark.asyncio
    async def test_async_callback(self):
        results = []

        async def acb(x):
            results.append(x)

        await gemini_maybe_await(acb, "y")
        assert results == ["y"]

    @pytest.mark.asyncio
    async def test_none_func(self):
        await gemini_maybe_await(None)


class TestGeminiPpFunctions:
    def test_pp_text_no_raise(self):
        gemini_pp_text("hello from gemini", theme="light")

    def test_pp_tool_use_no_raise(self):
        tu = {"name": "read_file", "id": "r1", "input": {"path": "/a.py"}}
        gemini_pp_tool_use(tu, theme="light")

    def test_pp_tool_result_no_raise(self):
        tr = {"content": "result data", "is_error": False}
        gemini_pp_tool_result(tr, theme="light")

    def test_pp_tool_result_error(self):
        tr = {"content": "error!", "is_error": True}
        gemini_pp_tool_result(tr, theme="light")

    def test_pp_final_no_raise(self):
        s = GeminiSession()
        s.result = "done"
        s.num_turns = 2
        s.duration_ms = 300
        s.usage = {"input_tokens": 80, "output_tokens": 40}
        gemini_pp_final(s, theme="light")


class TestGeminiNdjsonFromCli:
    @pytest.mark.asyncio
    async def test_valid_ndjson_stream(self, monkeypatch, tmp_path):
        objs = [
            {"type": "message", "message": {"content": "hi"}},
            {"type": "result", "result": "done"},
        ]
        fake_proc = FakeProcess([_ndjson_bytes(*objs)])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(gemini_module, "GEMINI_CLI", "gemini")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        collected = []
        async for obj in gemini_ndjson_from_cli(req):
            collected.append(obj)

        assert collected == objs

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises(self, monkeypatch, tmp_path):
        fake_proc = FakeProcess([], returncode=1, stderr_data=b"gemini failed")

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(gemini_module, "GEMINI_CLI", "gemini")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        with pytest.raises(RuntimeError, match="gemini failed"):
            async for _ in gemini_ndjson_from_cli(req):
                pass

    @pytest.mark.asyncio
    async def test_cli_not_installed_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gemini_module, "GEMINI_CLI", None)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        with pytest.raises(RuntimeError, match="Gemini CLI not found"):
            async for _ in gemini_ndjson_from_cli(req):
                pass

    @pytest.mark.asyncio
    async def test_invalid_json_tail_skipped(self, monkeypatch, tmp_path, caplog):
        valid = json.dumps({"type": "message", "message": {"content": "hi"}}).encode()
        invalid_tail = b"NOT JSON AT ALL"

        fake_proc = FakeProcess([valid + b"\n", invalid_tail])

        async def fake_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(gemini_module, "GEMINI_CLI", "gemini")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        import logging

        with caplog.at_level(logging.ERROR, logger="gemini-cli"):
            collected = []
            async for obj in gemini_ndjson_from_cli(req):
                collected.append(obj)

        assert len(collected) == 1
        assert collected[0]["type"] == "message"


class TestStreamGeminiCliEvents:
    @pytest.mark.asyncio
    async def test_yields_objects_then_done(self, monkeypatch, tmp_path):
        events = [{"type": "message"}, {"type": "result", "result": "ok"}]

        async def fake_ndjson(request):
            for e in events:
                yield e

        monkeypatch.setattr(gemini_module, "GEMINI_CLI", "gemini")
        monkeypatch.setattr(gemini_module, "_ndjson_from_cli", fake_ndjson)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        collected = []
        async for obj in stream_gemini_cli_events(req):
            collected.append(obj)

        assert collected[:-1] == events
        assert collected[-1] == {"type": "done"}

    @pytest.mark.asyncio
    async def test_cli_not_available_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gemini_module, "GEMINI_CLI", None)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        with pytest.raises(RuntimeError, match="Gemini CLI not found"):
            async for _ in stream_gemini_cli_events(req):
                pass

    @pytest.mark.asyncio
    async def test_empty_inner_stream_still_emits_done(self, monkeypatch, tmp_path):
        async def fake_ndjson(request):
            return
            yield  # make it an async generator

        monkeypatch.setattr(gemini_module, "GEMINI_CLI", "gemini")
        monkeypatch.setattr(gemini_module, "_ndjson_from_cli", fake_ndjson)

        req = GeminiCodeRequest(prompt="test", repo=tmp_path)
        result = []
        async for obj in stream_gemini_cli_events(req):
            result.append(obj)

        assert result == [{"type": "done"}]
