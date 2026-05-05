# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage-targeted tests for tools/code/bash.py uncovered paths."""

from __future__ import annotations

import io
import os
import signal

import pytest

from lionagi.tools.code.bash import (
    _MAX_OUTPUT_BYTES,
    BashTool,
    _decode_output,
    _drain,
    _kill_pgroup,
)

# ---------------------------------------------------------------------------
# _drain — truncation logic
# ---------------------------------------------------------------------------


class _FakeStream:
    """Fake stream that yields chunks then EOF."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_drain_small_data_not_truncated():
    buf = bytearray()
    stream = _FakeStream([b"hello world", b""])
    truncated = _drain(stream, buf)
    assert bytes(buf) == b"hello world"
    assert truncated is False


def test_drain_data_at_exact_cap_truncated():
    # Fill buf to just at cap, then next read would overflow
    buf = bytearray(b"x" * (_MAX_OUTPUT_BYTES - 5))
    stream = _FakeStream([b"y" * 10, b""])  # 10 bytes, only 5 fit
    truncated = _drain(stream, buf)
    assert len(buf) == _MAX_OUTPUT_BYTES
    assert truncated is True


def test_drain_empty_stream_not_truncated():
    buf = bytearray()
    stream = _FakeStream([b""])
    truncated = _drain(stream, buf)
    assert buf == bytearray()
    # empty stream breaks out of loop with no explicit return → None (not truncated)
    assert not truncated


def test_drain_exception_on_read_breaks_loop():
    class ErrorStream:
        def read(self, size):
            raise OSError("broken pipe")

    buf = bytearray()
    truncated = _drain(ErrorStream(), buf)
    # exception breaks out of loop with no explicit return → None (not truncated)
    assert not truncated


# ---------------------------------------------------------------------------
# _decode_output — truncation marker
# ---------------------------------------------------------------------------


def test_decode_output_no_truncation():
    buf = bytearray(b"hello")
    result = _decode_output(buf, truncated=False)
    assert result == "hello"
    assert "truncated" not in result


def test_decode_output_with_truncation_appends_marker():
    buf = bytearray(b"partial output")
    result = _decode_output(buf, truncated=True)
    assert result.startswith("partial output")
    assert "truncated" in result
    assert str(_MAX_OUTPUT_BYTES) in result


def test_decode_output_invalid_utf8_replaced():
    buf = bytearray(b"\xff\xfe invalid")
    result = _decode_output(buf, truncated=False)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _kill_pgroup — process group kill logic
# ---------------------------------------------------------------------------


def test_kill_pgroup_uses_os_killpg(monkeypatch):
    killed = []

    def fake_killpg(pgid, sig):
        killed.append((pgid, sig))

    monkeypatch.setattr(os, "killpg", fake_killpg)

    class FakeProc:
        pid = 12345

        def terminate(self):
            killed.append(("terminate", self.pid))

        def kill(self):
            killed.append(("kill", self.pid))

    _kill_pgroup(FakeProc())
    # Should have tried SIGTERM and SIGKILL
    assert any(s == signal.SIGTERM for _, s in killed if isinstance(s, signal.Signals))
    assert any(s == signal.SIGKILL for _, s in killed if isinstance(s, signal.Signals))


def test_kill_pgroup_falls_back_to_terminate_when_no_killpg(monkeypatch):
    terminated = []

    def raise_attr_error(pgid, sig):
        raise AttributeError("no killpg")

    monkeypatch.setattr(os, "killpg", raise_attr_error)

    class FakeProc:
        pid = 99

        def terminate(self):
            terminated.append("terminate")

        def kill(self):
            terminated.append("kill")

    _kill_pgroup(FakeProc())
    assert "terminate" in terminated or "kill" in terminated


# ---------------------------------------------------------------------------
# BashTool.to_tool — callable rename
# ---------------------------------------------------------------------------


def test_to_tool_default_name_not_renamed():
    tool = BashTool()
    t = tool.to_tool()
    assert t is not None
    # Default name is "bash_tool", rename branch NOT taken
    assert t.func_callable.__name__ == "bash_tool"


def test_to_tool_custom_system_name_renames_callable():
    class CustomBash(BashTool):
        system_tool_name = "my_bash"

    tool = CustomBash()
    t = tool.to_tool()
    assert t.func_callable.__name__ == "my_bash"


def test_to_tool_caches_tool():
    tool = BashTool()
    t1 = tool.to_tool()
    t2 = tool.to_tool()
    assert t1 is t2
