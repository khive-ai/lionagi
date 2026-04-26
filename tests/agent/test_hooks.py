# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for built-in coding-agent hooks."""

import inspect

import pytest

from lionagi.agent.hooks import auto_format_python, guard_paths


async def test_guard_paths_returns_callable_hook(tmp_path):
    allowed = tmp_path / "project"
    allowed.mkdir()

    hook = guard_paths(allowed_paths=[str(allowed)])

    assert callable(hook)
    assert not inspect.iscoroutine(hook)
    assert await hook("reader", "read", {"path": str(allowed / "ok.py")}) is None


async def test_guard_paths_blocks_prefix_sibling_escape(tmp_path):
    allowed = tmp_path / "project"
    sibling = tmp_path / "project-evil"
    allowed.mkdir()
    sibling.mkdir()
    hook = guard_paths(allowed_paths=[str(allowed)])

    with pytest.raises(PermissionError, match="allowed list"):
        await hook("reader", "read", {"path": str(sibling / "secret.py")})


async def test_auto_format_python_uses_argv_without_shell(monkeypatch):
    calls = []

    async def fake_run_sync(fn, cmd, shell, timeout, cwd):
        calls.append((cmd, shell, timeout, cwd))
        return {"returncode": 0}

    monkeypatch.setattr("lionagi.ln.concurrency.run_sync", fake_run_sync)

    result = await auto_format_python(
        "editor",
        "write",
        {"file_path": "src/weird;name.py"},
        {"success": True},
    )

    assert result is None
    assert calls == [(["ruff", "format", "src/weird;name.py"], False, 10.0, None)]
