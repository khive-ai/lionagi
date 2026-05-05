# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for PathGuard, ProcessGuard, _commit_sync and _merge_sync in tools/sandbox.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lionagi.tools.sandbox as sandbox_module
from lionagi.tools.sandbox import (
    PathGuard,
    ProcessGuard,
    SandboxSession,
    _commit_sync,
    _merge_sync,
)

# ---------------------------------------------------------------------------
# PathGuard.resolve
# ---------------------------------------------------------------------------


class TestPathGuardResolve:
    def test_resolve_valid_path_inside_root(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        f = tmp_path / "subdir" / "file.txt"
        f.parent.mkdir()
        f.write_text("content")
        resolved = guard.resolve(str(f))
        assert resolved == f.resolve()

    def test_resolve_relative_path(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        f = tmp_path / "hello.py"
        f.write_text("x")
        resolved = guard.resolve("hello.py")
        assert resolved == f.resolve()

    def test_resolve_escape_raises_permission_error(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        outside = tmp_path.parent / "escape.txt"
        with pytest.raises(PermissionError, match="escapes workspace"):
            guard.resolve(str(outside))

    def test_resolve_denied_name_raises(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=1")
        with pytest.raises(PermissionError, match="Access denied"):
            guard.resolve(".env")

    def test_resolve_denied_glob_pem_raises(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        pem = tmp_path / "cert.pem"
        pem.write_text("cert")
        with pytest.raises(PermissionError, match="Access denied"):
            guard.resolve("cert.pem")

    def test_resolve_symlink_raises_when_not_allowed(self, tmp_path):
        guard = PathGuard(root=tmp_path, allow_symlinks=False)
        real = tmp_path / "real.txt"
        real.write_text("real")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        with pytest.raises(PermissionError, match="Refusing symlink"):
            guard.resolve("link.txt")


# ---------------------------------------------------------------------------
# PathGuard.open_read
# ---------------------------------------------------------------------------


class TestPathGuardOpenRead:
    def test_open_read_valid_file(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        f = tmp_path / "readable.txt"
        f.write_text("hello world")
        fd, st = guard.open_read(str(f))
        try:
            assert fd >= 0
            assert st.st_size > 0
        finally:
            os.close(fd)

    def test_open_read_file_too_large_raises(self, tmp_path):
        guard = PathGuard(root=tmp_path, max_read_bytes=5)
        f = tmp_path / "large.txt"
        f.write_text("hello world this is more than 5 bytes")
        with pytest.raises(PermissionError, match="too large"):
            guard.open_read(str(f))


# ---------------------------------------------------------------------------
# PathGuard.atomic_write
# ---------------------------------------------------------------------------


class TestPathGuardAtomicWrite:
    def test_atomic_write_creates_file(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        target = tmp_path / "output.txt"
        guard.atomic_write(target, "written content")
        assert target.read_text() == "written content"

    def test_atomic_write_overwrites_existing_file(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        target = tmp_path / "existing.txt"
        target.write_text("old content")
        guard.atomic_write(target, "new content")
        assert target.read_text() == "new content"

    def test_atomic_write_too_large_raises(self, tmp_path):
        guard = PathGuard(root=tmp_path, max_write_bytes=10)
        target = tmp_path / "big.txt"
        with pytest.raises(PermissionError, match="too large"):
            guard.atomic_write(target, "x" * 100)


# ---------------------------------------------------------------------------
# PathGuard.validate_dir
# ---------------------------------------------------------------------------


class TestPathGuardValidateDir:
    def test_validate_dir_valid_directory(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        result = guard.validate_dir(str(subdir))
        assert result == subdir.resolve()

    def test_validate_dir_file_raises(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(PermissionError, match="Not a directory"):
            guard.validate_dir(str(f))

    def test_validate_dir_nonexistent_raises(self, tmp_path):
        guard = PathGuard(root=tmp_path)
        with pytest.raises(PermissionError, match="Not a directory"):
            guard.validate_dir("nonexistent_dir")


# ---------------------------------------------------------------------------
# ProcessGuard.safe_env
# ---------------------------------------------------------------------------


class TestProcessGuardSafeEnv:
    def test_safe_env_contains_required_keys(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        env = guard.safe_env()
        assert "PATH" in env
        assert "LANG" in env
        assert "HOME" in env

    def test_safe_env_includes_extra_env(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path, extra_env={"MY_VAR": "value"})
        env = guard.safe_env()
        assert env["MY_VAR"] == "value"

    def test_safe_env_does_not_include_system_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "topsecret")
        guard = ProcessGuard(workspace_root=tmp_path)
        env = guard.safe_env()
        assert "SECRET_KEY" not in env


# ---------------------------------------------------------------------------
# ProcessGuard.validate_cwd
# ---------------------------------------------------------------------------


class TestProcessGuardValidateCwd:
    def test_validate_cwd_none_returns_root(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        result = guard.validate_cwd(None)
        assert result == str(tmp_path.resolve())

    def test_validate_cwd_valid_subdirectory(self, tmp_path):
        subdir = tmp_path / "workdir"
        subdir.mkdir()
        guard = ProcessGuard(workspace_root=tmp_path)
        result = guard.validate_cwd(str(subdir))
        assert result == str(subdir.resolve())

    def test_validate_cwd_escapes_workspace_raises(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        with pytest.raises(PermissionError, match="escapes workspace"):
            guard.validate_cwd(str(tmp_path.parent))

    def test_validate_cwd_not_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        guard = ProcessGuard(workspace_root=tmp_path)
        with pytest.raises(PermissionError, match="not a directory"):
            guard.validate_cwd(str(f))


# ---------------------------------------------------------------------------
# ProcessGuard.check_command
# ---------------------------------------------------------------------------


class TestProcessGuardCheckCommand:
    def test_check_command_empty_raises(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        with pytest.raises(PermissionError, match="Empty command"):
            guard.check_command([])

    def test_check_command_denied_raises(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        with pytest.raises(PermissionError, match="Command denied"):
            guard.check_command(["rm", "-rf", "/"])

    def test_check_command_not_in_allowlist_raises(self, tmp_path):
        guard = ProcessGuard(
            workspace_root=tmp_path,
            allowed_commands=frozenset({"ls", "echo"}),
        )
        with pytest.raises(PermissionError, match="allowlist"):
            guard.check_command(["git", "https://example.com"])

    def test_check_command_allowed_passes(self, tmp_path):
        guard = ProcessGuard(
            workspace_root=tmp_path,
            allowed_commands=frozenset({"ls", "echo"}),
        )
        guard.check_command(["ls", "-la"])  # should not raise

    def test_check_command_no_allowlist_non_denied_passes(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        guard.check_command(["git", "status"])  # not in denied list


# ---------------------------------------------------------------------------
# ProcessGuard.clamp_timeout
# ---------------------------------------------------------------------------


class TestProcessGuardClampTimeout:
    def test_clamp_timeout_none_returns_default(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        result = guard.clamp_timeout(None)
        assert result == 30.0

    def test_clamp_timeout_normal_value(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        result = guard.clamp_timeout(5000)
        assert result == 5.0

    def test_clamp_timeout_exceeds_max(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        result = guard.clamp_timeout(10_000_000)
        assert result == guard.max_timeout_s

    def test_clamp_timeout_below_one_ms_clamped_to_one(self, tmp_path):
        guard = ProcessGuard(workspace_root=tmp_path)
        result = guard.clamp_timeout(0)
        assert result == 1 / 1000.0


# ---------------------------------------------------------------------------
# _commit_sync — "nothing to commit" path (line 283)
# ---------------------------------------------------------------------------


class TestCommitSync:
    def test_commit_sync_nothing_to_commit(self, monkeypatch, tmp_path):
        calls = []

        def fake_run_git(args, cwd=None):
            calls.append(args)
            if args[0] == "commit":
                return ("nothing to commit, working tree clean", "", 1)
            return ("", "", 0)

        monkeypatch.setattr(sandbox_module, "_run_git", fake_run_git)
        session = SandboxSession(
            worktree_path=str(tmp_path),
            branch_name="test-branch",
            base_branch="main",
            repo_root=str(tmp_path),
        )
        result = _commit_sync(session, "test message")
        assert result == {"success": True, "message": "Nothing to commit"}

    def test_commit_sync_other_failure(self, monkeypatch, tmp_path):
        def fake_run_git(args, cwd=None):
            if args[0] == "commit":
                return ("", "some other error", 1)
            return ("", "", 0)

        monkeypatch.setattr(sandbox_module, "_run_git", fake_run_git)
        session = SandboxSession(
            worktree_path=str(tmp_path),
            branch_name="test-branch",
            base_branch="main",
            repo_root=str(tmp_path),
        )
        result = _commit_sync(session, "test message")
        assert result["success"] is False
        assert "some other error" in result["error"]

    def test_commit_sync_success(self, monkeypatch, tmp_path):
        def fake_run_git(args, cwd=None):
            if args[0] == "commit":
                return ("1 file changed", "", 0)
            if args[0] == "rev-parse":
                return ("abc123", "", 0)
            return ("", "", 0)

        monkeypatch.setattr(sandbox_module, "_run_git", fake_run_git)
        session = SandboxSession(
            worktree_path=str(tmp_path),
            branch_name="test-branch",
            base_branch="main",
            repo_root=str(tmp_path),
        )
        result = _commit_sync(session, "test message")
        assert result["success"] is True
        assert result["commit"] == "abc123"


# ---------------------------------------------------------------------------
# _merge_sync — error path (line 322)
# ---------------------------------------------------------------------------


class TestMergeSync:
    def test_merge_sync_failure_returns_error(self, monkeypatch, tmp_path):
        def fake_run_git(args, cwd=None):
            if args[0] == "merge":
                return ("", "CONFLICT: merge failed", 1)
            return ("", "", 0)

        monkeypatch.setattr(sandbox_module, "_run_git", fake_run_git)
        session = SandboxSession(
            worktree_path=str(tmp_path),
            branch_name="test-branch",
            base_branch="main",
            repo_root=str(tmp_path),
        )
        result = _merge_sync(session)
        assert result["success"] is False
        assert "CONFLICT" in result["error"]

    def test_merge_sync_success(self, monkeypatch, tmp_path):
        def fake_run_git(args, cwd=None):
            return ("success", "", 0)

        monkeypatch.setattr(sandbox_module, "_run_git", fake_run_git)
        session = SandboxSession(
            worktree_path=str(tmp_path),
            branch_name="test-branch",
            base_branch="main",
            repo_root=str(tmp_path),
        )
        result = _merge_sync(session)
        assert result["success"] is True
        assert result["merged"] is True
