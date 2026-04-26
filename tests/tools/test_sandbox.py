# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for sandbox.py: create, diff, commit, merge, discard."""

import os
import subprocess
from pathlib import Path

import pytest

from lionagi.tools.sandbox import (
    SandboxSession,
    create_sandbox,
    sandbox_commit,
    sandbox_diff,
    sandbox_discard,
    sandbox_merge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    cmds = [
        ["git", "init"],
        ["git", "config", "user.email", "test@test.com"],
        ["git", "config", "user.name", "Test"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd=str(path), capture_output=True, check=True)
    (path / "README.md").write_text("initial\n")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(path), capture_output=True, check=True)


@pytest.fixture
def git_repo(tmp_path):
    _init_git_repo(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# create_sandbox
# ---------------------------------------------------------------------------


async def test_create_sandbox_creates_worktree(git_repo):
    session = await create_sandbox(str(git_repo))
    assert isinstance(session, SandboxSession)
    assert session.is_active is True
    assert os.path.isdir(session.worktree_path)
    assert session.branch_name.startswith("sandbox-")
    await sandbox_discard(session)


async def test_create_sandbox_custom_name(git_repo):
    session = await create_sandbox(str(git_repo), name="my-experiment")
    assert session.branch_name == "my-experiment"
    assert os.path.isdir(session.worktree_path)
    await sandbox_discard(session)


# ---------------------------------------------------------------------------
# sandbox_diff
# ---------------------------------------------------------------------------


async def test_sandbox_diff_empty_for_no_changes(git_repo):
    session = await create_sandbox(str(git_repo))
    diff = await sandbox_diff(session)
    assert diff["files_changed"] == [] and diff["patch"] == ""
    await sandbox_discard(session)


async def test_sandbox_diff_shows_new_file(git_repo):
    session = await create_sandbox(str(git_repo))
    (Path(session.worktree_path) / "new.txt").write_text("hello sandbox\n")
    diff = await sandbox_diff(session)
    assert "new.txt" in diff["files_changed"]
    assert "hello sandbox" in diff["patch"]
    await sandbox_discard(session)


# ---------------------------------------------------------------------------
# sandbox_commit
# ---------------------------------------------------------------------------


async def test_sandbox_commit_records_change(git_repo):
    session = await create_sandbox(str(git_repo))
    (Path(session.worktree_path) / "work.py").write_text("x = 1\n")
    result = await sandbox_commit(session, "add work.py")
    assert result["success"] is True
    assert "commit" in result
    assert result["message"] == "add work.py"
    await sandbox_discard(session)


async def test_sandbox_commit_nothing_to_commit(git_repo):
    session = await create_sandbox(str(git_repo))
    result = await sandbox_commit(session, "empty commit")
    assert result["success"] is True
    assert "Nothing to commit" in result.get("message", "")
    await sandbox_discard(session)


# ---------------------------------------------------------------------------
# sandbox_merge
# ---------------------------------------------------------------------------


async def test_sandbox_merge_applies_changes(git_repo):
    session = await create_sandbox(str(git_repo))
    (Path(session.worktree_path) / "merged.txt").write_text("from sandbox\n")
    result = await sandbox_merge(session)
    assert result["success"] is True and result["merged"] is True
    assert (git_repo / "merged.txt").read_text() == "from sandbox\n"


async def test_sandbox_merge_cleans_up_worktree(git_repo):
    session = await create_sandbox(str(git_repo))
    worktree_path = session.worktree_path
    await sandbox_merge(session)
    assert not os.path.exists(worktree_path)


# ---------------------------------------------------------------------------
# sandbox_discard
# ---------------------------------------------------------------------------


async def test_sandbox_discard_removes_worktree(git_repo):
    session = await create_sandbox(str(git_repo))
    worktree_path = session.worktree_path
    result = await sandbox_discard(session)
    assert result["worktree_removed"] is True
    assert not os.path.exists(worktree_path)


async def test_sandbox_discard_deletes_branch(git_repo):
    session = await create_sandbox(str(git_repo))
    branch_name = session.branch_name
    await sandbox_discard(session)
    out = subprocess.run(["git", "branch"], cwd=str(git_repo), capture_output=True, text=True)
    assert branch_name not in out.stdout


async def test_sandbox_discard_changes_not_in_base(git_repo):
    session = await create_sandbox(str(git_repo))
    (Path(session.worktree_path) / "ephemeral.txt").write_text("gone\n")
    await sandbox_discard(session)
    assert not (git_repo / "ephemeral.txt").exists()
