# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Sandbox policies (PathGuard, ProcessGuard) and git-worktree isolation (SandboxSession) for agent tools."""

from __future__ import annotations

import contextlib
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from lionagi.ln.concurrency import run_sync

__all__ = (
    "PathGuard",
    "ProcessGuard",
    "SandboxSession",
    "create_sandbox",
    "sandbox_commit",
    "sandbox_diff",
    "sandbox_discard",
    "sandbox_merge",
)

_DEFAULT_DENY_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".netrc",
        ".htpasswd",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
    }
)

_DEFAULT_DENY_GLOBS: tuple[str, ...] = (
    ".ssh/*",
    ".aws/*",
    ".config/**/credentials*",
    ".git/config",
    ".git-credentials",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "secrets.*",
)

_SAFE_ENV: dict[str, str] = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HOME": "/tmp",
}

_PRIVILEGED_COMMANDS: frozenset[str] = frozenset(
    {
        "bash",
        "sh",
        "zsh",
        "python",
        "python3",
        "node",
        "perl",
        "ruby",
        "curl",
        "wget",
        "ssh",
        "scp",
        "sudo",
        "su",
        "apt",
        "brew",
        "pip",
        "npm",
    }
)


@dataclass(frozen=True, slots=True)
class PathGuard:
    root: Path
    deny_names: frozenset[str] = _DEFAULT_DENY_NAMES
    deny_globs: tuple[str, ...] = _DEFAULT_DENY_GLOBS
    allow_symlinks: bool = False
    max_read_bytes: int = 5_000_000
    max_write_bytes: int = 5_000_000
    max_image_bytes: int = 10_000_000

    def resolve(self, path: str) -> Path:
        raw = Path(path).expanduser()
        root_resolved = self.root.resolve()
        candidate = raw if raw.is_absolute() else root_resolved / raw
        if not self.allow_symlinks and candidate.is_symlink():
            raise PermissionError(f"Refusing symlink: {path!r}")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as e:
            raise PermissionError(f"Path escapes workspace: {path!r}") from e
        if resolved.name in self.deny_names:
            raise PermissionError(f"Access denied: {resolved.name!r}")
        for glob in self.deny_globs:
            if resolved.match(glob):
                raise PermissionError(f"Access denied by pattern: {glob!r}")
        return resolved

    def open_read(self, path: str) -> tuple[int, os.stat_result]:
        p = self.resolve(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if not self.allow_symlinks and hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(p, flags)
        st = os.fstat(fd)
        if st.st_size > self.max_read_bytes:
            os.close(fd)
            raise PermissionError(
                f"File too large: {st.st_size} bytes > {self.max_read_bytes}"
            )
        return fd, st

    def atomic_write(self, path: Path, text: str, mode: int = 0o644) -> None:
        size = len(text.encode("utf-8"))
        if size > self.max_write_bytes:
            raise PermissionError(
                f"Content too large: {size} bytes > {self.max_write_bytes}"
            )
        tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()

    def validate_dir(self, path: str) -> Path:
        p = self.resolve(path)
        if not p.is_dir():
            raise PermissionError(f"Not a directory: {path!r}")
        return p


@dataclass(frozen=True, slots=True)
class ProcessGuard:
    workspace_root: Path
    allowed_commands: frozenset[str] | None = None
    denied_commands: frozenset[str] = field(
        default=frozenset(
            {
                "rm",
                "rmdir",
                "mkfs",
                "dd",
                "shutdown",
                "reboot",
                "kill",
                "killall",
            }
        )
    )
    privileged_commands: frozenset[str] = _PRIVILEGED_COMMANDS
    extra_env: dict[str, str] = field(default_factory=dict)
    max_output_bytes: int = 100_000
    default_timeout_s: float = 30.0
    max_timeout_s: float = 300.0

    def safe_env(self) -> dict[str, str]:
        env = dict(_SAFE_ENV)
        env.update(self.extra_env)
        return env

    def validate_cwd(self, cwd: str | None) -> str:
        root_resolved = self.workspace_root.resolve()
        if cwd is None:
            return str(root_resolved)
        resolved = Path(cwd).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as e:
            raise PermissionError(f"cwd escapes workspace: {cwd!r}") from e
        if not resolved.is_dir():
            raise PermissionError(f"cwd is not a directory: {cwd!r}")
        return str(resolved)

    def check_command(self, argv: list[str]) -> None:
        if not argv:
            raise PermissionError("Empty command")
        cmd = Path(argv[0]).name
        if cmd in self.denied_commands:
            raise PermissionError(f"Command denied: {cmd!r}")
        if self.allowed_commands is not None and cmd not in self.allowed_commands:
            raise PermissionError(f"Command not in allowlist: {cmd!r}")

    def clamp_timeout(self, timeout_ms: int | None) -> float:
        if timeout_ms is None:
            return self.default_timeout_s
        t = max(1, min(timeout_ms, int(self.max_timeout_s * 1000)))
        return t / 1000.0


@dataclass
class SandboxSession:
    worktree_path: str
    branch_name: str
    base_branch: str
    repo_root: str
    is_active: bool = True


def _run_git(args: list[str], cwd: str | None = None) -> tuple[str, str, int]:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def _create_worktree_sync(
    repo_root: str, branch_name: str, base_branch: str
) -> SandboxSession:
    root = Path(repo_root)
    worktree_dir = root / ".worktrees" / branch_name
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    _, err, rc = _run_git(
        ["worktree", "add", "-b", branch_name, str(worktree_dir), base_branch],
        cwd=repo_root,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to create worktree: {err}")

    return SandboxSession(
        worktree_path=str(worktree_dir),
        branch_name=branch_name,
        base_branch=base_branch,
        repo_root=repo_root,
    )


def _get_diff_sync(session: SandboxSession) -> dict:
    wt = session.worktree_path

    _run_git(["add", "-A"], cwd=wt)

    diff_stat, _, _ = _run_git(["diff", "--cached", "--stat"], cwd=wt)
    diff_patch, _, _ = _run_git(["diff", "--cached"], cwd=wt)

    changed, _, _ = _run_git(["diff", "--cached", "--name-only"], cwd=wt)
    files = [f for f in changed.split("\n") if f] if changed else []

    return {
        "files_changed": files,
        "stat": diff_stat,
        "patch": diff_patch[:10000] if len(diff_patch) > 10000 else diff_patch,
        "patch_truncated": len(diff_patch) > 10000,
        "full_patch_chars": len(diff_patch),
    }


def _commit_sync(session: SandboxSession, message: str) -> dict:
    wt = session.worktree_path
    _run_git(["add", "-A"], cwd=wt)

    stdout, stderr, rc = _run_git(["commit", "-m", message], cwd=wt)
    if rc != 0:
        if "nothing to commit" in stdout + stderr:
            return {"success": True, "message": "Nothing to commit"}
        return {"success": False, "error": stderr}

    sha, _, _ = _run_git(["rev-parse", "HEAD"], cwd=wt)
    return {"success": True, "commit": sha, "message": message}


def _cleanup_worktree_sync(session: SandboxSession) -> dict:
    _, err1, rc1 = _run_git(
        ["worktree", "remove", session.worktree_path, "--force"],
        cwd=session.repo_root,
    )
    _, err2, rc2 = _run_git(
        ["branch", "-D", session.branch_name],
        cwd=session.repo_root,
    )
    return {
        "worktree_removed": rc1 == 0,
        "branch_deleted": rc2 == 0,
        "errors": [e for e in [err1, err2] if e and "error" in e.lower()],
    }


def _merge_sync(session: SandboxSession) -> dict:
    _run_git(["add", "-A"], cwd=session.worktree_path)
    _run_git(
        ["commit", "-m", f"sandbox: {session.branch_name}"], cwd=session.worktree_path
    )

    stdout, stderr, rc = _run_git(
        [
            "merge",
            "--no-ff",
            session.branch_name,
            "-m",
            f"Merge sandbox {session.branch_name}",
        ],
        cwd=session.repo_root,
    )
    if rc != 0:
        return {"success": False, "error": stderr}

    cleanup = _cleanup_worktree_sync(session)
    return {"success": True, "merged": True, **cleanup}


async def create_sandbox(
    repo_root: str,
    base_branch: str | None = None,
    name: str | None = None,
) -> SandboxSession:
    """Create an isolated sandbox (git worktree) for safe code changes."""
    if base_branch is None:
        stdout, _, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        base_branch = stdout or "main"

    branch_name = name or f"sandbox-{uuid.uuid4().hex[:8]}"
    return await run_sync(_create_worktree_sync, repo_root, branch_name, base_branch)


async def sandbox_diff(session: SandboxSession) -> dict:
    """Get diff of changes made in the sandbox."""
    return await run_sync(_get_diff_sync, session)


async def sandbox_commit(session: SandboxSession, message: str) -> dict:
    """Commit changes in the sandbox."""
    return await run_sync(_commit_sync, session, message)


async def sandbox_merge(session: SandboxSession) -> dict:
    """Merge sandbox changes back and clean up."""
    return await run_sync(_merge_sync, session)


async def sandbox_discard(session: SandboxSession) -> dict:
    """Discard sandbox and all changes."""
    return await run_sync(_cleanup_worktree_sync, session)
