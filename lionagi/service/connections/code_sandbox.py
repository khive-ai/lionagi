# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental code sandbox for executing user code in a controlled environment."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import resource as posix_res  # Works on Linux/macOS; noop on Windows
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, Field, model_validator

from .endpoint import Endpoint
from .endpoint_config import EndpointConfig

# --------------------------------------------------------------------------- logging
log = logging.getLogger("lionagi.sandbox")
log.setLevel(logging.INFO)


# --------------------------------------------------------------------------- request / response models
class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="Source code to execute")
    language: str = Field("python", description="Runner key in RUNNER_TABLE")
    timeout: int = Field(
        30, ge=1, le=600, description="Wall‑clock timeout (s)"
    )
    cpu_time: int = Field(10, ge=1, le=600, description="CPU time limit (s)")
    memory_mb: int = Field(
        512, ge=64, le=8192, description="Max resident set (MB)"
    )
    persist_files: bool = Field(
        True, description="If true git‑commit modified files"
    )
    working_directory: str | None = Field(
        None, description="Sub‑folder inside the session sandbox"
    )

    @model_validator(mode="after")
    def _strip_code(cls, v):  # noqa: N805
        v.code = textwrap.dedent(v.code).lstrip()
        return v


class CodeExecutionResponse(BaseModel):
    stdout: str
    stderr: str
    return_code: int
    wall_time: float
    cpu_time: float
    max_rss_mb: float
    git_commit: str | None = None
    git_diff: str | None = None
    files_created: list[str] = []
    files_modified: list[str] = []
    ran_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- git wrapper
class GitRepository:
    def __init__(self, root: Path):
        self.root = root
        if not (root / ".git").exists():
            self._init_repo()

    # -------- public helpers
    def snapshot_tree(self) -> str:
        """Return the SHA of the current tree head (empty tree == 4b825dc642...)."""
        try:
            res = subprocess.run(
                ["git", "-C", str(self.root), "write-tree"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def add_and_commit(self, message: str) -> tuple[str, str]:
        """Stage all, commit if there are changes, return (commit_sha, diff)."""
        subprocess.run(["git", "-C", self.root, "add", "-A"], check=True)
        pre_tree = self.snapshot_tree()

        # Only commit when tree changed
        if self._is_index_clean(pre_tree):
            return "", ""

        subprocess.run(
            ["git", "-C", self.root, "commit", "-qm", message],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", self.root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", self.root, "diff", f"{commit}^!", "--minimal"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return commit, diff

    # -------- private helpers
    def _init_repo(self) -> None:
        subprocess.run(["git", "-C", self.root, "init", "-q"], check=True)
        for k, v in {
            "user.name": "lionagi-sandbox",
            "user.email": "sandbox@lionagi.ai",
        }.items():
            subprocess.run(
                ["git", "-C", self.root, "config", k, v], check=True
            )

    def _is_index_clean(self, pre_tree: str) -> bool:
        post_tree = self.snapshot_tree()
        return pre_tree == post_tree


# --------------------------------------------------------------------------- sandbox
class CodeSandbox:
    """
    Responsible for:
      • materialising files
      • executing the code in a constrained subprocess or a Docker container
      • returning structured result + git info
    """

    # ------------------------------------------------ runner table
    RUNNER_TABLE: dict[str, list[str]] = {
        "python": [sys.executable, "{file}"],  # default venv
        "bash": ["bash", "{file}"],
        "node": ["node", "{file}"],
    }

    def __init__(self, root: Path):
        self.root = root
        self.repo = GitRepository(root)
        self._lock = asyncio.Lock()

    # ------------------------------------------------ public
    async def run(self, req: CodeExecutionRequest) -> CodeExecutionResponse:
        async with self._lock:  # serialise concurrent calls
            abs_cwd, script_path = await self._prepare_fs(req)

            wall_start = time.perf_counter()
            stdout, stderr, rc, usage = await self._spawn_process(
                req, script_path, abs_cwd
            )
            wall = time.perf_counter() - wall_start

            created, modified = self._file_changes_since(script_path)
            commit, diff = ("", "")
            if rc == 0 and req.persist_files:
                commit, diff = self.repo.add_and_commit(
                    f"Run {script_path.name}"
                )

            return CodeExecutionResponse(
                stdout=stdout,
                stderr=stderr,
                return_code=rc,
                wall_time=wall,
                cpu_time=usage.ru_utime + usage.ru_stime,
                max_rss_mb=usage.ru_maxrss / 1024,
                git_commit=commit or None,
                git_diff=diff or None,
                files_created=created,
                files_modified=modified,
            )

    async def stream(
        self, req: CodeExecutionRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Same contract as Endpoint.stream: yields **dict** chunks (not pydantic models
        to avoid heavy conversions). The final chunk always contains `"done": true`.
        """
        async with self._lock:
            abs_cwd, script_path = await self._prepare_fs(req)
            runner = self._build_cmd(req.language, script_path)
            env = self._sandbox_env()

            proc = await asyncio.create_subprocess_exec(
                *runner,
                cwd=abs_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=lambda: self._apply_limits(req),
            )

            async def _read_stdout():
                async for line in _readlines(proc.stdout):
                    yield {"stream": "stdout", "data": line.decode()}

            async def _read_stderr():
                async for line in _readlines(proc.stderr):
                    yield {"stream": "stderr", "data": line.decode()}

            # Create async generators
            stdout_gen = _read_stdout()
            stderr_gen = _read_stderr()

            # Read from both streams until process completes
            try:
                while True:
                    # Try to read from stdout
                    try:
                        chunk = await asyncio.wait_for(
                            stdout_gen.__anext__(), timeout=0.01
                        )
                        yield chunk
                    except (StopAsyncIteration, asyncio.TimeoutError):
                        pass

                    # Try to read from stderr
                    try:
                        chunk = await asyncio.wait_for(
                            stderr_gen.__anext__(), timeout=0.01
                        )
                        yield chunk
                    except (StopAsyncIteration, asyncio.TimeoutError):
                        pass

                    # Check if process is done
                    if proc.returncode is not None:
                        break

                    await asyncio.sleep(0.01)
            except Exception:
                pass

            await asyncio.wait_for(proc.wait(), timeout=req.timeout)
            usage = posix_res.getrusage(posix_res.RUSAGE_CHILDREN)

            created, modified = self._file_changes_since(script_path)
            commit, diff = ("", "")
            if proc.returncode == 0 and req.persist_files:
                commit, diff = self.repo.add_and_commit(
                    f"Run {script_path.name}"
                )

            yield {
                "stream": "end",
                "data": "",
                "return_code": proc.returncode,
                "wall_time": getattr(usage, "ru_wallclock", 0),
                "cpu_time": usage.ru_utime + usage.ru_stime,
                "max_rss_mb": usage.ru_maxrss / 1024,
                "git_commit": commit or None,
                "git_diff": diff or None,
                "files_created": created,
                "files_modified": modified,
                "done": True,
            }

    # ------------------------------------------------ private helpers
    async def _prepare_fs(self, req: CodeExecutionRequest) -> tuple[str, Path]:
        cwd = self.root / (req.working_directory or "")
        cwd.mkdir(parents=True, exist_ok=True)

        script = cwd / f"{uuid.uuid4().hex[:8]}.{self._ext(req.language)}"
        async with aiofiles.open(script, "w") as f:
            await f.write(req.code)
        return str(cwd), script

    def _file_changes_since(
        self, exclude: Path
    ) -> tuple[list[str], list[str]]:
        created: list[str] = []
        modified: list[str] = []
        # Cheap heuristic: git status --porcelain
        res = subprocess.run(
            ["git", "-C", self.root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.splitlines():
            flag, path = line[:2], line[3:]
            if Path(path) == exclude.relative_to(self.root):
                continue
            if flag in {"??"}:
                created.append(path)
            else:
                modified.append(path)
        return created, modified

    async def _spawn_process(
        self, req: CodeExecutionRequest, script: Path, cwd: str
    ) -> tuple[str, str, int, posix_res.struct_rusage]:
        """
        Execute and collect entire output (non‑stream path).
        """
        runner = self._build_cmd(req.language, script)
        env = self._sandbox_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *runner,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=lambda: self._apply_limits(req),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=req.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return (
                    "",
                    f"Timed‑out after {req.timeout}s",
                    124,
                    posix_res.getrusage(posix_res.RUSAGE_CHILDREN),
                )
        except FileNotFoundError as e:
            return (
                "",
                str(e),
                127,
                posix_res.getrusage(posix_res.RUSAGE_CHILDREN),
            )

        usage = posix_res.getrusage(posix_res.RUSAGE_CHILDREN)
        return (
            stdout.decode(),
            stderr.decode(),
            proc.returncode,
            usage,
        )

    # ---------- misc
    def _build_cmd(self, lang: str, script: Path) -> list[str]:
        if lang not in self.RUNNER_TABLE:
            raise ValueError(f"Unsupported language '{lang}'")
        return [a.format(file=script) for a in self.RUNNER_TABLE[lang]]

    def _ext(self, lang: str) -> str:
        return {"python": "py", "bash": "sh", "node": "js"}.get(lang, lang)

    def _apply_limits(self, req: CodeExecutionRequest) -> None:
        try:
            import sys

            if (
                sys.platform != "darwin"
            ):  # Skip on macOS for now due to compatibility issues
                # CPU seconds
                posix_res.setrlimit(
                    posix_res.RLIMIT_CPU, (req.cpu_time, req.cpu_time)
                )
                # Address space
                mem_bytes = req.memory_mb * 1024 * 1024
                posix_res.setrlimit(
                    posix_res.RLIMIT_AS, (mem_bytes, mem_bytes)
                )
                # Disable fork bomb
                posix_res.setrlimit(posix_res.RLIMIT_NPROC, (50, 50))
                # No core dumps
                posix_res.setrlimit(posix_res.RLIMIT_CORE, (0, 0))
        except (OSError, AttributeError):
            # Windows or other systems that don't support these limits
            pass

    def _sandbox_env(self) -> dict[str, str]:
        env = {"PYTHONUNBUFFERED": "1"}
        safe = ["PATH", "LANG", "LC_ALL"]
        env.update({k: v for k, v in os.environ.items() if k in safe})
        return env


async def _readlines(stream: asyncio.StreamReader):
    """
    Non‑blocking async line iterator (keeps trailing newline).
    """
    buf = bytearray()
    while not stream.at_eof():
        chunk = await stream.read(1024)
        if not chunk:
            break
        buf.extend(chunk)
        while b"\n" in buf:
            line, _, buf = buf.partition(b"\n")
            yield line + b"\n"
    if buf:
        yield bytes(buf)


# --------------------------------------------------------------------------- endpoint config
SANDBOX_CONFIG = EndpointConfig(
    name="code_sandbox",
    provider="local_sandbox",
    base_url="local://sandbox",
    endpoint="execute",
    method="POST",
    openai_compatible=False,
    auth_type="none",
    request_options=CodeExecutionRequest,
)


# --------------------------------------------------------------------------- endpoint
class CodeSandboxEndpoint(Endpoint):
    def __init__(
        self,
        config: EndpointConfig = SANDBOX_CONFIG,
        sandbox_base_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.base = Path(
            sandbox_base_dir or tempfile.mkdtemp(prefix="lionagi_sbx_")
        )
        self.base.mkdir(parents=True, exist_ok=True)
        self._sandboxes: dict[str, CodeSandbox] = {}
        log.info("Sandbox root: %s", self.base)

    # -------------- public API
    async def call(
        self,
        request: CodeExecutionRequest | dict[str, Any],
        session_id: str = "default",
        **kwargs,
    ) -> dict[str, Any]:
        req = (
            request
            if isinstance(request, CodeExecutionRequest)
            else CodeExecutionRequest(**request)
        )
        sbx = self._ensure(session_id)
        res = await sbx.run(req)
        return res.model_dump()

    async def stream(
        self,
        request: CodeExecutionRequest | dict[str, Any],
        session_id: str = "default",
        **kwargs,
    ):
        req = (
            request
            if isinstance(request, CodeExecutionRequest)
            else CodeExecutionRequest(**request)
        )
        sbx = self._ensure(session_id)
        async for chunk in sbx.stream(req):
            yield chunk

    # -------------- utilities
    def cleanup_sandbox(self, session_id: str = "default"):
        if session_id in self._sandboxes:
            path = self._sandboxes.pop(session_id).root
            shutil.rmtree(path, ignore_errors=True)
            log.info("Removed sandbox %s", path)

    def cleanup_all_sandboxes(self):
        for sid in list(self._sandboxes):
            self.cleanup_sandbox(sid)

    def get_sandbox_info(self, session_id: str = "default"):
        sbx = self._sandboxes.get(session_id)
        if not sbx:
            return {"exists": False}
        tree = list(
            str(p.relative_to(sbx.root))
            for p in sbx.root.rglob("*")
            if p.is_file() and ".git" not in p.parts
        )
        return {
            "exists": True,
            "root": str(sbx.root),
            "files": tree,
            "last_commit": sbx.repo.snapshot_tree(),
        }

    # -------------- internals
    def _ensure(self, session_id: str) -> CodeSandbox:
        if session_id not in self._sandboxes:
            path = self.base / f"sess_{session_id}"
            path.mkdir(parents=True, exist_ok=True)
            self._sandboxes[session_id] = CodeSandbox(path)
        return self._sandboxes[session_id]

    def __del__(self):
        with contextlib.suppress(Exception):
            self.cleanup_all_sandboxes()
