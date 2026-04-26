# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from lionagi.protocols.action.tool import Tool

from ..base import LionTool

_TRUNCATE_THRESHOLD = 100_000
_TRUNCATE_HALF = 50_000


class BashRequest(BaseModel):
    command: str = Field(
        ...,
        description=(
            "Shell command to execute. Runs via /bin/sh (shell=True). "
            "Avoid interactive or long-running commands. "
            "Use absolute paths when the working directory matters."
        ),
    )
    timeout: int | None = Field(
        None,
        description=(
            "Maximum execution time in milliseconds. "
            "Defaults to 30000 (30 s). Maximum allowed is 300000 (5 min). "
            "The process is killed if it exceeds this limit."
        ),
    )
    cwd: str | None = Field(
        None,
        description=(
            "Working directory for the command. "
            "If omitted, inherits the current process working directory."
        ),
    )


class BashResponse(BaseModel):
    stdout: str = Field(
        default="",
        description="Standard output from the command.",
    )
    stderr: str = Field(
        default="",
        description="Standard error from the command.",
    )
    return_code: int = Field(
        ...,
        description="Exit code returned by the process. 0 typically means success.",
    )
    timed_out: bool = Field(
        default=False,
        description="True if the command was killed due to the timeout limit.",
    )


def _truncate(text: str) -> str:
    if len(text) <= _TRUNCATE_THRESHOLD:
        return text
    return (
        text[:_TRUNCATE_HALF]
        + f"\n\n[... {len(text) - _TRUNCATE_THRESHOLD} chars truncated ...]\n\n"
        + text[-_TRUNCATE_HALF:]
    )


class BashTool(LionTool):
    is_lion_system_tool = True
    system_tool_name = "bash_tool"

    def __init__(self):
        self._tool = None

    def handle_request(self, request: BashRequest) -> BashResponse:
        if isinstance(request, dict):
            request = BashRequest(**request)

        timeout_ms = request.timeout if request.timeout is not None else 30_000
        timeout_ms = min(max(timeout_ms, 1), 300_000)
        timeout_sec = timeout_ms / 1000.0

        try:
            result = subprocess.run(
                request.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=request.cwd or None,
            )
            return BashResponse(
                stdout=_truncate(result.stdout),
                stderr=_truncate(result.stderr),
                return_code=result.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return BashResponse(
                stdout="",
                stderr=f"Command timed out after {timeout_ms} ms",
                return_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return BashResponse(
                stdout="",
                stderr=f"Execution error: {e}",
                return_code=-1,
                timed_out=False,
            )

    def to_tool(self) -> Tool:
        if self._tool is None:

            def bash_tool(**kwargs):
                """
                Execute a shell command and return its output.

                Runs the command via /bin/sh. Captures stdout, stderr, and the exit
                code. Enforces a configurable timeout (default 30 s, max 5 min).
                Output exceeding 100 000 characters is truncated symmetrically
                (first 50 K + last 50 K). Prefer absolute paths; set cwd when the
                command depends on a specific working directory.
                """
                return self.handle_request(BashRequest(**kwargs)).model_dump()

            if self.system_tool_name != "bash_tool":
                bash_tool.__name__ = self.system_tool_name

            self._tool = Tool(
                func_callable=bash_tool,
                request_options=BashRequest,
            )
        return self._tool
