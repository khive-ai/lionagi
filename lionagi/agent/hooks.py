# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Built-in hook implementations for coding agents.

Usage::

    from lionagi.agent.hooks import guard_destructive, log_tool_use

    config = AgentConfig.coding()
    config.pre("bash", guard_destructive)
    config.post("*", log_tool_use)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bgit\s+push\s+--force\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-fd\b",
    r"\bdrop\s+table\b",
    r"\bdrop\s+database\b",
    r"\btruncate\s+table\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd[a-z]",
]

_DESTRUCTIVE_RE = re.compile("|".join(_DESTRUCTIVE_PATTERNS), re.IGNORECASE)


async def guard_destructive(tool_name: str, action: str, args: dict) -> dict | None:
    """Pre-hook: block destructive bash commands."""
    cmd = args.get("command", "")
    if _DESTRUCTIVE_RE.search(cmd):
        raise PermissionError(
            f"Blocked destructive command: {cmd}\n"
            "If you need this, explain why and ask the user to run it manually."
        )
    return None


async def guard_paths(
    allowed_paths: list[str] | None = None,
    denied_paths: list[str] | None = None,
):
    """Factory: create a pre-hook that restricts file access by path.

    Usage::

        config.pre("reader", guard_paths(allowed_paths=["/Users/me/project/"]))
        config.pre("editor", guard_paths(denied_paths=[".env", "credentials"]))
    """

    async def _guard(tool_name: str, action: str, args: dict) -> dict | None:
        path = args.get("path") or args.get("file_path") or ""
        if allowed_paths:
            if not any(path.startswith(p) for p in allowed_paths):
                raise PermissionError(f"Path not in allowed list: {path}")
        if denied_paths:
            if any(d in path for d in denied_paths):
                raise PermissionError(f"Path matches deny rule: {path}")
        return None

    return _guard


async def log_tool_use(
    tool_name: str, action: str, args: dict, result: dict
) -> dict | None:
    """Post-hook: log tool usage for observability."""
    success = result.get("success", result.get("return_code") == 0)
    logger.info("tool=%s action=%s success=%s", tool_name, action, success)
    return None


async def auto_format_python(
    tool_name: str, action: str, args: dict, result: dict
) -> dict | None:
    """Post-hook: run ruff format on edited Python files."""
    if not result.get("success"):
        return None

    file_path = args.get("file_path", "")
    if not file_path.endswith(".py"):
        return None

    import subprocess

    from lionagi.ln.concurrency import run_sync
    from lionagi.tools.coding import _subprocess_sync

    await run_sync(_subprocess_sync, f"ruff format {file_path}", True, 10.0, None)
    return None
