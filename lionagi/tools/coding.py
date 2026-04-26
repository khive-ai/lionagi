# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import subprocess
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from lionagi.ln.concurrency import run_sync
from lionagi.protocols.action.tool import Tool
from lionagi.service.token_calculator import TokenCalculator

from .base import LionTool

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


# ---------------------------------------------------------------------------
# Request models (LLM-facing schemas)
# ---------------------------------------------------------------------------


class ReaderAction(str, Enum):
    read = "read"
    list_dir = "list_dir"


class ReaderRequest(BaseModel):
    action: ReaderAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'read': Read a file and return its contents with line numbers.\n"
            "- 'list_dir': List files in a directory."
        ),
    )
    path: str = Field(
        ...,
        description="Absolute path to a file (for 'read') or directory (for 'list_dir').",
    )
    offset: int | None = Field(
        None,
        description="Zero-indexed line number to start reading from. Defaults to 0.",
    )
    limit: int | None = Field(
        None,
        description="Maximum number of lines to return. Defaults to 2000.",
    )
    recursive: bool | None = Field(
        None,
        description="Whether to list subdirectories recursively. Only for 'list_dir'.",
    )
    file_types: list[str] | None = Field(
        None,
        description="Filter by extensions (e.g. ['.py', '.rs']). Only for 'list_dir'.",
    )


class EditorAction(str, Enum):
    write = "write"
    edit = "edit"


class EditorRequest(BaseModel):
    action: EditorAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'write': Create or overwrite a file. Creates parent dirs automatically.\n"
            "- 'edit': Exact string replacement. Fails if old_string not found or ambiguous."
        ),
    )
    file_path: str = Field(
        ...,
        description="Absolute path to the target file.",
    )
    content: str | None = Field(
        None,
        description="Full file content. Required for 'write'.",
    )
    old_string: str | None = Field(
        None,
        description="Exact text to find. Required for 'edit'. Must match byte-for-byte.",
    )
    new_string: str | None = Field(
        None,
        description="Replacement text. Required for 'edit'. Empty string = deletion.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences. If False and multiple matches, edit fails.",
    )


class BashRequest(BaseModel):
    command: str = Field(
        ...,
        description="Shell command to execute.",
    )
    timeout: int | None = Field(
        None,
        description="Timeout in milliseconds. Default 30000, max 300000.",
    )
    cwd: str | None = Field(
        None,
        description="Working directory. Defaults to current directory.",
    )


class SearchAction(str, Enum):
    grep = "grep"
    find = "find"


class SearchRequest(BaseModel):
    action: SearchAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'grep': Search file contents with regex pattern.\n"
            "- 'find': Find files by name pattern."
        ),
    )
    pattern: str = Field(
        ...,
        description="Regex pattern (for 'grep') or glob pattern (for 'find').",
    )
    path: str | None = Field(
        None,
        description="File or directory to search in. Defaults to current directory.",
    )
    include: str | None = Field(
        None,
        description="Glob filter for grep, e.g. '*.py'. Only for 'grep'.",
    )
    max_results: int | None = Field(
        None,
        description="Max results to return. Default 50 for grep, 100 for find.",
    )


class ContextAction(str, Enum):
    status = "status"
    get_messages = "get_messages"
    evict = "evict"
    evict_action_results = "evict_action_results"


class ContextRequest(BaseModel):
    action: ContextAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'status': Context usage — message count, types, token estimate.\n"
            "- 'get_messages': List messages with index, role, preview.\n"
            "- 'evict': Remove messages by index range (protects system message).\n"
            "- 'evict_action_results': Remove old tool outputs, keep last N."
        ),
    )
    start: int | None = Field(None, description="Start index (inclusive, 0-based).")
    end: int | None = Field(None, description="End index (exclusive, 0-based).")
    keep_last: int | None = Field(
        None,
        description="For 'evict_action_results': keep N most recent. Default 5.",
    )


# ---------------------------------------------------------------------------
# Blocking helpers (run via run_sync in async tools)
# ---------------------------------------------------------------------------


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml",
}


def _read_image_sync(path: str) -> dict:
    p = Path(path)
    ext = p.suffix.lower()
    media_type = _IMAGE_MEDIA_TYPES.get(ext, "image/png")
    try:
        raw = p.read_bytes()
    except OSError as e:
        return {"success": False, "error": str(e)}
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "success": True,
        "type": "image",
        "media_type": media_type,
        "content": f"data:{media_type};base64,{encoded}",
        "size_bytes": len(raw),
    }


def _read_file_sync(path: str, offset: int, max_lines: int) -> dict:
    p = Path(path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {path}"}
    if not p.is_file():
        return {"success": False, "error": f"Not a file: {path}"}

    if p.suffix.lower() in _IMAGE_EXTENSIONS:
        return _read_image_sync(path)

    try:
        with open(p, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return {"success": False, "error": f"Binary file: {path}"}
    except OSError as e:
        return {"success": False, "error": str(e)}

    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return {"success": False, "error": str(e)}

    selected = lines[offset : offset + max_lines]
    numbered = "".join(
        f"{offset + i + 1}\t{line}" for i, line in enumerate(selected)
    )

    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0

    return {"success": True, "content": numbered, "_resolved": str(p.resolve()), "_mtime": mtime}


def _list_dir_sync(path: str, recursive: bool, file_types: list[str] | None) -> dict:
    from lionagi.libs.file.process import dir_to_files

    try:
        files = dir_to_files(path, recursive=recursive, file_types=file_types)
        return {"success": True, "content": "\n".join(str(f) for f in files)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _write_file_sync(file_path: str, content: str) -> dict:
    p = Path(file_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return {"success": False, "error": str(e)}

    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0

    return {"success": True, "content": f"Written: {p} ({len(content)} chars)", "_resolved": str(p.resolve()), "_mtime": mtime}


def _edit_file_sync(file_path: str, old_string: str, new_string: str, replace_all: bool) -> dict:
    p = Path(file_path)
    try:
        original = p.read_text(encoding="utf-8")
    except OSError as e:
        return {"success": False, "error": str(e)}

    count = original.count(old_string)
    if count == 0:
        return {"success": False, "error": f"old_string not found in {file_path}"}
    if count > 1 and not replace_all:
        return {"success": False, "error": f"old_string appears {count} times. Set replace_all=True."}

    updated = original.replace(old_string, new_string, -1 if replace_all else 1)

    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as e:
        return {"success": False, "error": str(e)}

    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0

    idx = updated.find(new_string)
    s = max(0, idx - 40)
    e = min(len(updated), idx + len(new_string) + 40)
    snippet = updated[s:e]

    return {
        "success": True,
        "content": f"Replaced {count if replace_all else 1}x. ...{snippet}...",
        "_resolved": str(p.resolve()),
        "_mtime": mtime,
    }


def _subprocess_sync(cmd, shell: bool, timeout_s: float, cwd: str | None) -> dict:
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=timeout_s, cwd=cwd or None,
        )
        return {"stdout": result.stdout or "", "stderr": result.stderr or "", "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout_s}s", "returncode": -1, "timed_out": True}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


# ---------------------------------------------------------------------------
# CodingToolkit
# ---------------------------------------------------------------------------


class CodingToolkit(LionTool):
    """Coding tools bound to a Branch with shared file state and hooks.

    Usage::

        toolkit = CodingToolkit()

        # Register hooks before binding
        async def guard_destructive(tool_name, action, args):
            cmd = args.get("command", "")
            if "rm -rf" in cmd:
                raise PermissionError(f"Blocked: {cmd}")

        async def auto_format(tool_name, action, args, result):
            if result.get("success") and args.get("file_path", "").endswith(".py"):
                # run formatter...
                pass
            return result

        toolkit.pre("bash", guard_destructive)
        toolkit.post("editor", auto_format)

        tools = toolkit.bind(branch)
        branch.register_tools(tools)

    Hook signatures:
        pre:  async def handler(tool_name: str, action: str, args: dict) -> dict | None
              - Return modified args dict to override, or None to pass through.
              - Raise to abort the tool call (exception propagates as error result).
        post: async def handler(tool_name: str, action: str, args: dict, result: dict) -> dict | None
              - Return modified result dict to override, or None to pass through.
        on_error: async def handler(tool_name: str, action: str, args: dict, error: Exception) -> dict | None
              - Return a result dict to suppress the error, or None to propagate.
    """

    is_lion_system_tool = True
    system_tool_name = "coding_toolkit"

    def __init__(self):
        self._pre_hooks: dict[str, list[Callable]] = {}
        self._post_hooks: dict[str, list[Callable]] = {}
        self._error_hooks: dict[str, list[Callable]] = {}

    def pre(self, tool_name: str, handler: Callable) -> CodingToolkit:
        self._pre_hooks.setdefault(tool_name, []).append(handler)
        return self

    def post(self, tool_name: str, handler: Callable) -> CodingToolkit:
        self._post_hooks.setdefault(tool_name, []).append(handler)
        return self

    def on_error(self, tool_name: str, handler: Callable) -> CodingToolkit:
        self._error_hooks.setdefault(tool_name, []).append(handler)
        return self

    def _build_preprocessor(self, tool_name: str) -> Callable | None:
        """Build a chained preprocessor from registered pre-hooks for this tool.

        Returns a callable matching Tool.preprocessor signature:
            preprocessor(args: dict, **kwargs) -> dict
        """
        hooks = [
            *self._pre_hooks.get(tool_name, []),
            *self._pre_hooks.get("*", []),
        ]
        if not hooks:
            return None

        async def chained_pre(args: dict, **_kw) -> dict:
            for handler in hooks:
                result = await handler(tool_name, args.get("action", ""), args)
                if isinstance(result, dict):
                    args = result
            return args

        return chained_pre

    def _build_postprocessor(self, tool_name: str) -> Callable | None:
        """Build a chained postprocessor from registered post-hooks for this tool.

        Returns a callable matching Tool.postprocessor signature:
            postprocessor(result: Any, **kwargs) -> Any
        """
        hooks = [
            *self._post_hooks.get(tool_name, []),
            *self._post_hooks.get("*", []),
        ]
        if not hooks:
            return None

        async def chained_post(result: Any, **_kw) -> Any:
            if not isinstance(result, dict):
                return result
            for handler in hooks:
                modified = await handler(tool_name, "", {}, result)
                if isinstance(modified, dict):
                    result = modified
            return result

        return chained_post

    notify: bool = True
    notify_threshold: float = 0.7  # warn when context exceeds this fraction
    notify_max_tokens: int = 200_000  # assumed context window size

    def bind(self, branch: Branch) -> list[Tool]:
        from lionagi.protocols.messages import ActionResponse

        file_state: dict[str, float] = {}
        call_count = [0]
        msgs = branch.msgs
        notify = self.notify
        threshold = self.notify_threshold
        max_tokens = self.notify_max_tokens

        def _system_status() -> str | None:
            if not notify:
                return None
            call_count[0] += 1
            progression = msgs.progression
            pile = msgs.messages
            n_msgs = len(progression)
            n_files = len(file_state)

            est_tokens = 0
            n_action_results = 0
            for uid in progression:
                if uid in pile:
                    msg = pile[uid]
                    if isinstance(msg, ActionResponse):
                        n_action_results += 1
                    c = msg.content if hasattr(msg, "content") else ""
                    if c:
                        est_tokens += TokenCalculator.tokenize(
                            str(c) if not isinstance(c, str) else c
                        )

            usage_pct = est_tokens / max_tokens if max_tokens > 0 else 0
            parts = [f"context {est_tokens // 1000}k/{max_tokens // 1000}k tokens ({usage_pct:.0%})"]
            parts.append(f"{n_msgs} messages")
            if n_action_results > 0:
                parts.append(f"{n_action_results} action results")
            if n_files > 0:
                parts.append(f"{n_files} files tracked")

            status = f"[System: {', '.join(parts)}]"

            if usage_pct >= 0.9:
                status += " ⚠️ Context nearly full — evict old action results now."
            elif usage_pct >= threshold:
                status += " Consider evicting earlier action results to free space."

            return status

        def _check_read_guard(path: str) -> str | None:
            resolved = str(Path(path).resolve())
            if resolved not in file_state:
                return f"Must read file before editing: {path}"
            try:
                current_mtime = Path(path).stat().st_mtime
            except OSError:
                return None
            if current_mtime != file_state[resolved]:
                return f"File changed since last read: {path}. Read it again."
            return None

        def _track(result: dict):
            resolved = result.pop("_resolved", None)
            mtime = result.pop("_mtime", None)
            if resolved and mtime is not None:
                file_state[resolved] = mtime

        # -- Reader ----------------------------------------------------------

        async def reader(
            action: str,
            path: str,
            offset: int = None,
            limit: int = None,
            recursive: bool = None,
            file_types: list[str] = None,
        ) -> dict:
            """Read files or list directory contents.

            Use action='read' to get file contents with line numbers.
            Use action='list_dir' to list files. Always read a file before editing it.
            """
            if action == "read":
                start = max(0, offset or 0)
                max_lines = limit if (limit and limit > 0) else 2000
                result = await run_sync(_read_file_sync, path, start, max_lines)
                _track(result)
                return result
            elif action == "list_dir":
                return await run_sync(_list_dir_sync, path, bool(recursive), file_types)
            return {"success": False, "error": f"Unknown action: {action}"}

        # -- Editor ----------------------------------------------------------

        async def editor(
            action: str,
            file_path: str,
            content: str = None,
            old_string: str = None,
            new_string: str = None,
            replace_all: bool = False,
        ) -> dict:
            """Write or edit files. You must read a file before editing it.

            Use action='write' to create or overwrite. Use action='edit' for
            exact string replacement — safer than full rewrites.
            """
            if action == "write":
                if content is None:
                    return {"success": False, "error": "'content' required for write"}
                if Path(file_path).exists():
                    guard = _check_read_guard(file_path)
                    if guard:
                        return {"success": False, "error": guard}
                result = await run_sync(_write_file_sync, file_path, content)
                _track(result)
                return result
            elif action == "edit":
                if old_string is None:
                    return {"success": False, "error": "'old_string' required for edit"}
                if new_string is None:
                    return {"success": False, "error": "'new_string' required for edit"}
                guard = _check_read_guard(file_path)
                if guard:
                    return {"success": False, "error": guard}
                result = await run_sync(_edit_file_sync, file_path, old_string, new_string, replace_all)
                _track(result)
                return result
            return {"success": False, "error": f"Unknown action: {action}"}

        # -- Bash ------------------------------------------------------------

        async def bash(
            command: str,
            timeout: int = None,
            cwd: str = None,
        ) -> dict:
            """Execute a shell command and return stdout, stderr, and return code.

            Use for running builds, tests, git commands, and any system operations.
            Output is truncated if it exceeds 100K characters.
            """
            timeout_ms = max(1, min(timeout or 30000, 300000))
            timeout_s = timeout_ms / 1000.0

            result = await run_sync(_subprocess_sync, command, True, timeout_s, cwd)

            max_chars = 100_000
            for key in ("stdout", "stderr"):
                val = result.get(key, "")
                if len(val) > max_chars:
                    half = max_chars // 2
                    result[key] = val[:half] + f"\n\n[...truncated {len(val) - max_chars} chars...]\n\n" + val[-half:]

            result.setdefault("timed_out", False)
            result["return_code"] = result.pop("returncode", -1)
            return result

        # -- Search ----------------------------------------------------------

        async def search(
            action: str,
            pattern: str,
            path: str = None,
            include: str = None,
            max_results: int = None,
        ) -> dict:
            """Search file contents (grep) or find files by name.

            Use action='grep' to search with regex. Use action='find' for file names.
            Results are capped at max_results to prevent context overflow.
            """
            if action == "grep":
                search_path = path or "."
                limit = max_results or 50
                cmd = ["grep", "-rn", "-E", pattern, search_path]
                if include:
                    cmd.insert(3, f"--include={include}")
                raw = await run_sync(_subprocess_sync, cmd, False, 30.0, None)
                if raw.get("returncode") == 2:
                    return {"success": False, "error": raw["stderr"].strip()}
                lines = raw["stdout"].strip().split("\n") if raw["stdout"].strip() else []
                total = len(lines)
                return {"success": True, "content": "\n".join(lines[:limit]), "total_matches": total, "shown": min(total, limit)}
            elif action == "find":
                search_path = path or "."
                limit = max_results or 100
                cmd = ["find", search_path, "-name", pattern]
                raw = await run_sync(_subprocess_sync, cmd, False, 30.0, None)
                if raw.get("returncode", 0) != 0 and raw.get("stderr", "").strip():
                    return {"success": False, "error": raw["stderr"].strip()}
                lines = raw["stdout"].strip().split("\n") if raw["stdout"].strip() else []
                total = len(lines)
                return {"success": True, "content": "\n".join(lines[:limit]), "total_found": total, "shown": min(total, limit)}
            return {"success": False, "error": f"Unknown action: {action}"}

        # -- Context ---------------------------------------------------------

        async def context(
            action: str,
            start: int = None,
            end: int = None,
            keep_last: int = None,
        ) -> dict:
            """Manage your conversation context — check usage, list messages, evict old ones.

            Use this to stay within context limits during long tasks. Evict verbose
            tool outputs you no longer need to free space for new work.
            """
            progression = msgs.progression
            pile = msgs.messages

            if action == "status":
                total = len(progression)
                by_type: dict[str, int] = {}
                total_tokens = 0
                for uid in progression:
                    if uid in pile:
                        msg = pile[uid]
                        role = msg.role if hasattr(msg, "role") else type(msg).__name__
                        by_type[role] = by_type.get(role, 0) + 1
                        c = msg.content if hasattr(msg, "content") else ""
                        if c:
                            total_tokens += TokenCalculator.tokenize(str(c) if not isinstance(c, str) else c)
                return {
                    "success": True,
                    "total_messages": total,
                    "by_type": by_type,
                    "estimated_tokens": total_tokens,
                    "files_tracked": len(file_state),
                }

            elif action == "get_messages":
                s = max(0, start or 0)
                e = min(len(progression), end or len(progression))
                summaries = []
                for i in range(s, e):
                    uid = progression[i]
                    if uid in pile:
                        msg = pile[uid]
                        role = msg.role if hasattr(msg, "role") else type(msg).__name__
                        c = ""
                        if hasattr(msg, "content") and msg.content:
                            raw = str(msg.content) if not isinstance(msg.content, str) else msg.content
                            c = raw[:120].replace("\n", " ")
                            if len(raw) > 120:
                                c += "..."
                        summaries.append(f"[{i}] {role}: {c}")
                return {"success": True, "range": f"[{s}:{e}] of {len(progression)}", "messages": summaries}

            elif action == "evict":
                s = max(1, start or 1)
                e = end if end is not None else s + 1
                e = min(len(progression), e)
                if s >= e:
                    return {"success": False, "error": f"Invalid range [{s}:{e})"}
                uids = [progression[i] for i in range(s, e) if i < len(progression)]
                removed = 0
                for uid in uids:
                    if uid in pile:
                        pile.exclude(uid)
                        removed += 1
                return {"success": True, "removed": removed, "remaining": len(progression)}

            elif action == "evict_action_results":
                keep = keep_last if keep_last is not None else 5
                ar_indices = []
                for i, uid in enumerate(progression):
                    if uid in pile and isinstance(pile[uid], ActionResponse):
                        ar_indices.append((i, uid))
                if len(ar_indices) <= keep:
                    return {"success": True, "removed": 0, "message": f"Only {len(ar_indices)} action results, keeping all."}
                to_evict = ar_indices[:-keep] if keep > 0 else ar_indices
                removed = 0
                for _, uid in to_evict:
                    if uid in pile:
                        pile.exclude(uid)
                        removed += 1
                return {"success": True, "removed": removed, "remaining": len(progression)}

            return {"success": False, "error": f"Unknown action: {action}"}

        # -- System notification as built-in post-hook -----------------------

        async def _notify_post(tool_name: str, action: str, args: dict, result: dict) -> dict | None:
            status = _system_status()
            if status and isinstance(result, dict):
                result["system"] = status
            return result

        if notify:
            self.post("*", _notify_post)

        # -- Assemble (hooks wired via Tool's native pre/postprocessor) ------

        tool_defs = [
            ("reader", reader, ReaderRequest),
            ("editor", editor, EditorRequest),
            ("bash", bash, BashRequest),
            ("search", search, SearchRequest),
            ("context", context, ContextRequest),
        ]

        tools = []
        for name, func, request_cls in tool_defs:
            tools.append(Tool(
                func_callable=func,
                request_options=request_cls,
                preprocessor=self._build_preprocessor(name),
                postprocessor=self._build_postprocessor(name),
            ))
        return tools

    def to_tool(self) -> Tool:
        raise NotImplementedError(
            "CodingToolkit requires branch context. Use toolkit.bind(branch) instead."
        )
