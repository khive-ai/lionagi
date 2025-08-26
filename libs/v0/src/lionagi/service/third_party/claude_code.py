# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import logging
import shutil
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from dataclasses import field as datafield
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from textwrap import shorten
from typing import Any, Literal

from json_repair import repair_json
from pydantic import BaseModel, Field, field_validator, model_validator

from lionagi import ln
from lionagi.libs.schema.as_readable import as_readable
from lionagi.utils import is_coro_func, is_import_installed

HAS_CLAUDE_CODE_SDK = is_import_installed("claude_code_sdk")
HAS_CLAUDE_CODE_CLI = False
CLAUDE_CLI = None

if (c := (shutil.which("claude") or "claude")) and shutil.which(c):
    HAS_CLAUDE_CODE_CLI = True
    CLAUDE_CLI = c

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("claude-cli")

# --------------------------------------------------------------------------- constants
ClaudePermission = Literal[
    "default",
    "acceptEdits",
    "bypassPermissions",
    "dangerously-skip-permissions",
]

CLAUDE_CODE_OPTION_PARAMS = {
    "allowed_tools",
    "max_thinking_tokens",
    "mcp_tools",
    "mcp_servers",
    "permission_mode",
    "continue_conversation",
    "resume",
    "max_turns",
    "disallowed_tools",
    "model",
    "permission_prompt_tool_name",
    "cwd",
    "system_prompt",
    "append_system_prompt",
}


__all__ = (
    "ClaudeCodeRequest",
    "ClaudeChunk",
    "ClaudeSession",
    "stream_claude_code_cli",
    "stream_cc_sdk_events",
)


# --------------------------------------------------------------------------- request model
class ClaudeCodeRequest(BaseModel):
    # -- conversational bits -------------------------------------------------
    prompt: str = Field(description="The prompt for Claude Code")
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    max_turns: int | None = None
    continue_conversation: bool = False
    resume: str | None = None

    # -- repo / workspace ----------------------------------------------------
    repo: Path = Field(default_factory=Path.cwd, exclude=True)
    ws: str | None = None  # sub-directory under repo
    add_dir: str | None = None  # extra read-only mount
    allowed_tools: list[str] | None = None

    # -- runtime & safety ----------------------------------------------------
    model: Literal["sonnet", "opus"] | str | None = "sonnet"
    max_thinking_tokens: int | None = None
    mcp_tools: list[str] = Field(default_factory=list)
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    mcp_config: str | Path | None = Field(None, exclude=True)
    permission_mode: ClaudePermission | None = None
    permission_prompt_tool_name: str | None = None
    disallowed_tools: list[str] = Field(default_factory=list)

    # -- internal use --------------------------------------------------------
    auto_finish: bool = Field(
        default=False,
        description="Automatically finish the conversation after the first response",
    )
    verbose_output: bool = Field(default=False)
    cli_display_theme: Literal["light", "dark"] = "dark"
    cli_include_summary: bool = Field(default=False)

    # ------------------------ validators & helpers --------------------------
    @field_validator("permission_mode", mode="before")
    def _norm_perm(cls, v):
        if v in {
            "dangerously-skip-permissions",
            "--dangerously-skip-permissions",
        }:
            return "bypassPermissions"
        return v

    # Workspace path derived from repo + ws
    def cwd(self) -> Path:
        if not self.ws:
            return self.repo

        # Convert to Path object for proper validation
        ws_path = Path(self.ws)

        # Check for absolute paths or directory traversal attempts
        if ws_path.is_absolute():
            raise ValueError(
                f"Workspace path must be relative, got absolute: {self.ws}"
            )

        if ".." in ws_path.parts:
            raise ValueError(
                f"Directory traversal detected in workspace path: {self.ws}"
            )

        # Resolve paths to handle symlinks and normalize
        repo_resolved = self.repo.resolve()
        result = (self.repo / ws_path).resolve()

        # Ensure the resolved path is within the repository bounds
        try:
            result.relative_to(repo_resolved)
        except ValueError:
            raise ValueError(
                f"Workspace path escapes repository bounds. "
                f"Repository: {repo_resolved}, Workspace: {result}"
            )

        return result

    @model_validator(mode="after")
    def _check_perm_workspace(self):
        if self.permission_mode == "bypassPermissions":
            # Use secure path validation with resolved paths
            repo_resolved = self.repo.resolve()
            cwd_resolved = self.cwd().resolve()

            # Check if cwd is within repo bounds using proper path methods
            try:
                cwd_resolved.relative_to(repo_resolved)
            except ValueError:
                raise ValueError(
                    f"With bypassPermissions, workspace must be within repository bounds. "
                    f"Repository: {repo_resolved}, Workspace: {cwd_resolved}"
                )
        return self

    # ------------------------ CLI helpers -----------------------------------
    def as_cmd_args(self) -> list[str]:
        """Build argument list for the *Node* `claude` CLI."""
        args: list[str] = ["-p", self.prompt, "--output-format", "stream-json"]
        if self.allowed_tools:
            args.append("--allowedTools")
            for tool in self.allowed_tools:
                args.append(f'"{tool}"')

        if self.disallowed_tools:
            args.append("--disallowedTools")
            for tool in self.disallowed_tools:
                args.append(f'"{tool}"')

        if self.resume:
            args += ["--resume", self.resume]
        elif self.continue_conversation:
            args.append("--continue")

        if self.max_turns:
            # +1 because CLI counts *pairs*
            args += ["--max-turns", str(self.max_turns + 1)]

        if self.permission_mode == "bypassPermissions":
            args += ["--dangerously-skip-permissions"]

        if self.add_dir:
            args += ["--add-dir", self.add_dir]

        if self.permission_prompt_tool_name:
            args += [
                "--permission-prompt-tool",
                self.permission_prompt_tool_name,
            ]

        if self.mcp_config:
            args += ["--mcp-config", f'"{self.mcp_config}"']

        args += ["--model", self.model or "sonnet", "--verbose"]
        return args

    # ------------------------ SDK helpers -----------------------------------
    def as_claude_options(self):
        from claude_code_sdk import ClaudeCodeOptions

        data = {
            k: v
            for k, v in self.model_dump(exclude_none=True).items()
            if k in CLAUDE_CODE_OPTION_PARAMS
        }
        return ClaudeCodeOptions(**data)

    # ------------------------ convenience constructor -----------------------
    @classmethod
    def create(
        cls,
        messages: list[dict[str, Any]],
        resume: str | None = None,
        continue_conversation: bool | None = None,
        **kwargs,
    ):
        if not messages:
            raise ValueError("messages may not be empty")

        prompt = ""

        # 1. if resume or continue_conversation, use the last message
        if resume or continue_conversation:
            continue_conversation = True
            prompt = messages[-1]["content"]
            if isinstance(prompt, (dict, list)):
                prompt = ln.json_dumps(prompt)

        # 2. else, use entire messages except system message
        else:
            prompts = []
            continue_conversation = False
            for message in messages:
                if message["role"] != "system":
                    content = message["content"]
                    prompts.append(
                        ln.json_dumps(content)
                        if isinstance(content, (dict, list))
                        else content
                    )

            prompt = "\n".join(prompts)

        # 3. assemble the request data
        data: dict[str, Any] = dict(
            prompt=prompt,
            resume=resume,
            continue_conversation=bool(continue_conversation),
        )

        # 4. extract system prompt if available
        if (messages[0]["role"] == "system") and (
            resume or continue_conversation
        ):
            data["system_prompt"] = messages[0]["content"]
        if kwargs.get("append_system_prompt"):
            data["append_system_prompt"] = str(
                kwargs.get("append_system_prompt")
            )

        data.update(kwargs)
        return cls.model_validate(data, strict=False)


@dataclass
class ClaudeChunk:
    """Low-level wrapper around every NDJSON object coming from the CLI."""

    raw: dict[str, Any]
    type: str
    # convenience views
    thinking: str | None = None
    text: str | None = None
    tool_use: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None


@dataclass
class ClaudeSession:
    """Aggregated view of a whole CLI conversation."""

    session_id: str | None = None
    model: str | None = None

    # chronological log
    chunks: list[ClaudeChunk] = datafield(default_factory=list)

    # materialised views
    thinking_log: list[str] = datafield(default_factory=list)
    messages: list[dict[str, Any]] = datafield(default_factory=list)
    tool_uses: list[dict[str, Any]] = datafield(default_factory=list)
    tool_results: list[dict[str, Any]] = datafield(default_factory=list)

    # final summary
    result: str = ""
    usage: dict[str, Any] = datafield(default_factory=dict)
    total_cost_usd: float | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    is_error: bool = False
    summary: dict | None = None

    def populate_summary(self) -> None:
        self.summary = _extract_summary(self)


def _extract_summary(session: ClaudeSession) -> dict[str, Any]:
    tool_counts = {}
    tool_details = []
    file_operations = {"reads": [], "writes": [], "edits": []}
    key_actions = []

    # Process tool uses from the clean materialized view
    for tool_use in session.tool_uses:
        tool_name = tool_use.get("name", "unknown")
        tool_input = tool_use.get("input", {})
        tool_id = tool_use.get("id", "")

        # Count tool usage
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Store detailed info
        tool_details.append(
            {"tool": tool_name, "id": tool_id, "input": tool_input}
        )

        # Categorize file operations and actions
        if tool_name in ["Read", "read"]:
            file_path = tool_input.get("file_path", "unknown")
            file_operations["reads"].append(file_path)
            key_actions.append(f"Read {file_path}")

        elif tool_name in ["Write", "write"]:
            file_path = tool_input.get("file_path", "unknown")
            file_operations["writes"].append(file_path)
            key_actions.append(f"Wrote {file_path}")

        elif tool_name in ["Edit", "edit", "MultiEdit"]:
            file_path = tool_input.get("file_path", "unknown")
            file_operations["edits"].append(file_path)
            key_actions.append(f"Edited {file_path}")

        elif tool_name in ["Bash", "bash"]:
            command = tool_input.get("command", "")
            command_summary = (
                command[:50] + "..." if len(command) > 50 else command
            )
            key_actions.append(f"Ran: {command_summary}")

        elif tool_name in ["Glob", "glob"]:
            pattern = tool_input.get("pattern", "")
            key_actions.append(f"Searched files: {pattern}")

        elif tool_name in ["Grep", "grep"]:
            pattern = tool_input.get("pattern", "")
            key_actions.append(f"Searched content: {pattern}")

        elif tool_name in ["Task", "task"]:
            description = tool_input.get("description", "")
            key_actions.append(f"Spawned task: {description}")

        elif tool_name.startswith("mcp__"):
            # MCP tool usage - extract the operation type
            operation = tool_name.replace("mcp__", "")
            key_actions.append(f"MCP {operation}")

        elif tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])
            key_actions.append(f"Created {len(todos)} todos")

        else:
            key_actions.append(f"Used {tool_name}")

    # Deduplicate key actions
    key_actions = (
        list(dict.fromkeys(key_actions))
        if key_actions
        else ["No specific actions detected"]
    )

    # Deduplicate file paths
    for op_type in file_operations:
        file_operations[op_type] = list(
            dict.fromkeys(file_operations[op_type])
        )

    # Extract result summary (first 200 chars)
    result_summary = (
        (session.result[:200] + "...")
        if len(session.result) > 200
        else session.result
    )

    return {
        "tool_counts": tool_counts,
        "tool_details": tool_details,
        "file_operations": file_operations,
        "key_actions": key_actions,
        "total_tool_calls": sum(tool_counts.values()),
        "result_summary": result_summary,
        "usage_stats": {
            "total_cost_usd": session.total_cost_usd,
            "num_turns": session.num_turns,
            "duration_ms": session.duration_ms,
            "duration_api_ms": session.duration_api_ms,
            **session.usage,
        },
    }


async def _ndjson_from_cli(request: ClaudeCodeRequest):
    """
    Yields each JSON object emitted by the *claude-code* CLI.

    • Robust against UTF-8 splits across chunks (incremental decoder).
    • Robust against braces inside strings (uses json.JSONDecoder.raw_decode)
    • Falls back to `json_repair.repair_json` when necessary.
    """
    workspace = request.cwd()
    workspace.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        CLAUDE_CLI,
        *request.as_cmd_args(),
        cwd=str(workspace),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    decoder = codecs.getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    buffer: str = ""  # text buffer that may hold >1 JSON objects

    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break

            # 1) decode *incrementally* so we never split multibyte chars
            buffer += decoder.decode(chunk)

            # 2) try to peel off as many complete JSON objs as possible
            while buffer:
                buffer = buffer.lstrip()  # remove leading spaces/newlines
                if not buffer:
                    break
                try:
                    obj, idx = json_decoder.raw_decode(buffer)
                    yield obj
                    buffer = buffer[idx:]  # keep remainder for next round
                except json.JSONDecodeError:
                    # incomplete → need more bytes
                    break

        # 3) flush any tail bytes in the incremental decoder
        buffer += decoder.decode(b"", final=True)
        buffer = buffer.strip()
        if buffer:
            try:
                obj, idx = json_decoder.raw_decode(buffer)
                yield obj
            except json.JSONDecodeError:
                try:
                    fixed = repair_json(buffer)
                    yield json.loads(fixed)
                    log.warning(
                        "Repaired malformed JSON fragment at stream end"
                    )
                except Exception:
                    log.error(
                        "Skipped unrecoverable JSON tail: %.120s…", buffer
                    )

        # 4) propagate non-zero exit code
        if await proc.wait() != 0:
            err = (await proc.stderr.read()).decode().strip()
            raise RuntimeError(err or "CLI exited non-zero")

    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        await proc.wait()


def _stream_claude_code(request: ClaudeCodeRequest):
    from claude_code_sdk import query as sdk_query

    return sdk_query(
        prompt=request.prompt, options=request.as_claude_options()
    )


def stream_cc_sdk_events(request: ClaudeCodeRequest):
    if not HAS_CLAUDE_CODE_SDK:
        raise RuntimeError(
            "claude_code_sdk not installed (uv pip install lionagi[claude_code])"
        )

    return _stream_claude_code(request)


# --------------------------------------------------------------------------- SSE route
async def stream_cc_cli_events(request: ClaudeCodeRequest):
    if not CLAUDE_CLI:
        raise RuntimeError(
            "Claude CLI binary not found (npm i -g @anthropic-ai/claude-code)"
        )
    async for obj in _ndjson_from_cli(request):
        yield obj
    yield {"type": "done"}


print_readable = partial(as_readable, md=True, display_str=True)


def _pp_system(sys_obj: dict[str, Any], theme) -> None:
    txt = (
        f"◼️  **Claude Code Session**  \n"
        f"- id: `{sys_obj.get('session_id', '?')}`  \n"
        f"- model: `{sys_obj.get('model', '?')}`  \n"
        f"- tools: {', '.join(sys_obj.get('tools', [])[:8])}"
        + ("…" if len(sys_obj.get("tools", [])) > 8 else "")
    )
    print_readable(txt, border=False, theme=theme)


def _pp_thinking(thought: str, theme) -> None:
    text = f"""
    🧠 Thinking:
    {thought}
    """
    print_readable(text, border=True, theme=theme)


def _pp_assistant_text(text: str, theme) -> None:
    txt = f"""
    > 🗣️ Claude:
    {text}
    """
    print_readable(txt, theme=theme)


def _pp_tool_use(tu: dict[str, Any], theme) -> None:
    preview = shorten(str(tu["input"]).replace("\n", " "), 130)
    body = f"- 🔧 Tool Use — {tu['name']}({tu['id']}) - input: {preview}"
    print_readable(body, border=False, panel=False, theme=theme)


def _pp_tool_result(tr: dict[str, Any], theme) -> None:
    body_preview = shorten(str(tr["content"]).replace("\n", " "), 130)
    status = "ERR" if tr.get("is_error") else "OK"
    body = f"- 📄 Tool Result({tr['tool_use_id']}) - {status}\n\n\tcontent: {body_preview}"
    print_readable(body, border=False, panel=False, theme=theme)


def _pp_final(sess: ClaudeSession, theme) -> None:
    usage = sess.usage or {}
    txt = (
        f"### ✅ Session complete - {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n"
        f"**Result:**\n\n{sess.result or ''}\n\n"
        f"- cost: **${sess.total_cost_usd:.4f}**  \n"
        f"- turns: **{sess.num_turns}**  \n"
        f"- duration: **{sess.duration_ms} ms** (API {sess.duration_api_ms} ms)  \n"
        f"- tokens in/out: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}"
    )
    print_readable(txt, theme=theme)


# --------------------------------------------------------------------------- internal utils


async def _maybe_await(func, *args, **kw):
    """Call func which may be sync or async."""
    res = func(*args, **kw) if func else None
    if is_coro_func(res):
        await res


# --------------------------------------------------------------------------- main parser
async def stream_claude_code_cli(  # noqa: C901  (complexity from branching is fine here)
    request: ClaudeCodeRequest,
    session: ClaudeSession = ClaudeSession(),
    *,
    on_system: Callable[[dict[str, Any]], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_tool_use: Callable[[dict[str, Any]], None] | None = None,
    on_tool_result: Callable[[dict[str, Any]], None] | None = None,
    on_final: Callable[[ClaudeSession], None] | None = None,
) -> AsyncIterator[ClaudeChunk | dict | ClaudeSession]:
    """
    Consume the ND-JSON stream produced by ndjson_from_cli()
    and return a fully-populated ClaudeSession.

    If callbacks are omitted a default pretty-print is emitted.
    """
    stream = stream_cc_cli_events(request)
    theme = request.cli_display_theme or "light"

    async for obj in stream:
        typ = obj.get("type", "unknown")
        chunk = ClaudeChunk(raw=obj, type=typ)
        session.chunks.append(chunk)

        # ------------------------ SYSTEM -----------------------------------
        if typ == "system":
            data = obj
            session.session_id = data.get("session_id", session.session_id)
            session.model = data.get("model", session.model)
            await _maybe_await(on_system, data)
            if request.verbose_output:
                _pp_system(data, theme)
            yield data

        # ------------------------ ASSISTANT --------------------------------
        elif typ == "assistant":
            msg = obj["message"]
            session.messages.append(msg)

            for blk in msg.get("content", []):
                btype = blk.get("type")
                if btype == "thinking":
                    thought = blk.get("thinking", "").strip()
                    chunk.thinking = thought
                    session.thinking_log.append(thought)
                    await _maybe_await(on_thinking, thought)
                    if request.verbose_output:
                        _pp_thinking(thought, theme)

                elif btype == "text":
                    text = blk.get("text", "")
                    chunk.text = text
                    await _maybe_await(on_text, text)
                    if request.verbose_output:
                        _pp_assistant_text(text, theme)

                elif btype == "tool_use":
                    tu = {
                        "id": blk["id"],
                        "name": blk["name"],
                        "input": blk["input"],
                    }
                    chunk.tool_use = tu
                    session.tool_uses.append(tu)
                    await _maybe_await(on_tool_use, tu)
                    if request.verbose_output:
                        _pp_tool_use(tu, theme)

                elif btype == "tool_result":
                    tr = {
                        "tool_use_id": blk["tool_use_id"],
                        "content": blk["content"],
                        "is_error": blk.get("is_error", False),
                    }
                    chunk.tool_result = tr
                    session.tool_results.append(tr)
                    await _maybe_await(on_tool_result, tr)
                    if request.verbose_output:
                        _pp_tool_result(tr, theme)
            yield chunk

        # ------------------------ USER (tool_result containers) ------------
        elif typ == "user":
            msg = obj["message"]
            session.messages.append(msg)
            for blk in msg.get("content", []):
                if blk.get("type") == "tool_result":
                    tr = {
                        "tool_use_id": blk["tool_use_id"],
                        "content": blk["content"],
                        "is_error": blk.get("is_error", False),
                    }
                    chunk.tool_result = tr
                    session.tool_results.append(tr)
                    await _maybe_await(on_tool_result, tr)
                    if request.verbose_output:
                        _pp_tool_result(tr, theme)
            yield chunk

        # ------------------------ RESULT -----------------------------------
        elif typ == "result":
            session.result = obj.get("result", "").strip()
            session.usage = obj.get("usage", {})
            session.total_cost_usd = obj.get("total_cost_usd")
            session.num_turns = obj.get("num_turns")
            session.duration_ms = obj.get("duration_ms")
            session.duration_api_ms = obj.get("duration_api_ms")
            session.is_error = obj.get("is_error", False)

        # ------------------------ DONE -------------------------------------
        elif typ == "done":
            break

    # final pretty print
    await _maybe_await(on_final, session)
    if request.verbose_output:
        _pp_final(session, theme)

    yield session
