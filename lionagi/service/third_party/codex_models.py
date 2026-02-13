# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import codecs
import contextlib
import inspect
import json
import logging
import shutil
import warnings
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from dataclasses import field as datafield
from pathlib import Path
from textwrap import shorten
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from lionagi import ln

HAS_CODEX_CLI = False
CODEX_CLI = None

if (c := (shutil.which("codex") or "codex")) and shutil.which(c):
    HAS_CODEX_CLI = True
    CODEX_CLI = c

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("codex-cli")

# --------------------------------------------------------------------------- constants
CodexSandboxMode = Literal[
    "read-only",
    "workspace-write",
    "danger-full-access",
]

__all__ = (
    "CodexChunk",
    "CodexCodeRequest",
    "CodexSession",
    "stream_codex_cli",
)


# --------------------------------------------------------------------------- request model
class CodexCodeRequest(BaseModel):
    """Request model for OpenAI Codex CLI execution."""

    # -- conversational bits -------------------------------------------------
    prompt: str = Field(description="The prompt for Codex CLI")
    system_prompt: str | None = None

    # -- repo / workspace ----------------------------------------------------
    repo: Path = Field(default_factory=Path.cwd, exclude=True)
    ws: str | None = None  # sub-directory under repo

    # -- runtime & safety ----------------------------------------------------
    model: str | None = Field(
        default="gpt-5.3-codex",
        description="Codex model to use (gpt-5.3-codex, o3, etc.)",
    )
    full_auto: bool = Field(
        default=False,
        description="Auto-approve with workspace-write sandbox (--full-auto)",
    )
    sandbox: CodexSandboxMode | None = Field(
        default=None,
        description="Sandbox mode: read-only, workspace-write, danger-full-access",
    )
    bypass_approvals: bool = Field(
        default=False,
        description="Skip all approvals and sandbox (--dangerously-bypass-approvals-and-sandbox)",
    )
    skip_git_repo_check: bool = Field(
        default=False,
        description="Allow running outside a git repository",
    )
    output_schema: str | Path | None = Field(
        default=None,
        description="Path to JSON Schema file for structured output",
    )
    include_plan_tool: bool = Field(
        default=False,
        description="Include the plan tool in the conversation",
    )
    images: list[str] = Field(
        default_factory=list,
        description="Image file paths to attach to the prompt",
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Config overrides as key=value pairs (-c flag)",
    )

    # -- internal use --------------------------------------------------------
    verbose_output: bool = Field(default=False)
    cli_include_summary: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def _validate_message_prompt(cls, data):
        """Convert messages format to prompt if needed."""
        if data.get("prompt"):
            return data

        if not (msg := data.get("messages")):
            raise ValueError("messages or prompt required")

        prompts = []
        for message in msg:
            if message["role"] != "system":
                content = message["content"]
                if isinstance(content, (dict, list)):
                    prompts.append(ln.json_dumps(content))
                else:
                    prompts.append(content)
            elif message["role"] == "system" and not data.get("system_prompt"):
                data["system_prompt"] = message["content"]

        data["prompt"] = "\n".join(prompts)
        return data

    @model_validator(mode="after")
    def _warn_dangerous_settings(self):
        """Emit security warnings for dangerous CLI settings."""
        if self.bypass_approvals:
            warnings.warn(
                "CodexCodeRequest: bypass_approvals=True skips ALL approval prompts "
                "and disables sandboxing. EXTREMELY DANGEROUS. Only use in "
                "externally sandboxed environments.",
                UserWarning,
                stacklevel=4,
            )
        return self

    def cwd(self) -> Path:
        """Get working directory, validating workspace path."""
        if not self.ws:
            return self.repo

        ws_path = Path(self.ws)

        if ws_path.is_absolute():
            raise ValueError(f"Workspace path must be relative, got absolute: {self.ws}")

        if ".." in ws_path.parts:
            raise ValueError(f"Directory traversal detected in workspace path: {self.ws}")

        repo_resolved = self.repo.resolve()
        result = (self.repo / ws_path).resolve()

        try:
            result.relative_to(repo_resolved)
        except ValueError:
            raise ValueError(
                f"Workspace path escapes repository bounds. "
                f"Repository: {repo_resolved}, Workspace: {result}"
            )

        return result

    def as_cmd_args(self) -> list[str]:
        """Build argument list for `codex exec` subcommand."""
        args: list[str] = ["exec", "--json", self.prompt]

        if self.model:
            args += ["-m", self.model]

        workspace = self.cwd()
        args += ["-C", str(workspace)]

        if self.bypass_approvals:
            args.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.full_auto:
            args.append("--full-auto")
        elif self.sandbox:
            args += ["-s", self.sandbox]

        if self.skip_git_repo_check:
            args.append("--skip-git-repo-check")

        if self.output_schema:
            args += ["--output-schema", str(self.output_schema)]

        if self.include_plan_tool:
            args.append("--include-plan-tool")

        for image in self.images:
            args += ["-i", image]

        for key, value in self.config_overrides.items():
            args += [
                "-c",
                f"{key}={json.dumps(value) if not isinstance(value, str) else value}",
            ]

        return args


@dataclass
class CodexChunk:
    """Low-level wrapper around every JSON object from the Codex CLI."""

    raw: dict[str, Any]
    type: str
    # convenience views
    text: str | None = None
    tool_use: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None


@dataclass
class CodexSession:
    """Aggregated view of a whole Codex CLI conversation."""

    session_id: str | None = None
    model: str | None = None

    # chronological log
    chunks: list[CodexChunk] = datafield(default_factory=list)

    # materialized views
    messages: list[dict[str, Any]] = datafield(default_factory=list)
    tool_uses: list[dict[str, Any]] = datafield(default_factory=list)
    tool_results: list[dict[str, Any]] = datafield(default_factory=list)

    # final summary
    result: str = ""
    usage: dict[str, Any] = datafield(default_factory=dict)
    total_cost_usd: float | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    is_error: bool = False
    summary: dict | None = None

    def populate_summary(self) -> None:
        self.summary = _extract_summary(self)


def _extract_summary(session: CodexSession) -> dict[str, Any]:
    """Extract summary from session data."""
    tool_counts: dict[str, int] = {}
    tool_details: list[dict[str, Any]] = []
    file_operations: dict[str, list[str]] = {
        "reads": [],
        "writes": [],
        "edits": [],
    }
    key_actions: list[str] = []

    for tool_use in session.tool_uses:
        tool_name = tool_use.get("name", "unknown")
        tool_input = tool_use.get("input", {})
        tool_id = tool_use.get("id", "")

        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        tool_details.append({"tool": tool_name, "id": tool_id, "input": tool_input})

        if tool_name in ("read_file", "Read", "read"):
            file_path = tool_input.get("path", tool_input.get("file_path", "unknown"))
            file_operations["reads"].append(file_path)
            key_actions.append(f"Read {file_path}")

        elif tool_name in ("write_file", "create_file", "Write", "write"):
            file_path = tool_input.get("path", tool_input.get("file_path", "unknown"))
            file_operations["writes"].append(file_path)
            key_actions.append(f"Wrote {file_path}")

        elif tool_name in ("edit_file", "patch", "Edit", "edit"):
            file_path = tool_input.get("path", tool_input.get("file_path", "unknown"))
            file_operations["edits"].append(file_path)
            key_actions.append(f"Edited {file_path}")

        elif tool_name in (
            "shell",
            "terminal",
            "run_shell_command",
            "Bash",
            "bash",
        ):
            command = tool_input.get("command", tool_input.get("cmd", ""))
            command_summary = command[:50] + "..." if len(command) > 50 else command
            key_actions.append(f"Ran: {command_summary}")

        elif tool_name.startswith("mcp_") or tool_name.startswith("mcp__"):
            operation = tool_name.replace("mcp__", "").replace("mcp_", "")
            key_actions.append(f"MCP {operation}")

        else:
            key_actions.append(f"Used {tool_name}")

    key_actions = list(dict.fromkeys(key_actions)) if key_actions else ["No specific actions"]

    for op_type in file_operations:
        file_operations[op_type] = list(dict.fromkeys(file_operations[op_type]))

    result_summary = (session.result[:200] + "...") if len(session.result) > 200 else session.result

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
            **session.usage,
        },
    }


async def _ndjson_from_cli(request: CodexCodeRequest):
    """
    Yields each JSON object emitted by the Codex CLI (JSONL mode).

    Robust against UTF-8 splits and uses json.JSONDecoder.raw_decode.
    """
    if CODEX_CLI is None:
        raise RuntimeError("Codex CLI not found. Install with: npm i -g @openai/codex")

    proc = await asyncio.create_subprocess_exec(
        CODEX_CLI,
        *request.as_cmd_args(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    decoder = codecs.getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    buffer: str = ""

    if proc.stdout is None:
        raise RuntimeError("Failed to capture stdout from Codex CLI")

    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break

            buffer += decoder.decode(chunk)

            while buffer:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                try:
                    obj, idx = json_decoder.raw_decode(buffer)
                    yield obj
                    buffer = buffer[idx:]
                except json.JSONDecodeError:
                    break

        buffer += decoder.decode(b"", final=True)
        buffer = buffer.strip()
        if buffer:
            try:
                obj, idx = json_decoder.raw_decode(buffer)
                yield obj
            except json.JSONDecodeError:
                log.error("Skipped unrecoverable JSON tail: %.120s...", buffer)

        if await proc.wait() != 0:
            err = ""
            if proc.stderr is not None:
                err = (await proc.stderr.read()).decode().strip()
            raise RuntimeError(err or "Codex CLI exited non-zero")

    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        await proc.wait()


async def stream_codex_cli_events(request: CodexCodeRequest):
    """Stream events from Codex CLI."""
    if not CODEX_CLI:
        raise RuntimeError("Codex CLI not found (npm i -g @openai/codex)")
    async for obj in _ndjson_from_cli(request):
        yield obj
    yield {"type": "done"}


async def _maybe_await(func, *args, **kw):
    """Call func which may be sync or async."""
    res = func(*args, **kw) if func else None
    if inspect.iscoroutine(res):
        await res


def _pp_text(text: str) -> None:
    print(f"\n> Codex:\n{text}\n")


def _pp_tool_use(tu: dict[str, Any]) -> None:
    preview = shorten(str(tu.get("input", {})).replace("\n", " "), 130)
    print(f"- Tool Use - {tu.get('name', 'unknown')}: {preview}")


def _pp_tool_result(tr: dict[str, Any]) -> None:
    body_preview = shorten(str(tr.get("content", "")).replace("\n", " "), 130)
    status = "ERR" if tr.get("is_error") else "OK"
    print(f"- Tool Result - {status}: {body_preview}")


def _pp_final(sess: CodexSession) -> None:
    usage = sess.usage or {}
    cost_str = f"${sess.total_cost_usd:.4f}" if sess.total_cost_usd else "N/A"
    print(
        f"\n### Codex Session complete\n"
        f"**Result:** {sess.result or ''}\n"
        f"- cost: {cost_str}\n"
        f"- turns: {sess.num_turns}\n"
        f"- duration: {sess.duration_ms} ms\n"
        f"- tokens: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}"
    )


async def stream_codex_cli(
    request: CodexCodeRequest,
    session: CodexSession | None = None,
    *,
    on_text: Callable[[str], None] | None = None,
    on_tool_use: Callable[[dict[str, Any]], None] | None = None,
    on_tool_result: Callable[[dict[str, Any]], None] | None = None,
    on_final: Callable[[CodexSession], None] | None = None,
) -> AsyncIterator[CodexChunk | dict | CodexSession]:
    """
    Consume the JSONL stream from Codex CLI and return a populated CodexSession.

    Handles flexible event type names since Codex CLI output format may vary.
    """
    if session is None:
        session = CodexSession()

    stream = stream_codex_cli_events(request)

    async for obj in stream:
        typ = obj.get("type", "unknown")
        chunk = CodexChunk(raw=obj, type=typ)
        session.chunks.append(chunk)

        # -- thread / session start ------------------------------------------
        if typ in ("thread.started", "system", "init", "session.start"):
            session.session_id = obj.get("thread_id", obj.get("session_id", obj.get("id")))
            session.model = obj.get("model")
            yield obj

        # -- item.completed (agent_message, reasoning, tool calls) -----------
        elif typ == "item.completed":
            item = obj.get("item", {})
            item_type = item.get("type", "")

            if item_type == "agent_message":
                text = item.get("text", "")
                chunk.text = text
                session.messages.append(item)
                await _maybe_await(on_text, text)
                if request.verbose_output:
                    _pp_text(text)
                yield chunk

            elif item_type in ("function_call", "tool_call"):
                tu = {
                    "id": item.get("id", item.get("call_id", "")),
                    "name": item.get("name", item.get("function", "")),
                    "input": item.get("arguments", item.get("input", item.get("args", {}))),
                }
                chunk.tool_use = tu
                session.tool_uses.append(tu)
                await _maybe_await(on_tool_use, tu)
                if request.verbose_output:
                    _pp_tool_use(tu)
                yield chunk

            elif item_type == "function_call_output":
                tr = {
                    "tool_use_id": item.get("call_id", item.get("id", "")),
                    "content": item.get("output", item.get("content", "")),
                    "is_error": item.get("is_error", False),
                }
                chunk.tool_result = tr
                session.tool_results.append(tr)
                await _maybe_await(on_tool_result, tr)
                if request.verbose_output:
                    _pp_tool_result(tr)
                yield chunk

            elif item_type == "reasoning":
                # reasoning traces — store but don't treat as result
                yield chunk

            else:
                yield chunk

        # -- turn.completed (usage stats) ------------------------------------
        elif typ == "turn.completed":
            session.usage = obj.get("usage", {})
            session.total_cost_usd = obj.get("total_cost_usd", obj.get("cost"))
            session.num_turns = (session.num_turns or 0) + 1

        # -- turn.failed / error ---------------------------------------------
        elif typ in ("turn.failed", "error"):
            session.is_error = True
            err = obj.get("error", {})
            session.result = (
                err.get("message", str(err))
                if isinstance(err, dict)
                else obj.get("message", str(err))
            )

        # -- legacy event types (older CLI versions) -------------------------
        elif typ in ("message", "assistant", "agent"):
            msg = obj.get("message", obj)
            session.messages.append(msg)

            content = msg.get("content", "")
            if isinstance(content, str):
                chunk.text = content
                await _maybe_await(on_text, content)
                if request.verbose_output:
                    _pp_text(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        btype = blk.get("type")
                        if btype == "text":
                            text = blk.get("text", "")
                            chunk.text = text
                            await _maybe_await(on_text, text)
                            if request.verbose_output:
                                _pp_text(text)
                        elif btype in ("tool_use", "function_call"):
                            tu = {
                                "id": blk.get("id", ""),
                                "name": blk.get(
                                    "name",
                                    blk.get("function", {}).get("name", ""),
                                ),
                                "input": blk.get("input", blk.get("arguments", {})),
                            }
                            chunk.tool_use = tu
                            session.tool_uses.append(tu)
                            await _maybe_await(on_tool_use, tu)
                            if request.verbose_output:
                                _pp_tool_use(tu)
            yield chunk

        elif typ in ("result", "response", "session.end"):
            session.result = obj.get("result", obj.get("response", obj.get("text", ""))).strip()
            session.usage = obj.get("usage", obj.get("stats", {}))
            session.total_cost_usd = obj.get("total_cost_usd", obj.get("cost"))
            session.num_turns = obj.get("num_turns", obj.get("turns"))
            session.duration_ms = obj.get("duration_ms", obj.get("duration"))
            session.is_error = obj.get("is_error", obj.get("error") is not None)

        elif typ == "done":
            break

    await _maybe_await(on_final, session)
    if request.verbose_output:
        _pp_final(session)

    yield session
