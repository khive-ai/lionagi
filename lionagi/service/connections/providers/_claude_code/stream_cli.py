# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import codecs
import contextlib
import dataclasses
import json
import logging
import shutil
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from functools import partial
from textwrap import shorten
from typing import Any

from json_repair import repair_json

from lionagi.libs.schema.as_readable import as_readable
from lionagi.utils import is_coro_func

from .models import ClaudeCodeRequest

CLAUDE = shutil.which("claude") or "claude"
if not shutil.which(CLAUDE):
    raise RuntimeError(
        "Claude CLI binary not found (npm i -g @anthropic-ai/claude-code)"
    )
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("claude-cli")


@dataclasses.dataclass
class ClaudeChunk:
    """Low-level wrapper around every NDJSON object coming from the CLI."""

    raw: dict[str, Any]
    type: str
    # convenience views
    thinking: str | None = None
    text: str | None = None
    tool_use: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None


@dataclasses.dataclass
class ClaudeSession:
    """Aggregated view of a whole CLI conversation."""

    session_id: str | None = None
    model: str | None = None

    # chronological log
    chunks: list[ClaudeChunk] = dataclasses.field(default_factory=list)

    # materialised views
    thinking_log: list[str] = dataclasses.field(default_factory=list)
    messages: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_uses: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_results: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )

    # final summary
    result: str = ""
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    total_cost_usd: float | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    is_error: bool = False


# --------------------------------------------------------------------------- helpers


async def ndjson_from_cli(request: ClaudeCodeRequest):
    """
    Yields each JSON object emitted by the *claude-code* CLI.

    • Robust against UTF-8 splits across chunks (incremental decoder).
    • Robust against braces inside strings (uses json.JSONDecoder.raw_decode)
    • Falls back to `json_repair.repair_json` when necessary.
    """
    workspace = request.cwd()
    workspace.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        CLAUDE,
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


# --------------------------------------------------------------------------- SSE route
async def stream_events(request: ClaudeCodeRequest):
    async for obj in ndjson_from_cli(request):
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
        f"### ✅ Session complete - {datetime.utcnow().isoformat(timespec='seconds')} UTC\n"
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
    stream = ndjson_from_cli(request)
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


__all__ = (
    "CLAUDE",
    "stream_claude_code_cli",
    "ndjson_from_cli",
    "ClaudeChunk",
    "ClaudeSession",
)
