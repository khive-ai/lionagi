from __future__ import annotations

import asyncio
import codecs
import contextlib
import dataclasses
import json
import logging
import shutil
from collections.abc import Callable
from datetime import datetime
from functools import partial
from textwrap import shorten
from typing import Any, AsyncIterator, Dict, List, Optional

from json_repair import repair_json
from pydantic import BaseModel

from lionagi.libs.schema.as_readable import as_readable
from lionagi.service.connections.endpoint import Endpoint, EndpointConfig
from lionagi.utils import to_dict

from .claude_code_ import ClaudeCodeRequest

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

    raw: Dict[str, Any]
    type: str
    # convenience views
    thinking: Optional[str] = None
    text: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ClaudeSession:
    """Aggregated view of a whole CLI conversation."""

    session_id: Optional[str] = None
    model: Optional[str] = None

    # chronological log
    chunks: List[ClaudeChunk] = dataclasses.field(default_factory=list)

    # materialised views
    thinking_log: List[str] = dataclasses.field(default_factory=list)
    messages: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_uses: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_results: List[Dict[str, Any]] = dataclasses.field(
        default_factory=list
    )

    # final summary
    result: str = ""
    usage: Dict[str, Any] = dataclasses.field(default_factory=dict)
    total_cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    duration_api_ms: Optional[int] = None
    is_error: bool = False


# --------------------------------------------------------------------------- helpers


async def ndjson_from_cli(request: ClaudeCodeRequest):
    """
    Yields each JSON object emitted by the *claude-code* CLI.

    • Robust against UTF‑8 splits across chunks (incremental decoder).
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

        # 4) propagate non‑zero exit code
        if await proc.wait() != 0:
            err = (await proc.stderr.read()).decode().strip()
            raise RuntimeError(err or "CLI exited non‑zero")

    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        await proc.wait()


# --------------------------------------------------------------------------- SSE route
async def stream_events(request: ClaudeCodeRequest):
    async for obj in ndjson_from_cli(request):
        yield obj
    yield {"type": "done"}


print_readable = partial(as_readable, md=True, display_str=True, theme="light")


def _pp_system(sys_obj: Dict[str, Any]) -> None:
    txt = (
        f"◼️  **Claude Code Session**  \n"
        f"- id: `{sys_obj.get('session_id', '?')}`  \n"
        f"- model: `{sys_obj.get('model', '?')}`  \n"
        f"- tools: {', '.join(sys_obj.get('tools', [])[:8])}"
        + ("…" if len(sys_obj.get("tools", [])) > 8 else "")
    )
    print_readable(txt, border=False)


def _pp_thinking(thought: str) -> None:
    print_readable(f"🧠 {thought}", border=False)


def _pp_assistant_text(text: str) -> None:
    print_readable(text)


def _pp_tool_use(tu: Dict[str, Any]) -> None:
    preview = shorten(str(tu["input"]).replace("\n", " "), 120)
    body = f"🔧 Tool Use — {tu['name']}  \nid: {tu['id']}  \ninput: {preview}"
    print_readable(body, border=False)


def _pp_tool_result(tr: Dict[str, Any]) -> None:
    body_preview = shorten(str(tr["content"]).replace("\n", " "), 120)
    status = "ERR" if tr.get("is_error") else "OK"
    body = (
        f"📄 Tool Result ({status}) — for {tr['tool_use_id']}  \n"
        f"content: `{body_preview}`"
    )
    print_readable(body, border=False)


def _pp_final(sess: ClaudeSession) -> None:
    usage = sess.usage or {}
    txt = (
        f"### ✅ Session complete – {datetime.utcnow().isoformat(timespec='seconds')} UTC\n"
        f"**Result (truncated):**\n\n{sess.result[:800]}{'…' if len(sess.result) > 800 else ''}\n\n"
        f"- cost: **${sess.total_cost_usd:.4f}**  \n"
        f"- turns: **{sess.num_turns}**  \n"
        f"- duration: **{sess.duration_ms} ms** (API {sess.duration_api_ms} ms)  \n"
        f"- tokens in/out: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}"
    )
    print_readable(txt)


# --------------------------------------------------------------------------- internal utils


async def _maybe_await(func, *args, **kw):
    """Call func which may be sync or async."""
    res = func(*args, **kw) if func else None
    if asyncio.iscoroutine(res):
        await res


# --------------------------------------------------------------------------- main parser


async def stream_claude_code_cli(  # noqa: C901  (complexity from branching is fine here)
    request: ClaudeCodeRequest,
    session: ClaudeSession = ClaudeSession(),
    *,
    on_system: Callable[[Dict[str, Any]], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_tool_use: Callable[[Dict[str, Any]], None] | None = None,
    on_tool_result: Callable[[Dict[str, Any]], None] | None = None,
    on_final: Callable[[ClaudeSession], None] | None = None,
) -> AsyncIterator[ClaudeChunk | dict | ClaudeSession]:
    """
    Consume the ND‑JSON stream produced by ndjson_from_cli()
    and return a fully‑populated ClaudeSession.

    If callbacks are omitted a default pretty‑print is emitted.
    """
    stream = ndjson_from_cli(request)

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
            if request.verbose_output and on_system is None:
                _pp_system(data)
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
                    if request.verbose_output and on_thinking is None:
                        _pp_thinking(thought)

                elif btype == "text":
                    text = blk.get("text", "")
                    chunk.text = text
                    await _maybe_await(on_text, text)
                    if request.verbose_output and on_text is None:
                        _pp_assistant_text(text)

                elif btype == "tool_use":
                    tu = {
                        "id": blk["id"],
                        "name": blk["name"],
                        "input": blk["input"],
                    }
                    chunk.tool_use = tu
                    session.tool_uses.append(tu)
                    await _maybe_await(on_tool_use, tu)
                    if request.verbose_output and on_tool_use is None:
                        _pp_tool_use(tu)

                elif btype == "tool_result":
                    tr = {
                        "tool_use_id": blk["tool_use_id"],
                        "content": blk["content"],
                        "is_error": blk.get("is_error", False),
                    }
                    chunk.tool_result = tr
                    session.tool_results.append(tr)
                    await _maybe_await(on_tool_result, tr)
                    if request.verbose_output and on_tool_result is None:
                        _pp_tool_result(tr)
            yield chunk

        # ------------------------ USER (tool_result containers) ------------
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
                    if request.verbose_output and on_tool_result is None:
                        _pp_tool_result(tr)
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
    if request.verbose_output and on_final is None:
        _pp_final(session)

    yield session


ENDPOINT_CONFIG = EndpointConfig(
    name="claude_code_cli",
    provider="claude_code",
    base_url="internal",
    endpoint="query_cli",
    api_key="dummy",
    request_options=ClaudeCodeRequest,
    timeout=12000,  # 20 mins
)


class ClaudeCodeCLIEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = ENDPOINT_CONFIG, **kwargs):
        super().__init__(config=config, **kwargs)

    def create_payload(self, request: dict | BaseModel, **kwargs):
        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages")
        req_obj = ClaudeCodeRequest.create(messages=messages, **req_dict)
        return {"request": req_obj}, {}

    async def stream(self, request: dict | BaseModel, **kwargs):
        payload, _ = self.create_payload(request, **kwargs)["request"]
        async for chunk in stream_claude_code_cli(payload):
            yield chunk

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ):
        responses = []
        request: ClaudeCodeRequest = payload["request"]
        session: ClaudeSession = ClaudeSession()
        system: dict = None

        # 1. stream the Claude Code response
        async for chunk in stream_claude_code_cli(request, session, **kwargs):
            if isinstance(chunk, dict):
                system = chunk
            responses.append(chunk)

        if request.auto_finish and not isinstance(
            responses[-1], ClaudeSession
        ):
            req2 = request.model_copy(deep=True)
            req2.max_turns = 1
            req2.continue_conversation = True
            if system:
                req2.resume = system.get("session_id") if system else None

            async for chunk in stream_claude_code_cli(req2, session, **kwargs):
                responses.append(chunk)
                if isinstance(chunk, ClaudeSession):
                    break
        print(
            f"Session {session.session_id} finished with {len(responses)} chunks"
        )

        return {
            "session_id": session.session_id,
            "model": session.model or "claude-code",
            "result": session.result,
            "tool_uses": session.tool_uses,
            "tool_results": session.tool_results,
            "is_error": session.is_error,
            "num_turns": session.num_turns,
            "total_cost_usd": session.total_cost_usd,
            "usage": session.usage,
            "chunks": [dataclasses.asdict(c) for c in session.chunks],
        }
