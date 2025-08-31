# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Claude Code CLI service implementation for v1 architecture."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from json_repair import repair_json

from lionagi import _err, ln

from ..core import CallContext, Service
from ..endpoint import RequestModel
from ..transport import SubprocessCLITransport

logger = logging.getLogger(__name__)

# Type definitions
ClaudePermission = Literal[
    "default",
    "acceptEdits",
    "bypassPermissions",
    "dangerously-skip-permissions",
]

# Check for Claude CLI availability
CLAUDE_CLI = shutil.which("claude")
HAS_CLAUDE_CODE_CLI = CLAUDE_CLI is not None

if not HAS_CLAUDE_CODE_CLI:
    logger.warning("Claude CLI not found. Install with: npm i -g @anthropic-ai/claude-code")


@dataclass
class ClaudeChunk:
    """Low-level wrapper around every NDJSON object from Claude CLI."""

    raw: dict[str, Any]
    type: str
    thinking: str | None = None
    text: str | None = None
    tool_use: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None


@dataclass
class ClaudeSession:
    """Aggregated view of a Claude CLI conversation."""

    session_id: str | None = None
    model: str | None = None
    chunks: list[ClaudeChunk] = field(default_factory=list)
    thinking_log: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    result: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    is_error: bool = False


class ClaudeCodeRequestModel(RequestModel):
    """Claude Code CLI request model extending v1 RequestModel."""

    # Override messages to be optional since Claude Code uses prompt
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None

    # Claude Code specific parameters
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    max_turns: int | None = None
    continue_conversation: bool = False
    resume: str | None = None

    # Workspace and security
    repo: str | None = None
    ws: str | None = None  # workspace subdirectory
    add_dir: str | None = None
    permission_mode: ClaudePermission | None = None
    permission_prompt_tool_name: str | None = None

    # Tools and capabilities
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    mcp_tools: list[str] | None = None
    mcp_servers: dict[str, Any] | None = None
    mcp_config: str | None = None

    # Runtime options
    auto_finish: bool = False
    verbose_output: bool = False
    cli_display_theme: Literal["light", "dark"] = "dark"
    cli_include_summary: bool = False

    def get_prompt(self) -> str:
        """Extract prompt from either prompt field or messages."""
        if self.prompt:
            return self.prompt

        if not self.messages:
            raise ValueError("Either prompt or messages must be provided")

        # Convert messages to prompt format
        if self.continue_conversation or self.resume:
            # Use only the last message for continuation
            return str(self.messages[-1].get("content", ""))
        else:
            # Combine all non-system messages
            prompts = []
            for msg in self.messages:
                if msg.get("role") != "system":
                    content = msg.get("content", "")
                    prompts.append(str(content))
            return "\n".join(prompts)

    def get_system_prompt(self) -> str | None:
        """Extract system prompt from messages or system_prompt field."""
        if self.system_prompt:
            return self.system_prompt

        if self.messages and self.messages[0].get("role") == "system":
            return str(self.messages[0].get("content", ""))

        return None

    def get_workspace_path(self, base_repo: Path) -> Path:
        """Get the workspace path, ensuring security."""
        if not self.ws:
            return base_repo

        ws_path = Path(self.ws)

        # Security checks
        if ws_path.is_absolute():
            raise ValueError(f"Workspace path must be relative: {self.ws}")

        if ".." in ws_path.parts:
            raise ValueError(f"Directory traversal detected: {self.ws}")

        # Resolve and validate bounds
        repo_resolved = base_repo.resolve()
        result = (base_repo / ws_path).resolve()

        try:
            result.relative_to(repo_resolved)
        except ValueError:
            raise ValueError(f"Workspace escapes repository bounds: {result}")

        return result

    def as_cli_args(self) -> list[str]:
        """Build Claude CLI command arguments."""
        prompt = self.get_prompt()
        args = ["-p", prompt, "--output-format", "stream-json"]

        if self.allowed_tools:
            args.append("--allowedTools")
            args.extend(f'"{tool}"' for tool in self.allowed_tools)

        if self.disallowed_tools:
            args.append("--disallowedTools")
            args.extend(f'"{tool}"' for tool in self.disallowed_tools)

        if self.resume:
            args.extend(["--resume", self.resume])
        elif self.continue_conversation:
            args.append("--continue")

        if self.max_turns:
            args.extend(["--max-turns", str(self.max_turns + 1)])

        if self.permission_mode == "bypassPermissions":
            args.append("--dangerously-skip-permissions")

        if self.add_dir:
            args.extend(["--add-dir", self.add_dir])

        if self.permission_prompt_tool_name:
            args.extend(["--permission-prompt-tool", self.permission_prompt_tool_name])

        if self.mcp_config:
            args.extend(["--mcp-config", f'"{self.mcp_config}"'])

        model = self.model or "sonnet"
        args.extend(["--model", model, "--verbose"])

        return args


class ClaudeCodeCLIService(Service):
    """Claude Code CLI service for v1 architecture."""

    name = "claude_code_cli"
    requires = frozenset({"exec:claude", "fs.read", "fs.write"})

    def __init__(self, base_repo: str | Path | None = None, *, max_concurrent: int = 5):
        """Initialize Claude Code CLI service.

        Args:
            base_repo: Base repository path for Claude Code operations
            max_concurrent: Maximum concurrent CLI operations
        """
        if not HAS_CLAUDE_CODE_CLI:
            raise _err.ServiceError(
                "Claude CLI not found. Install with: npm i -g @anthropic-ai/claude-code",
                context={"service": self.name},
            )

        self.base_repo = Path(base_repo or Path.cwd())
        self._transport = SubprocessCLITransport(max_concurrent=max_concurrent)

    async def call(self, req: ClaudeCodeRequestModel, *, ctx: CallContext) -> dict[str, Any]:
        """Execute Claude Code CLI call and return session data."""
        session = ClaudeSession()

        # Collect all chunks
        chunks = []
        async for chunk in self.stream(req, ctx=ctx):
            chunks.append(chunk)
            if isinstance(chunk, dict) and chunk.get("type") == "result":
                # Extract session data from result
                session.result = chunk.get("result", "")
                session.usage = chunk.get("usage", {})
                session.total_cost_usd = chunk.get("total_cost_usd")
                session.num_turns = chunk.get("num_turns")
                session.duration_ms = chunk.get("duration_ms")
                session.duration_api_ms = chunk.get("duration_api_ms")
                session.is_error = chunk.get("is_error", False)

        # Combine text chunks for final result
        text_parts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                if chunk.get("type") == "assistant":
                    message = chunk.get("message", {})
                    for block in message.get("content", []):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))

        if text_parts:
            session.result = "\n".join(text_parts) + "\n" + session.result

        return {
            "session_id": session.session_id,
            "model": session.model,
            "result": session.result,
            "usage": session.usage,
            "total_cost_usd": session.total_cost_usd,
            "num_turns": session.num_turns,
            "duration_ms": session.duration_ms,
            "is_error": session.is_error,
        }

    async def stream(
        self, req: ClaudeCodeRequestModel, *, ctx: CallContext
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute streaming Claude Code CLI call."""

        async def do_stream() -> AsyncIterator[dict[str, Any]]:
            """Core streaming operation with Claude CLI via transport layer."""
            workspace = req.get_workspace_path(self.base_repo)
            workspace.mkdir(parents=True, exist_ok=True)

            # Build command for transport
            command = [CLAUDE_CLI] + req.as_cli_args()

            # Calculate timeout from context
            timeout_s = None
            if ctx.deadline_s is not None:
                import anyio

                timeout_s = max(0.1, ctx.deadline_s - anyio.current_time())

            # Stream through transport layer with JSON parsing
            json_decoder = json.JSONDecoder()
            buffer = ""

            try:
                async for line in self._transport.stream(
                    command,
                    cwd=workspace,
                    timeout_s=timeout_s,
                ):
                    # Accumulate line into buffer for JSON parsing
                    buffer += line + "\n"

                    # Parse complete JSON objects
                    while buffer:
                        buffer = buffer.lstrip()
                        if not buffer:
                            break

                        try:
                            obj, idx = json_decoder.raw_decode(buffer)
                            yield obj
                            buffer = buffer[idx:]
                        except json.JSONDecodeError:
                            # Incomplete JSON, need more data
                            break

                # Handle remaining buffer
                buffer = buffer.strip()
                if buffer:
                    try:
                        obj, _ = json_decoder.raw_decode(buffer)
                        yield obj
                    except json.JSONDecodeError:
                        try:
                            # Attempt repair for malformed JSON
                            fixed = repair_json(buffer)
                            yield json.loads(fixed)
                            logger.warning("Repaired malformed JSON at stream end")
                        except Exception:
                            logger.error("Failed to parse JSON tail: %s", buffer[:120])

            except _err.ServiceError as e:
                # Re-raise service errors from transport with context
                raise _err.ServiceError(
                    f"Claude CLI execution failed: {e}",
                    context={
                        "call_id": str(ctx.call_id),
                        "service": self.name,
                        "command": command[0],  # Just the command name for security
                    },
                    cause=e,
                )

        # Apply deadline enforcement if specified
        if ctx.deadline_s is None:
            async for chunk in do_stream():
                yield chunk
        else:
            with ln.fail_at(ctx.deadline_s):
                async for chunk in do_stream():
                    yield chunk

    async def close(self) -> None:
        """Cleanup transport resources."""
        await self._transport.close()


# Factory function
def create_claude_code_service(
    base_repo: str | Path | None = None,
    *,
    max_concurrent: int = 5,
) -> ClaudeCodeCLIService:
    """Create Claude Code CLI service instance."""
    return ClaudeCodeCLIService(base_repo=base_repo, max_concurrent=max_concurrent)
