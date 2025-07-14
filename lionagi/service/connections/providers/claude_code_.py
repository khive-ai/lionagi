# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from claude_code_sdk import ClaudeCodeOptions
from claude_code_sdk import query as sdk_query
from pydantic import BaseModel, Field, field_validator, model_validator

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict

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


# --------------------------------------------------------------------------- request model
class ClaudeCodeRequest(BaseModel):
    # -- conversational bits -------------------------------------------------
    prompt: str = Field(description="The prompt for Claude Code")
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
    permission_mode: ClaudePermission | None = None
    permission_prompt_tool_name: str | None = None
    disallowed_tools: list[str] = Field(default_factory=list)

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
        full_prompt = f"Human User: {self.prompt}\n\nAssistant:"
        args: list[str] = ["-p", full_prompt, "--output-format", "stream-json"]
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

        args += ["--model", self.model or "sonnet", "--verbose"]
        return args

    # ------------------------ SDK helpers -----------------------------------
    def as_claude_options(self) -> ClaudeCodeOptions:
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

        prompt = messages[-1]["content"]
        if isinstance(prompt, (dict, list)):
            prompt = json.dumps(prompt)

        if resume and continue_conversation is None:
            continue_conversation = True

        data: dict[str, Any] = dict(
            prompt=prompt,
            resume=resume,
            continue_conversation=bool(continue_conversation),
        )

        if (messages[0]["role"] == "system") and (
            resume or continue_conversation
        ):
            data["system_prompt"] = messages[0]["content"]

        # Merge optional system prompts
        if kwargs.get("system_prompt"):
            data["append_system_prompt"] = kwargs.pop("system_prompt")

        data.update(kwargs)
        return cls.model_validate(data, strict=False)


# --------------------------------------------------------------------------- SDK endpoint
ENDPOINT_CONFIG = EndpointConfig(
    name="claude_code",
    provider="claude_code",
    base_url="internal",
    endpoint="query",
    api_key="dummy",
    request_options=ClaudeCodeRequest,
    timeout=3000,
)


class ClaudeCodeEndpoint(Endpoint):
    """Direct Python-SDK (non-CLI) endpoint - unchanged except for bug-fixes."""

    def __init__(self, config: EndpointConfig = ENDPOINT_CONFIG, **kwargs):
        super().__init__(config=config, **kwargs)

    def create_payload(self, request: dict | BaseModel, **kwargs):
        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages")
        req_obj = ClaudeCodeRequest.create(messages=messages, **req_dict)
        return {"request": req_obj}, {}

    def _stream_claude_code(self, request: ClaudeCodeRequest):
        return sdk_query(
            prompt=request.prompt, options=request.as_claude_options()
        )

    async def stream(self, request: dict | BaseModel, **kwargs):
        payload = self.create_payload(request, **kwargs)["request"]
        async for chunk in self._stream_claude_code(payload):
            yield chunk

    def _parse_claude_code_response(self, responses: list) -> dict:
        """Parse Claude Code responses into a clean chat completions-like format.

        Claude Code returns a list of messages:
        - SystemMessage: initialization info
        - AssistantMessage(s): actual assistant responses with content blocks
        - UserMessage(s): for tool use interactions
        - ResultMessage: final result with metadata

        When Claude Code uses tools, the ResultMessage.result may be None.
        In that case, we need to look at the tool results in UserMessages.
        """
        results = {
            "session_id": None,
            "model": "claude-code",
            "result": "",
            "tool_results": [],
            "is_error": False,
            "num_turns": None,
            "total_cost_usd": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        from claude_code_sdk import types

        for response in responses:
            if isinstance(response, types.SystemMessage):
                results["session_id"] = response.data.get("session_id")
                results["model"] = response.data.get("model", "claude-code")
            if isinstance(response, types.AssistantMessage):
                for block in response.content:
                    if isinstance(block, types.TextBlock):
                        results["result"] += block.text.strip() + "\n"
                    if isinstance(block, types.ToolResultBlock):
                        results["tool_results"].append(
                            {
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                        )
            if isinstance(response, types.ResultMessage):
                results["result"] += response.result.strip() or ""
                results["usage"] = response.usage
                results["is_error"] = response.is_error
                results["total_cost_usd"] = response.total_cost_usd
                results["num_turns"] = response.num_turns
                results["duration_ms"] = response.duration_ms
                results["duration_api_ms"] = response.duration_api_ms

        return results

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ):
        responses = []
        async for chunk in self._stream_claude_code(**payload):
            responses.append(chunk)

        return self._parse_claude_code_response(responses)
