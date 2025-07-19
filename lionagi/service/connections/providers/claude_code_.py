# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from claude_code_sdk import ClaudeCodeOptions
from claude_code_sdk import query as sdk_query
from claude_code_sdk import types as cc_types
from pydantic import BaseModel, Field, field_validator, model_validator

from lionagi.libs.schema.as_readable import as_readable
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict, to_list

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

    # -- internal use --------------------------------------------------------
    auto_finish: bool = Field(
        default=True,
        exclude=True,
        description="Automatically finish the conversation after the first response",
    )
    verbose_output: bool = Field(default=False, exclude=True)

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

        prompt = ""

        # 1. if resume or continue_conversation, use the last message
        if resume or continue_conversation:
            continue_conversation = True
            prompt = messages[-1]["content"]
            if isinstance(prompt, (dict, list)):
                prompt = json.dumps(prompt)

        # 2. else, use entire messages except system message
        else:
            prompts = []
            continue_conversation = False
            for message in messages:
                if message["role"] != "system":
                    content = message["content"]
                    prompts.append(
                        json.dumps(content)
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
        payload, _ = self.create_payload(request, **kwargs)["request"]
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
            "tool_uses": [],
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

        for response in responses:
            if isinstance(response, cc_types.SystemMessage):
                results["session_id"] = response.data.get("session_id")
                results["model"] = response.data.get("model", "claude-code")
            if isinstance(
                response, cc_types.AssistantMessage | cc_types.UserMessage
            ):
                for block in to_list(
                    response.content,
                    flatten=True,
                    flatten_tuple_set=True,
                    dropna=True,
                ):
                    if isinstance(block, cc_types.TextBlock):
                        results["result"] += block.text.strip() + "\n"

                    if isinstance(block, cc_types.ToolUseBlock):
                        entry = {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        results["tool_uses"].append(entry)

                    if isinstance(block, cc_types.ToolResultBlock):
                        results["tool_results"].append(
                            {
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                        )

            if isinstance(response, cc_types.ResultMessage):
                if response.result:
                    results["result"] = str(response.result).strip()
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
        request: ClaudeCodeRequest = payload["request"]
        system: cc_types.SystemMessage = None

        # 1. stream the Claude Code response
        async for chunk in self._stream_claude_code(**payload):
            if request.verbose_output:
                _display_message(chunk)

            if isinstance(chunk, cc_types.SystemMessage):
                system = chunk
            responses.append(chunk)

        # 2. If the last response is not a ResultMessage and auto_finish is True,
        #   we need to query Claude Code again to get the final result message.
        if request.auto_finish and not isinstance(
            responses[-1], cc_types.ResultMessage
        ):
            options = request.as_claude_options()
            options.continue_conversation = True
            options.max_turns = 1
            if system:
                options.resume = (
                    system.data.get("session_id", None) if system else None
                )

            async for chunk in sdk_query(
                prompt="Please provide a the final result message only",
                options=options,
            ):
                if isinstance(chunk, cc_types.ResultMessage):
                    if request.verbose_output:
                        str_ = _verbose_output(chunk)
                        if str_:
                            as_readable(
                                str_,
                                md=True,
                                display_str=True,
                                format_curly=True,
                                max_panel_width=100,
                                theme="light",
                            )

                    responses.append(chunk)

        # 3. Parse the responses into a clean format
        return self._parse_claude_code_response(responses)


def _display_message(chunk):
    if isinstance(
        chunk,
        cc_types.SystemMessage
        | cc_types.AssistantMessage
        | cc_types.UserMessage,
    ):
        str_ = _verbose_output(chunk)
        if str_:
            if str_.startswith("Claude:"):
                as_readable(
                    str_,
                    md=True,
                    display_str=True,
                    max_panel_width=100,
                    theme="light",
                )
            else:
                as_readable(
                    str_,
                    format_curly=True,
                    display_str=True,
                    max_panel_width=100,
                    theme="light",
                )

    if isinstance(chunk, cc_types.ResultMessage):
        str_ = _verbose_output(chunk)
        as_readable(
            str_,
            md=True,
            display_str=True,
            format_curly=True,
            max_panel_width=100,
            theme="light",
        )


def _verbose_output(res: cc_types.Message) -> str:
    str_ = ""
    if isinstance(res, cc_types.SystemMessage):
        str_ = f"Claude Code Session Started: {res.data.get('session_id', 'unknown')}"
        str_ += f"\nModel: {res.data.get('model', 'claude-code')}\n---"
        return str_

    if isinstance(res, cc_types.AssistantMessage | cc_types.UserMessage):
        for block in to_list(
            res.content, flatten=True, flatten_tuple_set=True, dropna=True
        ):
            if isinstance(block, cc_types.TextBlock):
                text = (
                    block.text.strip() if isinstance(block.text, str) else ""
                )
                str_ += f"Claude:\n{text}"

            if isinstance(block, cc_types.ToolUseBlock):
                input = (
                    json.dumps(block.input, indent=2)
                    if isinstance(block.input, dict)
                    else str(block.input)
                )
                input = input[:200] + "..." if len(input) > 200 else input
                str_ += (
                    f"Tool Use: {block.name} - {block.id}\n  - Input: {input}"
                )

            if isinstance(block, cc_types.ToolResultBlock):
                content = str(block.content)
                content = (
                    content[:200] + "..." if len(content) > 200 else content
                )
                str_ += (
                    f"Tool Result: {block.tool_use_id}\n  - Content: {content}"
                )
        return str_

    if isinstance(res, cc_types.ResultMessage):
        str_ += f"Session Completion - {res.session_id}"
        str_ += f"\nResult: {res.result or 'No result'}"
        str_ += f"\n- Cost: ${res.total_cost_usd:.4f} USD"
        str_ += f"\n- Duration: {res.duration_ms} ms (API: {res.duration_api_ms} ms)"
        str_ += f"\n- Turns: {res.num_turns}"
        return str_
