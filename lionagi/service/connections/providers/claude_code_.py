# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

from claude_code_sdk import ClaudeCodeOptions, PermissionMode
from pydantic import BaseModel, Field

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict


class ClaudeCodeRequest(BaseModel):
    prompt: str = Field(description="The prompt for Claude Code")
    allowed_tools: list[str] = Field(
        default_factory=list, description="List of allowed tools"
    )
    max_thinking_tokens: int = 8000
    mcp_tools: list[str] = list
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    disallowed_tools: list[str] = Field(default_factory=list)
    model: str | None = None
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None

    def as_claude_options(self) -> ClaudeCodeOptions:
        dict_ = self.model_dump(exclude_unset=True)
        dict_.pop("prompt")
        return ClaudeCodeOptions(**dict_)

    @classmethod
    def create(
        cls,
        messages: list[dict],
        resume: str | None = None,
        continue_conversation: bool = None,
        **kwargs,
    ):
        prompt = messages[-1]["content"]
        if isinstance(prompt, dict | list):
            prompt = json.dumps(prompt)

        # If resume is provided, set continue_conversation to True
        if resume is not None and continue_conversation is None:
            continue_conversation = True

        dict_ = dict(
            prompt=prompt,
            continue_conversation=continue_conversation,
            resume=resume,
        )

        if resume is not None or continue_conversation is not None:
            if messages[0]["role"] == "system":
                dict_["system_prompt"] = messages[0]["content"]

        if (a := kwargs.get("system_prompt")) is not None:
            dict_["append_system_prompt"] = a

        if (a := kwargs.get("append_system_prompt")) is not None:
            dict_.setdefault("append_system_prompt", "")
            dict_["append_system_prompt"] += str(a)

        dict_ = {**dict_, **kwargs}
        dict_ = {k: v for k, v in dict_.items() if v is not None}
        return cls(**dict_)


ENDPOINT_CONFIG = EndpointConfig(
    name="claude_code",
    provider="anthropic",
    base_url="internal",
    endpoint="query",
    api_key="dummy",
    request_options=ClaudeCodeRequest,
    timeout=3000,
)


class ClaudeCodeEndpoint(Endpoint):
    def __init__(self, config=ENDPOINT_CONFIG, **kwargs):
        super().__init__(config=config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        **kwargs,
    ):
        request_dict = to_dict(request)
        # Merge stored kwargs from config, then request, then additional kwargs
        request_dict = {**self.config.kwargs, **request_dict, **kwargs}
        messages = request_dict.pop("messages", None)

        resume = request_dict.pop("resume", None)
        continue_conversation = request_dict.pop("continue_conversation", None)

        request_obj = ClaudeCodeRequest.create(
            messages=messages,
            resume=resume,
            continue_conversation=continue_conversation,
            **{
                k: v
                for k, v in request_dict.items()
                if v is not None and k in ClaudeCodeRequest.model_fields
            },
        )
        request_options = request_obj.as_claude_options()
        payload = {
            "prompt": request_obj.prompt,
            "options": request_options,
        }
        return (payload, {})

    def _stream_claude_code(self, prompt: str, options: ClaudeCodeOptions):
        from claude_code_sdk import query

        return query(prompt=prompt, options=options)

    async def stream(
        self,
        request: dict | BaseModel,
        **kwargs,
    ):
        async for chunk in self._stream_claude_code(**request, **kwargs):
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
            "cost_usd": None,
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
                results["cost_usd"] = response.cost_usd
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

        # Parse the responses into a consistent format
        return self._parse_claude_code_response(responses)
