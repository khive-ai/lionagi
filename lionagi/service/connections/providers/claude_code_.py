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
        result_message = None
        model = "claude-code"
        assistant_text_content = []
        tool_results = []

        # Process all messages
        for response in responses:
            class_name = response.__class__.__name__

            if class_name == "SystemMessage" and hasattr(response, "data"):
                model = response.data.get("model", "claude-code")

            elif class_name == "AssistantMessage":
                # Extract text content from assistant messages
                if hasattr(response, "content") and response.content:
                    for block in response.content:
                        if hasattr(block, "text"):
                            assistant_text_content.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            assistant_text_content.append(block["text"])

            elif class_name == "UserMessage":
                # Extract tool results from user messages
                if hasattr(response, "content") and response.content:
                    for item in response.content:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "tool_result"
                        ):
                            tool_results.append(item.get("content", ""))

            elif class_name == "ResultMessage":
                result_message = response

        # Determine the final content
        final_content = ""
        if (
            result_message
            and hasattr(result_message, "result")
            and result_message.result
        ):
            # Use ResultMessage.result if available
            final_content = result_message.result
        elif assistant_text_content:
            # Use assistant text content if available
            final_content = "\n".join(assistant_text_content)
        elif tool_results:
            # If only tool results are available, use a generic summary
            # (Claude Code typically provides its own summary after tool use)
            final_content = (
                "I've completed the requested task using the available tools."
            )

        # Build the clean chat completions response
        result = {
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_content},
                    "finish_reason": (
                        "stop"
                        if not (
                            result_message
                            and hasattr(result_message, "is_error")
                            and result_message.is_error
                        )
                        else "error"
                    ),
                }
            ],
        }

        # Add usage information if available
        if result_message and hasattr(result_message, "usage"):
            result["usage"] = result_message.usage

        # Add only essential Claude Code metadata
        if result_message:
            if hasattr(result_message, "cost_usd"):
                result["usage"]["cost_usd"] = result_message.cost_usd
            if hasattr(result_message, "session_id"):
                result["session_id"] = result_message.session_id
            if hasattr(result_message, "is_error"):
                result["is_error"] = result_message.is_error
            if hasattr(result_message, "num_turns"):
                result["num_turns"] = result_message.num_turns

        return result

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
