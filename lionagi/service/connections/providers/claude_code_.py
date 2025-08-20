# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

from pydantic import BaseModel

from lionagi import ln
from lionagi.libs.schema.as_readable import as_readable
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict, to_list

from ...third_party.claude_code import (
    CLAUDE_CODE_OPTION_PARAMS,
    HAS_CLAUDE_CODE_SDK,
    ClaudeCodeRequest,
    ClaudePermission,
    stream_cc_sdk_events,
)

__all__ = (
    "ClaudeCodeRequest",
    "CLAUDE_CODE_OPTION_PARAMS",  # backward compatibility
    "ClaudePermission",  # backward compatibility
    "ClaudeCodeEndpoint",
)


# --------------------------------------------------------------------------- SDK endpoint

_get_config = lambda: EndpointConfig(
    name="claude_code",
    provider="claude_code",
    base_url="internal",
    endpoint="query",
    request_options=ClaudeCodeRequest,
    timeout=3000,
    api_key="dummy-key",
)


ENDPOINT_CONFIG = _get_config()  # backward compatibility


class ClaudeCodeEndpoint(Endpoint):
    """Direct Python-SDK (non-CLI) endpoint - unchanged except for bug-fixes."""

    def __init__(self, config: EndpointConfig = None, **kwargs):
        if not HAS_CLAUDE_CODE_SDK:
            raise ImportError(
                "claude_code_sdk is not installed. "
                "Please install it with `uv pip install lionagi[claude_code_sdk]`."
            )
        warnings.warn(
            "The claude_code `query` endpoint is deprecated. "
            "Use `query_cli` endpoint instead.",
            DeprecationWarning,
        )

        config = config or _get_config()
        super().__init__(config=config, **kwargs)

    def create_payload(self, request: dict | BaseModel, **kwargs):
        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages")
        req_obj = ClaudeCodeRequest.create(messages=messages, **req_dict)
        return {"request": req_obj}, {}

    async def stream(self, request: dict | BaseModel, **kwargs):
        payload, _ = self.create_payload(request, **kwargs)
        async for chunk in stream_cc_sdk_events(payload["request"]):
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
        from claude_code_sdk import types as cc_types

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
        from claude_code_sdk import query as sdk_query
        from claude_code_sdk import types as cc_types

        responses = []
        request: ClaudeCodeRequest = payload["request"]
        system: cc_types.SystemMessage = None

        # 1. stream the Claude Code response
        async for chunk in self._stream_claude_code(**payload):
            if request.verbose_output:
                _display_message(chunk, theme=request.cli_display_theme)

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
                                theme=request.cli_display_theme,
                            )

                    responses.append(chunk)


def _display_message(chunk, theme):
    from claude_code_sdk import types as cc_types

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
                    theme=theme,
                )
            else:
                as_readable(
                    str_,
                    format_curly=True,
                    display_str=True,
                    max_panel_width=100,
                    theme=theme,
                )

    if isinstance(chunk, cc_types.ResultMessage):
        str_ = _verbose_output(chunk)
        as_readable(
            str_,
            md=True,
            display_str=True,
            format_curly=True,
            max_panel_width=100,
            theme=theme,
        )


def _verbose_output(res) -> str:
    from claude_code_sdk import types as cc_types

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
                inp_ = None

                if isinstance(block.input, dict | list):
                    inp_ = ln.json_dumps(block.input)
                else:
                    inp_ = str(block.input)

                input = inp_[:200] + "..." if len(inp_) > 200 else inp_
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
