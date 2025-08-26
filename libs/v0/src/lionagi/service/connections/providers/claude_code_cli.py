# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint, EndpointConfig
from lionagi.utils import to_dict

from ...third_party.claude_code import (
    ClaudeChunk,
    ClaudeCodeRequest,
    ClaudeSession,
)
from ...third_party.claude_code import log as cc_log
from ...third_party.claude_code import (
    stream_claude_code_cli,
)

_get_config = lambda: EndpointConfig(
    name="claude_code_cli",
    provider="claude_code",
    base_url="internal",
    endpoint="query_cli",
    api_key="dummy-key",
    request_options=ClaudeCodeRequest,
    timeout=18000,  # 30 mins
)

ENDPOINT_CONFIG = _get_config()  # backward compatibility


_CLAUDE_HANDLER_PARAMS = (
    "on_thinking",
    "on_text",
    "on_tool_use",
    "on_tool_result",
    "on_system",
    "on_final",
)


def _validate_handlers(handlers: dict[str, Callable | None], /) -> None:
    if not isinstance(handlers, dict):
        raise ValueError("Handlers must be a dictionary")
    for k, v in handlers.items():
        if k not in _CLAUDE_HANDLER_PARAMS:
            raise ValueError(f"Invalid handler key: {k}")
        if not (v is None or callable(v)):
            raise ValueError(
                f"Handler value must be callable or None, got {type(v)}"
            )


class ClaudeCodeCLIEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_config()
        super().__init__(config=config, **kwargs)

    @property
    def claude_handlers(self):
        handlers = {k: None for k in _CLAUDE_HANDLER_PARAMS}
        return self.config.kwargs.get("claude_handlers", handlers)

    @claude_handlers.setter
    def claude_handlers(self, value: dict):
        _validate_handlers(value)
        self.config.kwargs["claude_handlers"] = value

    def update_handlers(self, **kwargs):
        _validate_handlers(kwargs)
        handlers = {**self.claude_handlers, **kwargs}
        self.claude_handlers = handlers

    def create_payload(self, request: dict | BaseModel, **kwargs):
        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages")
        req_obj = ClaudeCodeRequest.create(messages=messages, **req_dict)
        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[ClaudeChunk | dict | ClaudeSession]:
        payload, _ = self.create_payload(request, **kwargs)["request"]
        async for chunk in stream_claude_code_cli(payload):
            yield chunk

    async def _call(
        self,
        payload: dict,
        headers: dict,  # type: ignore[unused-argument]
        **kwargs,
    ):
        responses = []
        request: ClaudeCodeRequest = payload["request"]
        session: ClaudeSession = ClaudeSession()
        system: dict = None

        # 1. stream the Claude Code response
        async for chunk in stream_claude_code_cli(
            request, session, **self.claude_handlers, **kwargs
        ):
            if isinstance(chunk, dict):
                if chunk.get("type") == "done":
                    break
                system = chunk
            responses.append(chunk)

        if request.auto_finish and not isinstance(
            responses[-1], ClaudeSession
        ):
            req2 = request.model_copy(deep=True)
            req2.prompt = "Please provide a the final result message only"
            req2.max_turns = 1
            req2.continue_conversation = True
            if system:
                req2.resume = system.get("session_id") if system else None

            async for chunk in stream_claude_code_cli(req2, session, **kwargs):
                responses.append(chunk)
                if isinstance(chunk, ClaudeSession):
                    break
        cc_log.info(
            f"Session {session.session_id} finished with {len(responses)} chunks"
        )
        texts = []
        for i in session.chunks:
            if i.text is not None:
                texts.append(i.text)

        texts.append(session.result)
        session.result = "\n".join(texts)
        if request.cli_include_summary:
            session.populate_summary()

        return to_dict(session, recursive=True)
