# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint, EndpointConfig
from lionagi.utils import to_dict

from ._claude_code.models import ClaudeCodeRequest
from ._claude_code.stream_cli import (
    ClaudeChunk,
    ClaudeSession,
    log,
    stream_claude_code_cli,
)

ENDPOINT_CONFIG = EndpointConfig(
    name="claude_code_cli",
    provider="claude_code",
    base_url="internal",
    endpoint="query_cli",
    api_key="dummy",
    request_options=ClaudeCodeRequest,
    timeout=18000,  # 30 mins
)


class ClaudeCodeCLIEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = ENDPOINT_CONFIG, **kwargs):
        super().__init__(config=config, **kwargs)

    @property
    def claude_handlers(self):
        handlers = {
            "on_thinking": None,
            "on_text": None,
            "on_tool_use": None,
            "on_tool_result": None,
            "on_system": None,
            "on_final": None,
        }
        return self.config.kwargs.get("claude_handlers", handlers)

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
        log.info(
            f"Session {session.session_id} finished with {len(responses)} chunks"
        )
        texts = []
        for i in session.chunks:
            if i.text is not None:
                texts.append(i.text)

        texts.append(session.result)
        session.result = "\n".join(texts)
        return to_dict(session, recursive=True)
