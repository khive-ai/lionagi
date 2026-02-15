# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from lionagi.service.connections.cli_endpoint import CLIEndpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict

from ...third_party.gemini_models import (
    GeminiChunk,
    GeminiCodeRequest,
    GeminiSession,
    stream_gemini_cli,
)
from ...third_party.gemini_models import log as gemini_log

_get_config = lambda: EndpointConfig(
    name="gemini_cli",
    provider="gemini_code",
    base_url="internal",
    endpoint="query_cli",
    api_key="dummy-key",
    request_options=GeminiCodeRequest,
    timeout=18000,  # 30 mins
)

ENDPOINT_CONFIG = _get_config()

_GEMINI_HANDLER_PARAMS = (
    "on_text",
    "on_tool_use",
    "on_tool_result",
    "on_final",
)


def _validate_handlers(handlers: dict[str, Callable | None], /) -> None:
    if not isinstance(handlers, dict):
        raise ValueError("Handlers must be a dictionary")
    for k, v in handlers.items():
        if k not in _GEMINI_HANDLER_PARAMS:
            raise ValueError(f"Invalid handler key: {k}")
        if not (v is None or callable(v)):
            raise ValueError(f"Handler value must be callable or None, got {type(v)}")


class GeminiCLIEndpoint(CLIEndpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_config()
        super().__init__(config=config, **kwargs)

    @property
    def gemini_handlers(self):
        handlers = {k: None for k in _GEMINI_HANDLER_PARAMS}
        return self.config.kwargs.get("gemini_handlers", handlers)

    @gemini_handlers.setter
    def gemini_handlers(self, value: dict):
        _validate_handlers(value)
        self.config.kwargs["gemini_handlers"] = value

    def update_handlers(self, **kwargs):
        _validate_handlers(kwargs)
        handlers = {**self.gemini_handlers, **kwargs}
        self.gemini_handlers = handlers

    def create_payload(self, request: dict | BaseModel, **kwargs):
        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages")
        req_obj = GeminiCodeRequest(messages=messages, **req_dict)
        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[GeminiChunk | dict | GeminiSession]:
        payload, _ = self.create_payload(request, **kwargs)
        request_obj = payload["request"]
        async for chunk in stream_gemini_cli(request_obj):
            yield chunk

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ):
        responses = []
        request: GeminiCodeRequest = payload["request"]
        session: GeminiSession = GeminiSession()

        async for chunk in stream_gemini_cli(request, session, **self.gemini_handlers, **kwargs):
            if isinstance(chunk, dict):
                if chunk.get("type") == "done":
                    break
            responses.append(chunk)

        gemini_log.info(f"Session {session.session_id} finished with {len(responses)} chunks")

        # Accumulate text from chunks, concatenating delta fragments
        parts = []
        current_delta: list[str] = []
        for i in session.chunks:
            if i.text is not None:
                if i.is_delta:
                    current_delta.append(i.text)
                else:
                    if current_delta:
                        parts.append("".join(current_delta))
                        current_delta = []
                    parts.append(i.text)
        if current_delta:
            parts.append("".join(current_delta))

        # Use chunk text if available, fall back to session.result
        if parts:
            session.result = "\n".join(parts)
        # else: keep session.result from the "result" event as-is
        if request.cli_include_summary:
            session.populate_summary()

        return to_dict(session, recursive=True)
