# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from lionagi.service.connections.cli_endpoint import CLIEndpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import to_dict

from ...third_party.codex_models import (
    CodexChunk,
    CodexCodeRequest,
    CodexSession,
    stream_codex_cli,
)
from ...third_party.codex_models import log as codex_log

_get_config = lambda: EndpointConfig(
    name="codex_cli",
    provider="codex",
    base_url="internal",
    endpoint="query_cli",
    api_key="dummy-key",
    request_options=CodexCodeRequest,
    timeout=18000,  # 30 mins
)

ENDPOINT_CONFIG = _get_config()

_CODEX_HANDLER_PARAMS = (
    "on_text",
    "on_tool_use",
    "on_tool_result",
    "on_final",
)


def _validate_handlers(handlers: dict[str, Callable | None], /) -> None:
    if not isinstance(handlers, dict):
        raise ValueError("Handlers must be a dictionary")
    for k, v in handlers.items():
        if k not in _CODEX_HANDLER_PARAMS:
            raise ValueError(f"Invalid handler key: {k}")
        if not (v is None or callable(v)):
            raise ValueError(f"Handler value must be callable or None, got {type(v)}")


class CodexCLIEndpoint(CLIEndpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_config()
        super().__init__(config=config, **kwargs)

    @property
    def codex_handlers(self):
        handlers = {k: None for k in _CODEX_HANDLER_PARAMS}
        return self.config.kwargs.get("codex_handlers", handlers)

    @codex_handlers.setter
    def codex_handlers(self, value: dict):
        _validate_handlers(value)
        self.config.kwargs["codex_handlers"] = value

    def update_handlers(self, **kwargs):
        _validate_handlers(kwargs)
        handlers = {**self.codex_handlers, **kwargs}
        self.codex_handlers = handlers

    def create_payload(self, request: dict | BaseModel, **kwargs):
        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages")
        req_obj = CodexCodeRequest(messages=messages, **req_dict)
        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[CodexChunk | dict | CodexSession]:
        payload, _ = self.create_payload(request, **kwargs)
        request_obj = payload["request"]
        async for chunk in stream_codex_cli(request_obj):
            yield chunk

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ):
        responses = []
        request: CodexCodeRequest = payload["request"]
        session: CodexSession = CodexSession()

        async for chunk in stream_codex_cli(request, session, **self.codex_handlers, **kwargs):
            if isinstance(chunk, dict):
                if chunk.get("type") == "done":
                    break
            responses.append(chunk)

        codex_log.info(f"Session {session.session_id} finished with {len(responses)} chunks")
        texts = []
        for i in session.chunks:
            if i.text is not None:
                texts.append(i.text)

        texts.append(session.result)
        session.result = "\n".join(texts)
        if request.cli_include_summary:
            session.populate_summary()

        return to_dict(session, recursive=True)
