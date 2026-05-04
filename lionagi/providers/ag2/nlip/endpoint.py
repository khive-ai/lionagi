# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""AG2 NLIP Remote Agent endpoint for lionagi.

Wraps AG2's NlipRemoteAgent as a lionagi agentic endpoint.
Connects to a remote NLIP server (e.g. another sandbox running
AG2NlipApplication) and streams responses as StreamChunks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel

from lionagi.service.connections import AgenticEndpoint, EndpointConfig
from lionagi.service.types import StreamChunk
from lionagi.utils import to_dict

from .._config import AG2Configs

logger = logging.getLogger(__name__)


@AG2Configs.NLIP.register
class AG2NlipEndpoint(AgenticEndpoint):
    """Connects to a remote NLIP server via AG2's NlipRemoteAgent.

    Each call sends messages to the remote NLIP endpoint and streams
    the response back as StreamChunks. The remote server can be another
    sandbox running AG2NlipApplication, or any NLIP-compliant server.
    """

    DEFAULT_CONCURRENCY_LIMIT = 3
    DEFAULT_QUEUE_CAPACITY = 10

    # Keys consumed by this endpoint — must not leak into EndpointConfig.kwargs
    _NLIP_KEYS = frozenset({"url", "timeout", "max_retries", "agent_name"})

    def __init__(self, config: EndpointConfig | None = None, **kwargs):
        # Pop NLIP-specific kwargs before they reach EndpointConfig
        nlip_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in self._NLIP_KEYS}
        super().__init__(config=config, **kwargs)
        self._url: str = nlip_kw.get("url", "")
        self._timeout: float = nlip_kw.get("timeout", 60.0)
        self._max_retries: int = nlip_kw.get("max_retries", 3)
        self._agent_name: str = nlip_kw.get("agent_name", "remote")

    async def _call(self, payload, headers, **kwargs):
        """Call the remote NLIP server and return a structured result dict.

        Calls call_nlip_remote() directly (NLIP is already non-streaming)
        and returns:
          {
            "result":  str  — response text from the remote agent,
            "content": str  — same as result (raw field from NLIP response),
            "context": any  — context dict returned by the remote server,
            "input_required": any — non-None when the server requests more input,
            "agent":   str  — agent name used for the call,
            "url":     str  — remote NLIP endpoint URL,
          }
        """
        from .models import AG2NlipRequest, call_nlip_remote

        if isinstance(payload, dict) and "request" in payload:
            request_obj = payload["request"]
        else:
            request_obj = AG2NlipRequest(**payload)

        prompt = request_obj.prompt or (
            request_obj.messages[-1]["content"] if request_obj.messages else ""
        )
        if not prompt:
            raise ValueError(
                "AG2NlipEndpoint requires a non-empty prompt or at least one message."
            )

        url = kwargs.get("url", self._url)
        if not url:
            raise ValueError("AG2NlipEndpoint requires a url")

        timeout = kwargs.get("timeout", self._timeout)
        max_retries = kwargs.get("max_retries", self._max_retries)
        agent_name = kwargs.get("agent_name", self._agent_name)
        messages = request_obj.messages or [{"role": "user", "content": prompt}]

        result = await call_nlip_remote(
            url=url,
            messages=messages,
            agent_name=agent_name,
            timeout=timeout,
            max_retries=max_retries,
        )

        content = result.get("content", "")
        return {
            "result": content,
            "content": content,
            "context": result.get("context"),
            "input_required": result.get("input_required"),
            "agent": agent_name,
            "url": url,
        }

    def create_payload(self, request: dict | BaseModel, **kwargs):
        from .models import AG2NlipRequest

        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages", [])
        prompt = req_dict.pop("prompt", "")
        return {"request": AG2NlipRequest(messages=messages, prompt=prompt)}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        from .models import call_nlip_remote

        if isinstance(request, dict) and "request" in request:
            request_obj = request["request"]
        else:
            payload, _ = self.create_payload(request, **kwargs)
            request_obj = payload["request"]

        prompt = request_obj.prompt or (
            request_obj.messages[-1]["content"] if request_obj.messages else ""
        )
        if not prompt:
            raise ValueError(
                "AG2NlipEndpoint requires a non-empty prompt or at least one message."
            )

        url = kwargs.get("url", self._url)
        if not url:
            raise ValueError("AG2NlipEndpoint requires a url")

        timeout = kwargs.get("timeout", self._timeout)
        max_retries = kwargs.get("max_retries", self._max_retries)
        agent_name = kwargs.get("agent_name", self._agent_name)

        yield StreamChunk(
            type="system",
            metadata={
                "provider": "ag2",
                "api": "nlip",
                "url": url,
                "agent": agent_name,
            },
        )

        messages = request_obj.messages or [{"role": "user", "content": prompt}]

        try:
            result = await call_nlip_remote(
                url=url,
                messages=messages,
                agent_name=agent_name,
                timeout=timeout,
                max_retries=max_retries,
            )

            if result.get("content"):
                yield StreamChunk(
                    type="text",
                    content=result["content"],
                    metadata={
                        "agent": agent_name,
                        "url": url,
                        "context": result.get("context"),
                    },
                )

            if result.get("input_required"):
                yield StreamChunk(
                    type="system",
                    content=f"Input required: {result['input_required']}",
                    metadata={"event": "input_required", "agent": agent_name},
                )

        except Exception:
            logger.exception("AG2 NLIP remote call failed")
            raise

        yield StreamChunk(
            type="result",
            content=result.get("content", ""),
            metadata={"agent": agent_name, "url": url},
        )
