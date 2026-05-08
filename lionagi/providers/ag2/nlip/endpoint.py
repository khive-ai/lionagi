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
from typing import Any

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

    transport_arg_keys = ("url", "timeout", "max_retries", "agent_name", "silent")

    # Keys consumed by this endpoint — must not leak into EndpointConfig.kwargs
    _NLIP_KEYS = frozenset({"url", "timeout", "max_retries", "agent_name", "silent"})

    def __init__(self, config: EndpointConfig | None = None, **kwargs):
        # Pop NLIP-specific kwargs before they reach EndpointConfig
        nlip_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in self._NLIP_KEYS}
        super().__init__(config=config, **kwargs)
        config_nlip_kw = {
            k: self.config.kwargs.pop(k)
            for k in list(self.config.kwargs)
            if k in self._NLIP_KEYS
        }
        nlip_kw = {**config_nlip_kw, **nlip_kw}
        self._url: str = nlip_kw.get("url", "")
        self._timeout: float = nlip_kw.get("timeout", 60.0)
        self._max_retries: int = nlip_kw.get("max_retries", 3)
        self._agent_name: str = nlip_kw.get("agent_name", "remote")
        self._silent: bool | None = nlip_kw.get("silent")

    def copy_runtime_state_to(self, other):
        if isinstance(other, AG2NlipEndpoint):
            other._url = self._url
            other._timeout = self._timeout
            other._max_retries = self._max_retries
            other._agent_name = self._agent_name
            other._silent = self._silent

    def as_agent_config(
        self,
        *,
        name: str | None = None,
        role: str = "remote agent",
        description: str | None = None,
        url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        silent: bool | None = None,
        context_variables: dict[str, Any] | None = None,
        client_tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a GroupChat AgentSpec-compatible config for this remote agent."""

        from .models import NlipRemoteAgentSpec

        resolved_url = url or self._url
        if not resolved_url:
            raise ValueError("AG2NlipEndpoint requires a url to build agent config")
        return NlipRemoteAgentSpec(
            url=resolved_url,
            name=name or self._agent_name,
            role=role,
            description=description,
            silent=self._silent if silent is None else silent,
            timeout=timeout or self._timeout,
            max_retries=max_retries or self._max_retries,
            context_variables=context_variables or {},
            client_tools=client_tools or [],
        ).to_agent_config()

    def as_conversable_agent(self, **overrides: Any):
        """Build an AG2 ConversableAgent for direct composition."""

        from .models import NlipRemoteAgentSpec, build_nlip_remote_agent

        config = self.as_agent_config(**overrides)
        return build_nlip_remote_agent(
            NlipRemoteAgentSpec(
                url=config["nlip_url"],
                name=config["name"],
                role=config.get("role", "remote agent"),
                description=config.get("description"),
                silent=config.get("nlip_silent"),
                timeout=config.get("nlip_timeout", 60.0),
                max_retries=config.get("nlip_max_retries", 3),
                context_variables=config.get("context_variables", {}),
                client_tools=config.get("nlip_client_tools", []),
            )
        )

    def _runtime_config(self, kwargs: dict) -> dict:
        return {
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k in {"url", "timeout", "max_retries", "agent_name", "silent"}
        }

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

        runtime = self._runtime_config(kwargs)

        if isinstance(payload, dict) and "request" in payload:
            request_obj = payload["request"]
        else:
            request_obj = AG2NlipRequest.model_validate(payload)
        if isinstance(request_obj, dict):
            request_obj = AG2NlipRequest.model_validate(request_obj)

        if not request_obj.prompt and not request_obj.messages:
            raise ValueError(
                "AG2NlipEndpoint requires a non-empty prompt or at least one message."
            )

        url = runtime.get("url") or request_obj.url or self._url
        if not url:
            raise ValueError("AG2NlipEndpoint requires a url")

        timeout = runtime.get("timeout") or request_obj.timeout or self._timeout
        max_retries = (
            runtime.get("max_retries") or request_obj.max_retries or self._max_retries
        )
        agent_name = (
            runtime.get("agent_name") or request_obj.agent_name or self._agent_name
        )
        messages = request_obj.messages_for_call()

        result = await call_nlip_remote(
            url=url,
            messages=messages,
            agent_name=agent_name,
            timeout=timeout,
            max_retries=max_retries,
            context=request_obj.context_variables,
            client_tools=request_obj.client_tools,
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
        req_obj = AG2NlipRequest.model_validate(req_dict)
        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        from .models import AG2NlipRequest, call_nlip_remote

        runtime = self._runtime_config(kwargs)

        if isinstance(request, dict) and "request" in request:
            request_obj = request["request"]
        else:
            payload, _ = self.create_payload(request, **kwargs)
            request_obj = payload["request"]
        if isinstance(request_obj, dict):
            request_obj = AG2NlipRequest.model_validate(request_obj)

        if not request_obj.prompt and not request_obj.messages:
            raise ValueError(
                "AG2NlipEndpoint requires a non-empty prompt or at least one message."
            )

        url = runtime.get("url") or request_obj.url or self._url
        if not url:
            raise ValueError("AG2NlipEndpoint requires a url")

        timeout = runtime.get("timeout") or request_obj.timeout or self._timeout
        max_retries = (
            runtime.get("max_retries") or request_obj.max_retries or self._max_retries
        )
        agent_name = (
            runtime.get("agent_name") or request_obj.agent_name or self._agent_name
        )

        yield StreamChunk(
            type="system",
            metadata={
                "provider": "ag2",
                "api": "nlip",
                "url": url,
                "agent": agent_name,
            },
        )

        messages = request_obj.messages_for_call()

        try:
            result = await call_nlip_remote(
                url=url,
                messages=messages,
                agent_name=agent_name,
                timeout=timeout,
                max_retries=max_retries,
                context=request_obj.context_variables,
                client_tools=request_obj.client_tools,
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
