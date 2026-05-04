# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""AG2 beta Agent endpoint for lionagi.

Wraps autogen.beta.Agent as a lionagi agentic endpoint.
Events from the beta stream are converted to StreamChunks.
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


@AG2Configs.AGENT.register
class AG2BetaEndpoint(AgenticEndpoint):
    """Wraps AG2 beta Agent as a lionagi endpoint.

    Single-agent execution with full middleware stack:
    tools, observers, policies, knowledge, subtasks.
    """

    DEFAULT_CONCURRENCY_LIMIT = 1
    DEFAULT_QUEUE_CAPACITY = 3

    # Keys consumed by this endpoint — must not leak into EndpointConfig.kwargs
    _AG2_KEYS = frozenset({"agent_config", "llm_config", "tool_registry"})

    def __init__(self, config: EndpointConfig | None = None, **kwargs):
        # Pop AG2-specific kwargs before they reach EndpointConfig
        ag2_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in self._AG2_KEYS}
        super().__init__(config=config, **kwargs)
        self._agent_config: dict[str, Any] = ag2_kw.get("agent_config", {})
        self._llm_config: Any = ag2_kw.get("llm_config", None)
        self._tool_registry: dict[str, Any] = ag2_kw.get("tool_registry", {})

    async def _call(self, payload, headers, **kwargs):
        """Collect all stream events and return a structured result dict.

        Accumulates every StreamChunk from stream() into a rich result dict
        mirroring the claude_code / codex pattern:
          {
            "result":       str  — final text from the agent response,
            "transcript":   list — ordered list of all events,
            "tool_calls":   list — tool_use entries (name + args),
            "tool_results": list — tool_result entries (output),
          }
        """
        transcript: list[dict] = []
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        async for chunk in self.stream(payload, **kwargs):
            agent = (chunk.metadata or {}).get("agent", "unknown")

            if chunk.type == "text":
                if chunk.content:
                    text_parts.append(chunk.content)
                entry: dict = {"type": "text", "agent": agent}
                if chunk.content:
                    entry["content"] = chunk.content
                typed_result = (chunk.metadata or {}).get("typed_result")
                if typed_result is not None:
                    entry["typed_result"] = typed_result
                transcript.append(entry)

            elif chunk.type == "tool_use":
                entry = {
                    "type": "tool_use",
                    "agent": agent,
                    "name": chunk.tool_name,
                    "id": chunk.tool_id,
                    "args": chunk.tool_input,
                }
                tool_calls.append(entry)
                transcript.append(entry)

            elif chunk.type == "tool_result":
                entry = {
                    "type": "tool_result",
                    "agent": agent,
                    "output": chunk.tool_output,
                    "tool_name": (chunk.metadata or {}).get("tool_name"),
                }
                tool_results.append(entry)
                transcript.append(entry)

            elif chunk.type == "system":
                entry = {"type": "system"}
                if chunk.content:
                    entry["content"] = chunk.content
                meta = chunk.metadata or {}
                if meta:
                    entry["metadata"] = meta
                transcript.append(entry)

            # "result" chunk is the terminal sentinel — skip it; we build our own

        return {
            "result": "\n".join(text_parts),
            "transcript": transcript,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    def create_payload(self, request: dict | BaseModel, **kwargs):
        from .models import AG2AgentRequest

        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages", [])
        prompt = req_dict.pop("prompt", "")
        agent_config = req_dict.pop("agent_config", None)
        return {
            "request": AG2AgentRequest(
                messages=messages,
                prompt=prompt,
                agent_config=agent_config,
            )
        }, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        from .models import AgentConfig, run_beta_agent

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
                "AG2BetaEndpoint requires a non-empty prompt or at least one message."
            )

        agent_config = request_obj.agent_config
        if agent_config is None:
            agent_config = AgentConfig(**kwargs.get("agent_config", self._agent_config))

        llm_config = kwargs.get("llm_config", self._llm_config)
        tool_registry = kwargs.get("tool_registry", self._tool_registry)

        if llm_config is None:
            raise ValueError("AG2BetaEndpoint requires llm_config")

        model_config = _resolve_model_config(llm_config)

        yield StreamChunk(
            type="system",
            metadata={
                "provider": "ag2",
                "api": "beta",
                "agent": agent_config.name,
                "tools": agent_config.tools,
                "observers": agent_config.observers,
                "policies": agent_config.policies,
            },
        )

        try:
            async for event in run_beta_agent(
                config=agent_config,
                message=prompt,
                llm_config=model_config,
                tool_registry=tool_registry,
            ):
                etype = event.get("type")

                if etype == "tool_use":
                    yield StreamChunk(
                        type="tool_use",
                        tool_name=event.get("name"),
                        tool_id=event.get("id"),
                        tool_input=event.get("arguments"),
                        metadata={"agent": agent_config.name},
                    )

                elif etype == "tool_result":
                    yield StreamChunk(
                        type="tool_result",
                        tool_output=event.get("content"),
                        metadata={
                            "agent": agent_config.name,
                            "tool_name": event.get("name"),
                        },
                    )

                elif etype == "response":
                    content = event.get("text", "")
                    typed_result = event.get("typed_result")

                    yield StreamChunk(
                        type="text",
                        content=content,
                        metadata={
                            "agent": agent_config.name,
                            "typed_result": typed_result,
                        },
                    )

        except Exception:
            logger.exception("AG2 beta Agent execution failed")
            raise

        yield StreamChunk(
            type="result",
            content="Agent complete",
            metadata={"agent": agent_config.name},
        )


def _resolve_model_config(llm_config: Any) -> Any:
    """Convert a dict llm_config to an AG2 beta ModelConfig."""
    if not isinstance(llm_config, dict):
        return llm_config

    api_type = llm_config.get("api_type", "openai")
    model = llm_config.get("model", "gpt-4o-mini")
    api_key = llm_config.get("api_key")
    temperature = llm_config.get("temperature")

    kwargs = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if temperature is not None:
        kwargs["temperature"] = temperature

    if api_type == "openai":
        from autogen.beta.config.openai.config import OpenAIConfig

        return OpenAIConfig(**kwargs)

    elif api_type == "anthropic":
        from autogen.beta.config.anthropic.config import AnthropicConfig

        return AnthropicConfig(**kwargs)

    elif api_type == "gemini":
        from autogen.beta.config.gemini.config import GeminiConfig

        kwargs.pop("temperature", None)
        return GeminiConfig(**kwargs)

    elif api_type == "ollama":
        from autogen.beta.config.ollama.config import OllamaConfig

        kwargs.pop("api_key", None)
        kwargs.pop("temperature", None)
        return OllamaConfig(**kwargs)

    raise ValueError(f"Unknown api_type: {api_type}")
