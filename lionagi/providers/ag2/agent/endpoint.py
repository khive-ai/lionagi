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

    def __init__(self, config: EndpointConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._agent_config: dict[str, Any] = kwargs.get("agent_config", {})
        self._llm_config: Any = kwargs.get("llm_config", None)
        self._tool_registry: dict[str, Any] = kwargs.get("tool_registry", {})

    async def _call(self, payload, headers, **kwargs):
        raise NotImplementedError(
            "AG2 beta Agent is stream-only. Use stream() to iterate events."
        )

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
        from .models import AG2AgentRequest, AgentConfig, run_beta_agent

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

        # Convert dict llm_config to AG2 beta ModelConfig
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
                if event.get("type") == "response":
                    response = event["content"]
                    content = ""
                    if response.message:
                        content = getattr(response.message, "content", str(response.message))

                    yield StreamChunk(
                        type="text",
                        content=content,
                        metadata={"agent": agent_config.name},
                    )

                    if response.tool_calls:
                        for call_event in response.tool_calls.calls:
                            yield StreamChunk(
                                type="tool_use",
                                tool_name=call_event.name,
                                tool_id=call_event.id,
                                tool_input=call_event.arguments,
                                metadata={"agent": agent_config.name},
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
