# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""AG2 (AutoGen) GroupChat endpoint for lionagi.

Thin adapter: delegates to build_group_chat() + stream_group_chat()
from models.py, converts AG2 events to StreamChunk.
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


@AG2Configs.GROUP_CHAT.register
class AG2GroupChatEndpoint(AgenticEndpoint):
    """Wraps AG2 v0.9 GroupChat as a lionagi endpoint.

    Delegates to ``build_group_chat()`` and ``stream_group_chat()``
    for all AG2 logic. Config auto-created from ``_ENDPOINT_META``.
    """

    DEFAULT_CONCURRENCY_LIMIT = 1
    DEFAULT_QUEUE_CAPACITY = 3

    def __init__(self, config: EndpointConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._agent_configs: list[dict[str, Any]] = kwargs.get("agent_configs", [])
        self._llm_config: dict[str, Any] = kwargs.get("llm_config", {})
        self._tool_registry: dict[str, Any] = kwargs.get("tool_registry", {})

    def create_payload(self, request: dict | BaseModel, **kwargs):
        from .models import AG2GroupChatRequest

        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        messages = req_dict.pop("messages", [])
        prompt = req_dict.pop("prompt", "")
        max_round = req_dict.pop("max_round", 15)
        ctx = req_dict.pop("context_variables", {})
        return {
            "request": AG2GroupChatRequest(
                messages=messages,
                prompt=prompt,
                max_round=max_round,
                context_variables=ctx,
            )
        }, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        from .models import GroupChatSpec, build_group_chat, stream_group_chat

        if isinstance(request, dict) and "request" in request:
            request_obj = request["request"]
        else:
            payload, _ = self.create_payload(request, **kwargs)
            request_obj = payload["request"]

        prompt = request_obj.prompt or (
            request_obj.messages[-1]["content"] if request_obj.messages else ""
        )

        agent_configs = kwargs.get("agent_configs", self._agent_configs)
        llm_config = kwargs.get("llm_config", self._llm_config)
        tool_registry = kwargs.get("tool_registry", self._tool_registry)

        spec = GroupChatSpec(
            name="endpoint_chat",
            objective=prompt,
            agents=[
                {
                    "name": c["name"],
                    "role": c.get("role", ""),
                    "system_message": c.get("system_message", ""),
                    "tools": c.get("tools", []),
                    "handoffs": [
                        {"target": h["target"], "condition": h["condition"]}
                        for h in c.get("handoff_conditions", c.get("handoffs", []))
                    ],
                    "state_template": c.get("state_template"),
                }
                for c in agent_configs
            ],
            context=request_obj.context_variables,
            max_round=request_obj.max_round,
        )

        user, pattern, agents_by_name = build_group_chat(
            spec,
            llm_config,
            tool_registry,
        )

        yield StreamChunk(
            type="system",
            metadata={
                "provider": "ag2",
                "api": "v0.9",
                "pattern": "DefaultPattern",
                "agent_count": len(agent_configs),
                "max_round": request_obj.max_round,
            },
        )

        async for event in stream_group_chat(
            pattern=pattern,
            prompt=prompt,
            max_rounds=request_obj.max_round,
        ):
            chunk = _event_to_chunk(event)
            if chunk:
                yield chunk

        yield StreamChunk(
            type="result",
            content="GroupChat complete",
            metadata={"agents": list(agents_by_name.keys())},
        )


def _event_to_chunk(event) -> StreamChunk | None:
    from autogen.events.agent_events import (
        SelectSpeakerEvent,
        TextEvent,
        ToolCallEvent,
        ToolResponseEvent,
    )

    if isinstance(event, TextEvent):
        return StreamChunk(
            type="text",
            content=getattr(event, "content", str(event)),
            metadata={"agent": getattr(event, "source", "unknown")},
        )
    if isinstance(event, SelectSpeakerEvent):
        return StreamChunk(
            type="system",
            content=f"Speaker: {getattr(event, 'selected_agent_name', '?')}",
            metadata={"event": "speaker_selected"},
        )
    if isinstance(event, ToolCallEvent):
        return StreamChunk(
            type="tool_use",
            tool_name=getattr(event, "tool_name", None),
            tool_id=getattr(event, "tool_call_id", None),
            tool_input=getattr(event, "arguments", None),
            metadata={"agent": getattr(event, "source", "unknown")},
        )
    if isinstance(event, ToolResponseEvent):
        return StreamChunk(
            type="tool_result",
            tool_output=getattr(event, "content", None),
            metadata={"agent": getattr(event, "source", "unknown")},
        )
    return None
