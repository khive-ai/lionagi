# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""AG2 (AutoGen) GroupChat endpoint for lionagi.

Thin adapter: delegates to build_group_chat() + stream_group_chat()
from models.py, converts AG2 events to StreamChunk.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from lionagi.service.connections import AgenticEndpoint, EndpointConfig
from lionagi.service.types import StreamChunk
from lionagi.utils import to_dict

from .._config import AG2Configs

logger = logging.getLogger(__name__)


def _validate_handlers(handlers: dict[str, Callable | None], /) -> None:
    from .models import AG2_HANDLER_PARAMS

    if not isinstance(handlers, dict):
        raise ValueError("Handlers must be a dictionary")
    for k, v in handlers.items():
        if k not in AG2_HANDLER_PARAMS:
            raise ValueError(f"Invalid handler key: {k}")
        if not (v is None or callable(v)):
            raise ValueError(f"Handler value must be callable or None, got {type(v)}")


@AG2Configs.GROUP_CHAT.register
class AG2GroupChatEndpoint(AgenticEndpoint):
    """Wraps AG2 v0.12 GroupChat as a lionagi endpoint.

    Delegates to ``build_group_chat()`` and ``stream_group_chat()``
    for all AG2 logic. Config auto-created from ``_ENDPOINT_META``.
    """

    DEFAULT_CONCURRENCY_LIMIT = 1
    DEFAULT_QUEUE_CAPACITY = 3

    from .models import AG2_HANDLER_PARAMS

    transport_arg_keys = AG2_HANDLER_PARAMS + (
        "agent_configs",
        "llm_config",
        "tool_registry",
        "code_executor",
    )

    # Keys consumed by this endpoint — must not leak into EndpointConfig.kwargs
    _AG2_KEYS = frozenset(
        {
            "agent_configs",
            "llm_config",
            "tool_registry",
            "code_executor",
            "ag2_handlers",
        }
    )

    def __init__(self, config: EndpointConfig | None = None, **kwargs):
        # Pop AG2-specific kwargs before they reach EndpointConfig
        ag2_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in self._AG2_KEYS}
        super().__init__(config=config, **kwargs)
        config_ag2_kw = {
            k: self.config.kwargs.pop(k)
            for k in list(self.config.kwargs)
            if k in self._AG2_KEYS
        }
        ag2_kw = {**config_ag2_kw, **ag2_kw}
        self._agent_configs: list[dict[str, Any]] = ag2_kw.get("agent_configs", [])
        self._llm_config: Any = ag2_kw.get("llm_config", False)
        self._tool_registry: dict[str, Any] = ag2_kw.get("tool_registry", {})
        self._code_executor: Any = ag2_kw.get("code_executor")
        self._ag2_handlers = {k: None for k in self.AG2_HANDLER_PARAMS}
        if handlers := ag2_kw.get("ag2_handlers"):
            _validate_handlers(handlers)
            self._ag2_handlers.update(handlers)

    def copy_runtime_state_to(self, other):
        if isinstance(other, AG2GroupChatEndpoint):
            other._agent_configs = _copy_runtime_value(self._agent_configs)
            other._llm_config = _copy_runtime_value(self._llm_config)
            other._tool_registry = _copy_runtime_value(self._tool_registry)
            other._code_executor = self._code_executor
            other.ag2_handlers = self.ag2_handlers.copy()

    @property
    def ag2_handlers(self):
        return self._ag2_handlers

    @ag2_handlers.setter
    def ag2_handlers(self, value: dict):
        _validate_handlers(value)
        self._ag2_handlers = {k: None for k in self.AG2_HANDLER_PARAMS}
        self._ag2_handlers.update(value)

    def update_handlers(self, **kwargs):
        _validate_handlers(kwargs)
        handlers = {**self.ag2_handlers, **kwargs}
        self.ag2_handlers = handlers

    def _runtime_handlers(self, kwargs: dict) -> dict:
        handlers = self.ag2_handlers.copy()
        call_handlers = {
            k: kwargs.pop(k) for k in list(kwargs) if k in self.AG2_HANDLER_PARAMS
        }
        if call_handlers:
            _validate_handlers(call_handlers)
            handlers.update(call_handlers)
        return {k: v for k, v in handlers.items() if v is not None}

    def _runtime_config(self, kwargs: dict) -> dict:
        return {
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k in {"agent_configs", "llm_config", "tool_registry", "code_executor"}
        }

    async def _call(self, payload, headers, **kwargs):
        """Collect all stream events and return a structured transcript dict.

        Accumulates every StreamChunk from stream() into a rich result dict
        mirroring the claude_code / codex pattern:
          {
            "result":       str  — final concatenated text from all agents,
            "transcript":   list — ordered list of all events (text, tool_use,
                                   tool_result, system),
            "agents":       list — unique agent names that produced output,
            "tool_calls":   list — tool_use entries (name + args),
            "tool_results": list — tool_result entries (output),
          }
        """
        transcript: list[dict] = []
        text_parts: list[str] = []
        agents: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        async for chunk in self.stream(payload, **kwargs):
            agent = (chunk.metadata or {}).get("agent", "unknown")

            if chunk.type == "text":
                if chunk.content:
                    text_parts.append(chunk.content)
                    if agent and agent not in agents:
                        agents.append(agent)
                    transcript.append(
                        {"agent": agent, "type": "text", "content": chunk.content}
                    )

            elif chunk.type == "tool_use":
                entry = {
                    "agent": agent,
                    "type": "tool_use",
                    "name": chunk.tool_name,
                    "args": chunk.tool_input,
                }
                tool_calls.append(entry)
                transcript.append(entry)

            elif chunk.type == "tool_result":
                entry = {
                    "agent": agent,
                    "type": "tool_result",
                    "output": chunk.tool_output,
                    "tool_call_id": (chunk.metadata or {}).get("tool_call_id"),
                }
                tool_results.append(entry)
                transcript.append(entry)

            elif chunk.type == "system":
                # Speaker-turn and candidate events carry agent metadata
                entry: dict = {"type": "system"}
                if chunk.content:
                    entry["content"] = chunk.content
                meta = chunk.metadata or {}
                if meta:
                    entry["metadata"] = meta
                if agent and agent != "unknown" and agent not in agents:
                    agents.append(agent)
                transcript.append(entry)

            # "result" chunk is the terminal sentinel — skip it; we build our own

        return {
            "result": "\n".join(text_parts),
            "transcript": transcript,
            "agents": agents,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    def create_payload(self, request: dict | BaseModel, **kwargs):
        from .models import AG2GroupChatRequest

        req_dict = {**self.config.kwargs, **to_dict(request), **kwargs}
        req_obj = AG2GroupChatRequest.model_validate(req_dict)
        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        from .models import AG2GroupChatRequest, build_group_chat, stream_group_chat

        handlers = self._runtime_handlers(kwargs)
        runtime = self._runtime_config(kwargs)

        if isinstance(request, dict) and "request" in request:
            request_obj = request["request"]
        else:
            payload, _ = self.create_payload(request, **kwargs)
            request_obj = payload["request"]
        if isinstance(request_obj, dict):
            request_obj = AG2GroupChatRequest.model_validate(request_obj)

        run_messages = request_obj.run_messages()
        if not request_obj.prompt and not request_obj.messages:
            raise ValueError(
                "AG2GroupChatEndpoint requires a non-empty prompt or at least one message."
            )

        agent_configs = runtime.get("agent_configs", self._agent_configs)
        llm_config = (
            runtime["llm_config"]
            if "llm_config" in runtime
            else (
                request_obj.llm_config
                if request_obj.llm_config is not None
                else self._llm_config
            )
        )
        tool_registry = runtime.get("tool_registry", self._tool_registry)
        code_executor = runtime.get("code_executor", self._code_executor)

        spec = request_obj.to_group_chat_spec(agent_configs=agent_configs)

        user, pattern, agents_by_name = build_group_chat(
            spec, llm_config, tool_registry, code_executor
        )

        yield StreamChunk(
            type="system",
            metadata={
                "provider": "ag2",
                "api": "v0.12",
                "pattern": spec.pattern,
                "agent_count": len(spec.agents),
                "max_round": request_obj.max_round,
                "user_agent": getattr(user, "name", None),
            },
        )

        async for event in stream_group_chat(
            pattern=pattern,
            prompt=run_messages,
            max_rounds=request_obj.max_round,
            safeguard_policy=request_obj.safeguard_policy,
            safeguard_llm_config=request_obj.safeguard_llm_config,
            mask_llm_config=request_obj.mask_llm_config,
            yield_on=request_obj.yield_on,
            **handlers,
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
    """Convert an AG2 wrapped event to a StreamChunk.

    AG2's @wrap_event decorator produces a two-layer structure:
      event.type    — the event type string
      event.content — the inner event with actual data

    All attribute access for payload data must go through event.content.
    """
    from autogen.events.agent_events import (
        ErrorEvent,
        GroupChatRunChatEvent,
        RunCompletionEvent,
        SelectSpeakerEvent,
        TerminationEvent,
        TextEvent,
        ToolCallEvent,
        ToolResponseEvent,
    )

    inner = getattr(event, "content", None)

    if isinstance(event, TextEvent):
        text = (
            getattr(inner, "content", str(event)) if inner is not None else str(event)
        )
        sender = getattr(inner, "sender", "unknown") if inner is not None else "unknown"
        return StreamChunk(
            type="text",
            content=text,
            metadata={"agent": sender},
        )
    if isinstance(event, GroupChatRunChatEvent):
        speaker = (
            getattr(inner, "speaker", "unknown") if inner is not None else "unknown"
        )
        return StreamChunk(
            type="system",
            content=f"Speaker: {speaker}",
            metadata={"event": "speaker_turn", "agent": speaker},
        )
    if isinstance(event, SelectSpeakerEvent):
        agents = getattr(inner, "agents", []) if inner is not None else []
        if agents:
            names = ", ".join(getattr(a, "name", str(a)) for a in agents)
        else:
            names = "?"
        return StreamChunk(
            type="system",
            content=f"Speaker candidates: {names}",
            metadata={"event": "speaker_candidates"},
        )
    if isinstance(event, ToolCallEvent):
        tool_calls = getattr(inner, "tool_calls", []) if inner is not None else []
        first = tool_calls[0] if tool_calls else None
        tool_name = (
            getattr(getattr(first, "function", None), "name", None) if first else None
        )
        tool_args = (
            getattr(getattr(first, "function", None), "arguments", None)
            if first
            else None
        )
        sender = getattr(inner, "sender", "unknown") if inner is not None else "unknown"
        return StreamChunk(
            type="tool_use",
            tool_name=tool_name,
            tool_id=None,
            tool_input=tool_args,
            metadata={"agent": sender},
        )
    if isinstance(event, ToolResponseEvent):
        tool_responses = (
            getattr(inner, "tool_responses", []) if inner is not None else []
        )
        first = tool_responses[0] if tool_responses else None
        tool_output = getattr(first, "content", None) if first else None
        tool_id = getattr(first, "tool_call_id", None) if first else None
        sender = getattr(inner, "sender", "unknown") if inner is not None else "unknown"
        return StreamChunk(
            type="tool_result",
            tool_output=tool_output,
            metadata={"agent": sender, "tool_call_id": tool_id},
        )
    if isinstance(event, RunCompletionEvent):
        summary = getattr(inner, "summary", None) if inner is not None else None
        last_speaker = (
            getattr(inner, "last_speaker", None) if inner is not None else None
        )
        return StreamChunk(
            type="result",
            content=summary or "GroupChat complete",
            metadata={
                "event": "run_completion",
                "last_speaker": last_speaker,
                "cost": getattr(inner, "cost", None) if inner is not None else None,
            },
        )
    if isinstance(event, TerminationEvent):
        return StreamChunk(
            type="system",
            content=getattr(inner, "reason", None) if inner is not None else None,
            metadata={"event": "termination"},
        )
    if isinstance(event, ErrorEvent):
        error = getattr(inner, "error", None) if inner is not None else None
        return StreamChunk(
            type="result",
            content=str(error) if error is not None else "AG2 GroupChat error",
            metadata={"event": "error"},
            is_error=True,
        )
    return None


def _copy_runtime_value(value):
    try:
        return deepcopy(value)
    except Exception:
        return value
