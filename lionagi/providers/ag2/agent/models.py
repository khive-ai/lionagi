# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "AG2AgentRequest",
    "AgentConfig",
    "run_beta_agent",
]


class AgentConfig(BaseModel):
    """Declarative config for an AG2 beta Agent."""

    name: str = Field(default="agent", description="Agent name")
    prompt: str | list[str] = Field(
        default="You are a helpful assistant.",
        description="System prompt(s)",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names from the registry",
    )
    enable_subtasks: bool = Field(
        default=False,
        description="Enable run_subtask / run_subtasks tools",
    )
    knowledge: bool = Field(
        default=False,
        description="Enable MemoryKnowledgeStore",
    )
    observers: list[str] = Field(
        default_factory=list,
        description="Observer names: 'loop_detector', 'token_monitor'",
    )
    policies: list[str] = Field(
        default_factory=list,
        description="Policy names: 'sliding_window', 'token_budget', "
        "'working_memory', 'episodic_memory', 'alert', 'conversation'",
    )
    response_schema: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic model for structured output",
    )


class AG2AgentRequest(BaseModel):
    """Request for AG2 beta Agent endpoint."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str = ""
    agent_config: AgentConfig | None = None


def _build_observers(names: list[str]) -> list:
    observers = []
    for name in names:
        if name == "loop_detector":
            from autogen.beta.observer.loop_detector import LoopDetector

            observers.append(LoopDetector())
        elif name == "token_monitor":
            from autogen.beta.observer.token_monitor import TokenMonitor

            observers.append(TokenMonitor())
    return observers


def _build_policies(names: list[str]) -> list:
    policies = []
    for name in names:
        if name == "sliding_window":
            from autogen.beta.policies.sliding_window import SlidingWindowPolicy

            policies.append(SlidingWindowPolicy(max_events=20))
        elif name == "token_budget":
            from autogen.beta.policies.token_budget import TokenBudgetPolicy

            policies.append(TokenBudgetPolicy(max_tokens=8000))
        elif name == "working_memory":
            from autogen.beta.policies.working_memory import WorkingMemoryPolicy

            policies.append(WorkingMemoryPolicy())
        elif name == "episodic_memory":
            from autogen.beta.policies.episodic_memory import EpisodicMemoryPolicy

            policies.append(EpisodicMemoryPolicy())
        elif name == "alert":
            from autogen.beta.policies.alert import AlertPolicy

            policies.append(AlertPolicy())
        elif name == "conversation":
            from autogen.beta.policies.conversation import ConversationPolicy

            policies.append(ConversationPolicy())
    return policies


async def run_beta_agent(
    config: AgentConfig,
    message: str,
    llm_config: Any,
    tool_registry: dict[str, Callable] | None = None,
) -> AsyncIterator[Any]:
    """Run an AG2 beta Agent and yield events from its stream."""
    from autogen.beta.agent import Agent, KnowledgeConfig, TaskConfig
    from autogen.beta.events import ModelResponse
    from autogen.beta.knowledge.memory import MemoryKnowledgeStore
    from autogen.beta.stream import MemoryStream
    from autogen.beta.tools.final import tool as ag2_tool

    tool_registry = tool_registry or {}

    agent_kwargs: dict[str, Any] = {
        "name": config.name,
        "prompt": config.prompt if isinstance(config.prompt, list) else [config.prompt],
        "config": llm_config,
    }

    # Tools
    ag2_tools = []
    for tool_name in config.tools:
        if tool_name in tool_registry:
            fn = tool_registry[tool_name]
            wrapped = ag2_tool(fn, name=tool_name, description=getattr(fn, "__doc__", "") or tool_name)
            ag2_tools.append(wrapped)
    if ag2_tools:
        agent_kwargs["tools"] = ag2_tools

    # Observers
    observers = _build_observers(config.observers)
    if observers:
        agent_kwargs["observers"] = observers

    # Policies
    policies = _build_policies(config.policies)
    if policies:
        agent_kwargs["assembly"] = policies

    # Knowledge
    if config.knowledge:
        agent_kwargs["knowledge"] = KnowledgeConfig(store=MemoryKnowledgeStore())

    # Subtasks
    if config.enable_subtasks:
        agent_kwargs["tasks"] = TaskConfig()

    # Response schema
    if config.response_schema:
        agent_kwargs["response_schema"] = config.response_schema

    agent = Agent(**agent_kwargs)

    stream = MemoryStream()

    reply = await agent.ask(message, stream=stream)

    yield {"type": "response", "content": reply.response, "stream": stream}
