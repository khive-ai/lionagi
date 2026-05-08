# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..nlip.models import NlipRemoteAgentSpec

logger = logging.getLogger(__name__)

HumanInputMode = Literal["ALWAYS", "NEVER", "TERMINATE"]
GroupChatPattern = Literal["default", "auto", "round_robin", "random", "manual"]
ConditionKind = Literal["llm", "context", "context_expression"]
AvailableKind = Literal["context", "context_expression"]

__all__ = [
    "HandoffCondition",
    "AfterWorkCondition",
    "AgentSpec",
    "GroupChatSpec",
    "ResearchPlan",
    "AG2GroupChatRequest",
    "AG2_HANDLER_PARAMS",
    "build_group_chat",
    "stream_group_chat",
]


# ---------------------------------------------------------------------------
# Models (structured output targets)
# ---------------------------------------------------------------------------


class HandoffCondition(BaseModel):
    """AG2 handoff condition for a ConversableAgent.

    ``condition_type="llm"`` maps to ``OnCondition``. Context variants map to
    ``OnContextCondition`` and are evaluated before LLM conditions by AG2.
    """

    target: str = Field(description="Target agent name or transition target name")
    condition: str | None = Field(
        default=None,
        description="LLM prompt, context variable name, or context expression",
    )
    condition_type: ConditionKind = Field(
        default="llm",
        description="How AG2 should evaluate this transition condition",
    )
    available: str | None = Field(
        default=None,
        description="Optional context gate for making this handoff available",
    )
    available_type: AvailableKind = Field(
        default="context",
        description="How AG2 should evaluate the availability gate",
    )
    llm_function_name: str | None = Field(
        default=None,
        description="Optional tool/function name AG2 exposes for an LLM handoff",
    )


class AfterWorkCondition(BaseModel):
    """AG2 after-work condition evaluated after explicit handoffs."""

    target: str = Field(description="Target agent name or transition target name")
    condition: str | None = Field(
        default=None,
        description="Context variable name or expression; None is a fallback",
    )
    condition_type: Literal["context", "context_expression"] = Field(
        default="context",
        description="How AG2 should evaluate this after-work condition",
    )
    available: str | None = Field(
        default=None,
        description="Optional context gate for this after-work condition",
    )
    available_type: AvailableKind = Field(default="context")


class AgentSpec(BaseModel):
    """Specification for a single AG2 agent in a GroupChat."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Agent name, e.g. 'Researcher'")
    role: str = Field(default="", description="Short role description")
    system_message: str | list | None = Field(
        default="",
        description="Full system prompt for the agent",
    )
    description: str | None = Field(
        default=None,
        description="Description used by AG2 group-manager speaker selection",
    )

    # ConversableAgent controls
    llm_config: Any | None = Field(
        default=None,
        description="Per-agent LLM config; falls back to endpoint llm_config",
    )
    human_input_mode: HumanInputMode = Field(default="NEVER")
    max_consecutive_auto_reply: int | None = Field(default=None, ge=0)
    default_auto_reply: str | dict[str, Any] = Field(default="")
    silent: bool | None = Field(default=None)
    code_execution_config: dict[str, Any] | Literal[False] | None = Field(default=None)
    context_variables: dict[str, Any] = Field(default_factory=dict)

    # Tool/function controls. Tool names are resolved from tool_registry.
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names from the registry this agent can invoke",
    )
    function_map: dict[str, Any] = Field(
        default_factory=dict,
        description="Function map values can be callables or tool_registry names",
    )
    functions: list[Any] = Field(
        default_factory=list,
        description="Callable objects or tool_registry names passed to AG2",
    )

    # Transition controls.
    handoffs: list[HandoffCondition] = Field(
        default_factory=list,
        description="LLM or context conditions for handing off to targets",
    )
    after_work: str | None = Field(
        default=None,
        description="Fallback target after this agent finishes its reply",
    )
    after_work_conditions: list[AfterWorkCondition] = Field(
        default_factory=list,
        description="Context-based after-work transitions",
    )

    # Remote/dynamic-agent controls.
    nlip: NlipRemoteAgentSpec | None = Field(
        default=None,
        description="Full NLIP remote-agent config",
    )
    nlip_url: str | None = Field(
        default=None,
        description=(
            "If set, this agent is a remote NLIP agent at the given URL. "
            "system_message and llm_config are handled by the remote server."
        ),
    )
    nlip_timeout: float | None = Field(default=None, gt=0)
    nlip_max_retries: int | None = Field(default=None, ge=1)
    nlip_silent: bool | None = Field(default=None)
    nlip_client_tools: list[dict[str, Any]] = Field(default_factory=list)
    state_template: str | None = Field(
        default=None,
        description="Dynamic system message template using context variables",
    )
    state_templates: list[str] = Field(
        default_factory=list,
        description="Additional UpdateSystemMessage templates",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "nlip" in data and isinstance(data["nlip"], dict):
            nlip = dict(data["nlip"])
            nlip.setdefault("name", data.get("name", "remote"))
            if data.get("role") and "role" not in nlip:
                nlip["role"] = data["role"]
            if data.get("description") and "description" not in nlip:
                nlip["description"] = data["description"]
            data["nlip"] = nlip
            if "nlip_url" not in data and "url" in nlip:
                data["nlip_url"] = nlip["url"]
        if "handoff_conditions" in data and "handoffs" not in data:
            data["handoffs"] = data.pop("handoff_conditions")
        return data


def _normalize_agent_config(value: Any) -> Any:
    if isinstance(value, AgentSpec):
        return value
    if hasattr(value, "as_agent_config") and callable(value.as_agent_config):
        return value.as_agent_config()
    if hasattr(value, "to_agent_config") and callable(value.to_agent_config):
        return value.to_agent_config()
    return value


class GroupChatSpec(BaseModel):
    """Specification for an AG2 GroupChat team."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="endpoint_chat", description="GroupChat identifier")
    objective: str = Field(default="", description="What the team should accomplish")
    agents: list[AgentSpec] = Field(default_factory=list)
    max_round: int = Field(default=15, gt=0, description="Maximum rounds")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial context variables shared across agents",
    )

    # Pattern/orchestration controls.
    pattern: GroupChatPattern = Field(default="default")
    initial_agent: str | None = Field(
        default=None,
        description="Agent name that should speak first",
    )
    group_after_work: str | None = Field(
        default=None,
        description="Group-level fallback target when no agent target is chosen",
    )
    group_manager_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to AG2 GroupChatManager",
    )
    exclude_transit_message: bool = Field(default=True)
    summary_method: Any = Field(default="last_msg")
    selection_message: str | dict[str, Any] | None = Field(
        default=None,
        description="AutoPattern group-manager speaker-selection message",
    )
    mechanical_fallback: bool = Field(
        default=True,
        description="For default pattern, route each agent to the next by order",
    )
    terminate_after_last: bool = Field(
        default=True,
        description="For default pattern, terminate after the final agent",
    )

    # User proxy controls.
    user_name: str = Field(default="User")
    user_human_input_mode: HumanInputMode = Field(default="NEVER")
    user_system_message: str | list | None = Field(
        default="You are a user proxy for the group chat."
    )
    user_default_auto_reply: str | dict[str, Any] = Field(default="")
    user_description: str | None = Field(default=None)
    user_silent: bool | None = Field(default=None)
    user_code_execution_config: dict[str, Any] | Literal[False] | None = Field(
        default=None
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "max_rounds" in data and "max_round" not in data:
            data["max_round"] = data.pop("max_rounds")
        if "context_variables" in data and "context" not in data:
            data["context"] = data.pop("context_variables")
        if "agent_configs" in data and "agents" not in data:
            data["agents"] = data.pop("agent_configs")
        if "agents" in data and isinstance(data["agents"], list):
            data["agents"] = [_normalize_agent_config(i) for i in data["agents"]]
        return data


class ResearchPlan(BaseModel):
    """Full research plan generated by LLM via structured output."""

    topic: str = Field(description="The research topic")
    hypothesis: str = Field(description="Central hypothesis to test")
    group_chats: list[GroupChatSpec] = Field(
        description="Independent GroupChat teams running in parallel",
    )
    synthesis_instruction: str = Field(
        description="How to combine all GroupChat outputs",
    )
    expected_output: str = Field(
        description="What the final deliverable should look like",
    )
    safeguard_policy: list[str] = Field(
        default_factory=list,
        description="Research governance guidelines enforced at conversation level",
    )


class AG2GroupChatRequest(BaseModel):
    """Configuration + prompt for AG2 GroupChat execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Prompt/history
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str = Field(default="")

    # GroupChatSpec-compatible fields
    name: str = Field(default="endpoint_chat")
    objective: str = Field(default="")
    agents: list[AgentSpec] = Field(default_factory=list)
    max_round: int = Field(default=15, gt=0)
    context_variables: dict[str, Any] = Field(default_factory=dict)
    pattern: GroupChatPattern = Field(default="default")
    initial_agent: str | None = Field(default=None)
    group_after_work: str | None = Field(default=None)
    group_manager_args: dict[str, Any] = Field(default_factory=dict)
    exclude_transit_message: bool = Field(default=True)
    summary_method: Any = Field(default="last_msg")
    selection_message: str | dict[str, Any] | None = Field(default=None)
    mechanical_fallback: bool = Field(default=True)
    terminate_after_last: bool = Field(default=True)

    # User proxy controls
    user_name: str = Field(default="User")
    user_human_input_mode: HumanInputMode = Field(default="NEVER")
    user_system_message: str | list | None = Field(
        default="You are a user proxy for the group chat."
    )
    user_default_auto_reply: str | dict[str, Any] = Field(default="")
    user_description: str | None = Field(default=None)
    user_silent: bool | None = Field(default=None)
    user_code_execution_config: dict[str, Any] | Literal[False] | None = Field(
        default=None
    )

    # Runtime SDK controls
    llm_config: Any | None = Field(default=None)
    safeguard_policy: dict[str, Any] | str | None = Field(default=None)
    safeguard_llm_config: Any | None = Field(default=None)
    mask_llm_config: Any | None = Field(default=None)
    yield_on: list[Any] | None = Field(
        default=None,
        description="AG2 event classes or event aliases to yield",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "max_rounds" in data and "max_round" not in data:
            data["max_round"] = data.pop("max_rounds")
        if "context" in data and "context_variables" not in data:
            data["context_variables"] = data.pop("context")
        if "agent_configs" in data and "agents" not in data:
            data["agents"] = data.pop("agent_configs")
        if "agents" in data and isinstance(data["agents"], list):
            data["agents"] = [_normalize_agent_config(i) for i in data["agents"]]
        return data

    @model_validator(mode="after")
    def _derive_prompt_and_objective(self):
        if not self.prompt and self.messages:
            self.prompt = _prompt_from_messages(self.messages)
        if not self.objective:
            self.objective = self.prompt
        return self

    def run_messages(self) -> list[dict[str, Any]] | str:
        """Return AG2-compatible initial messages without losing chat history."""

        return self.messages if self.messages else self.prompt

    def to_group_chat_spec(
        self,
        *,
        agent_configs: list[dict[str, Any]] | None = None,
    ) -> GroupChatSpec:
        """Build a GroupChatSpec from request fields and runtime agent configs."""

        agents: list[AgentSpec]
        if self.agents:
            agents = self.agents
        else:
            agents = [
                AgentSpec.model_validate(_normalize_agent_config(i))
                for i in agent_configs or []
            ]

        return GroupChatSpec(
            name=self.name,
            objective=self.objective or self.prompt,
            agents=agents,
            max_round=self.max_round,
            context=self.context_variables,
            pattern=self.pattern,
            initial_agent=self.initial_agent,
            group_after_work=self.group_after_work,
            group_manager_args=self.group_manager_args,
            exclude_transit_message=self.exclude_transit_message,
            summary_method=self.summary_method,
            selection_message=self.selection_message,
            mechanical_fallback=self.mechanical_fallback,
            terminate_after_last=self.terminate_after_last,
            user_name=self.user_name,
            user_human_input_mode=self.user_human_input_mode,
            user_system_message=self.user_system_message,
            user_default_auto_reply=self.user_default_auto_reply,
            user_description=self.user_description,
            user_silent=self.user_silent,
            user_code_execution_config=self.user_code_execution_config,
        )


# ---------------------------------------------------------------------------
# Callback types (parallel to claude_code.py on_* handlers)
# ---------------------------------------------------------------------------

AG2_HANDLER_PARAMS = (
    "on_text",
    "on_tool_use",
    "on_tool_result",
    "on_speaker",
    "on_complete",
)


# ---------------------------------------------------------------------------
# Runner (in-process — no subprocess, unlike claude_code.py)
# ---------------------------------------------------------------------------


def build_group_chat(
    spec: GroupChatSpec,
    llm_config: Any,
    tool_registry: dict[str, Callable] | None = None,
    code_executor: Any | None = None,
):
    """Build AG2 agents and an orchestration Pattern from a GroupChatSpec.

    Returns (user_proxy, pattern, agents_by_name).
    """

    from autogen import ConversableAgent, register_function
    from autogen.agentchat.conversable_agent import UpdateSystemMessage
    from autogen.agentchat.group import ContextVariables
    from autogen.agentchat.group.patterns import (
        AutoPattern,
        DefaultPattern,
        ManualPattern,
        RandomPattern,
        RoundRobinPattern,
    )

    tool_registry = tool_registry or {}
    user_configs: dict[str, Any] = {
        "name": spec.user_name,
        "human_input_mode": spec.user_human_input_mode,
        "llm_config": False,
        "system_message": spec.user_system_message,
        "default_auto_reply": spec.user_default_auto_reply,
    }
    if spec.user_description is not None:
        user_configs["description"] = spec.user_description
    if spec.user_silent is not None:
        user_configs["silent"] = spec.user_silent
    if spec.user_code_execution_config is not None:
        user_configs["code_execution_config"] = spec.user_code_execution_config
    elif code_executor:
        user_configs["code_execution_config"] = {"executor": code_executor}

    user = ConversableAgent(**user_configs)
    agents_by_name: dict[str, ConversableAgent] = {}
    ordered: list[ConversableAgent] = []

    for agent_spec in spec.agents:
        if agent_spec.nlip_url or agent_spec.nlip is not None:
            agent = _build_nlip_agent(agent_spec, llm_config)
        else:
            kwargs: dict[str, Any] = {
                "name": agent_spec.name,
                "system_message": agent_spec.system_message
                or f"You are {agent_spec.name}.",
                "llm_config": (
                    agent_spec.llm_config
                    if agent_spec.llm_config is not None
                    else llm_config
                ),
                "human_input_mode": agent_spec.human_input_mode,
                "default_auto_reply": agent_spec.default_auto_reply,
            }
            if agent_spec.description or agent_spec.role:
                kwargs["description"] = agent_spec.description or agent_spec.role
            if agent_spec.max_consecutive_auto_reply is not None:
                kwargs["max_consecutive_auto_reply"] = (
                    agent_spec.max_consecutive_auto_reply
                )
            if agent_spec.silent is not None:
                kwargs["silent"] = agent_spec.silent
            if agent_spec.code_execution_config is not None:
                kwargs["code_execution_config"] = agent_spec.code_execution_config
            if agent_spec.context_variables:
                kwargs["context_variables"] = ContextVariables(
                    data=agent_spec.context_variables
                )

            state_templates = [
                i for i in [agent_spec.state_template, *agent_spec.state_templates] if i
            ]
            if state_templates:
                kwargs["update_agent_state_before_reply"] = [
                    UpdateSystemMessage(i) for i in state_templates
                ]

            function_map = _resolve_function_map(agent_spec.function_map, tool_registry)
            if function_map:
                kwargs["function_map"] = function_map

            functions = _resolve_functions(agent_spec.functions, tool_registry)
            if functions:
                kwargs["functions"] = functions

            agent = ConversableAgent(**kwargs)

            for tool_name in agent_spec.tools:
                if tool_name in tool_registry:
                    fn = tool_registry[tool_name]
                    desc = getattr(fn, "__doc__", "") or tool_name
                    try:
                        register_function(
                            fn,
                            caller=agent,
                            executor=user,
                            name=tool_name,
                            description=desc,
                        )
                    except AssertionError as exc:
                        logger.warning(
                            "Tool %r for agent %r could not be registered: %s",
                            tool_name,
                            agent_spec.name,
                            exc,
                        )
                else:
                    logger.warning(
                        "Tool %r for agent %r not found in registry; skipped",
                        tool_name,
                        agent_spec.name,
                    )

        agents_by_name[agent_spec.name] = agent
        ordered.append(agent)

    _install_agent_transitions(spec, agents_by_name, ordered, user)

    initial_agent = (
        agents_by_name.get(spec.initial_agent) if spec.initial_agent else None
    )
    if spec.initial_agent and initial_agent is None:
        logger.warning(
            "Initial agent %r not found; using first agent", spec.initial_agent
        )
    initial_agent = initial_agent or (ordered[0] if ordered else user)

    group_after_work = _resolve_transition_target(
        spec.group_after_work, agents_by_name, ordered, user
    )
    pattern_kwargs: dict[str, Any] = {
        "initial_agent": initial_agent,
        "agents": ordered,
        "user_agent": user,
        "group_manager_args": spec.group_manager_args,
        "context_variables": ContextVariables(data=spec.context),
        "exclude_transit_message": spec.exclude_transit_message,
        "summary_method": spec.summary_method,
    }

    if spec.pattern == "auto":
        pattern_kwargs["selection_message"] = _build_selection_message(
            spec.selection_message
        )
        pattern = AutoPattern(**pattern_kwargs)
    elif spec.pattern == "round_robin":
        pattern = RoundRobinPattern(
            **pattern_kwargs,
            group_after_work=group_after_work,
        )
    elif spec.pattern == "random":
        pattern = RandomPattern(**pattern_kwargs, group_after_work=group_after_work)
    elif spec.pattern == "manual":
        pattern = ManualPattern(**pattern_kwargs)
    else:
        pattern = DefaultPattern(
            **pattern_kwargs,
            group_after_work=group_after_work,
        )

    return user, pattern, agents_by_name


async def stream_group_chat(
    pattern,
    prompt: str | list[dict[str, Any]],
    max_rounds: int = 15,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: Any | None = None,
    mask_llm_config: Any | None = None,
    yield_on: list[Any] | None = None,
    on_text: Callable[[str, str], None] | None = None,
    on_tool_use: Callable[[str, str, Any], None] | None = None,
    on_tool_result: Callable[[str, Any], None] | None = None,
    on_speaker: Callable[[str], None] | None = None,
    on_complete: Callable[[Any], None] | None = None,
) -> AsyncIterator[Any]:
    """Stream AG2 GroupChat events. In-process, no subprocess."""

    from autogen.agentchat.group.multi_agent_chat import a_run_group_chat_iter
    from autogen.events.agent_events import (
        GroupChatRunChatEvent,
        SelectSpeakerEvent,
        TextEvent,
        ToolCallEvent,
        ToolResponseEvent,
    )

    if yield_on is None:
        yield_on = [
            TextEvent,
            ToolCallEvent,
            ToolResponseEvent,
            SelectSpeakerEvent,
            GroupChatRunChatEvent,
        ]
    else:
        yield_on = _resolve_yield_events(yield_on)

    kwargs: dict[str, Any] = {
        "pattern": pattern,
        "messages": prompt,
        "max_rounds": max_rounds,
        "yield_on": yield_on,
    }
    if safeguard_policy is not None:
        kwargs["safeguard_policy"] = safeguard_policy
    if safeguard_llm_config is not None:
        kwargs["safeguard_llm_config"] = safeguard_llm_config
    if mask_llm_config is not None:
        kwargs["mask_llm_config"] = mask_llm_config

    try:
        async for event in a_run_group_chat_iter(**kwargs):
            inner = getattr(event, "content", None)

            if isinstance(event, TextEvent) and on_text:
                text = getattr(inner, "content", "") if inner is not None else ""
                sender = (
                    getattr(inner, "sender", "unknown")
                    if inner is not None
                    else "unknown"
                )
                await _maybe_await(on_text, text, sender)
            elif isinstance(event, ToolCallEvent) and on_tool_use:
                tool_calls = (
                    getattr(inner, "tool_calls", []) if inner is not None else []
                )
                first = tool_calls[0] if tool_calls else None
                tool_name = (
                    getattr(getattr(first, "function", None), "name", "")
                    if first
                    else ""
                )
                tool_args = (
                    getattr(getattr(first, "function", None), "arguments", None)
                    if first
                    else None
                )
                sender = (
                    getattr(inner, "sender", "unknown")
                    if inner is not None
                    else "unknown"
                )
                await _maybe_await(on_tool_use, tool_name, sender, tool_args)
            elif isinstance(event, ToolResponseEvent) and on_tool_result:
                tool_responses = (
                    getattr(inner, "tool_responses", []) if inner is not None else []
                )
                first = tool_responses[0] if tool_responses else None
                tool_output = getattr(first, "content", None) if first else None
                sender = (
                    getattr(inner, "sender", "unknown")
                    if inner is not None
                    else "unknown"
                )
                await _maybe_await(on_tool_result, sender, tool_output)
            elif isinstance(event, GroupChatRunChatEvent) and on_speaker:
                speaker = (
                    getattr(inner, "speaker", "unknown")
                    if inner is not None
                    else "unknown"
                )
                await _maybe_await(on_speaker, speaker)
            elif isinstance(event, SelectSpeakerEvent) and on_speaker:
                agents = getattr(inner, "agents", []) if inner is not None else []
                first_name = (
                    getattr(agents[0], "name", str(agents[0])) if agents else "unknown"
                )
                await _maybe_await(on_speaker, first_name)
            yield event
    except Exception:
        logger.exception("AG2 GroupChat streaming failed")
        raise

    if on_complete:
        await _maybe_await(on_complete, None)


def _install_agent_transitions(
    spec: GroupChatSpec,
    agents_by_name: dict[str, Any],
    ordered: list[Any],
    user: Any,
) -> None:
    for idx, agent_spec in enumerate(spec.agents):
        agent = agents_by_name[agent_spec.name]

        for hc in agent_spec.handoffs:
            target = _resolve_transition_target(
                hc.target, agents_by_name, ordered, user
            )
            if target is None:
                _warn_missing_target(hc.target, agent_spec.name)
                continue
            condition = _build_handoff_condition(hc, target)
            if hc.condition_type == "llm":
                agent.handoffs.add_llm_condition(condition)
            else:
                agent.handoffs.add_context_condition(condition)

        for aw in agent_spec.after_work_conditions:
            target = _resolve_transition_target(
                aw.target, agents_by_name, ordered, user
            )
            if target is None:
                _warn_missing_target(aw.target, agent_spec.name)
                continue
            agent.handoffs.add_after_work(_build_after_work_condition(aw, target))

        explicit_after_work = _resolve_transition_target(
            agent_spec.after_work, agents_by_name, ordered, user
        )
        if explicit_after_work is not None:
            agent.handoffs.add_after_work(
                _build_after_work_condition(
                    AfterWorkCondition(target=agent_spec.after_work or "terminate"),
                    explicit_after_work,
                )
            )
            continue

        if not spec.mechanical_fallback or spec.pattern != "default":
            continue

        has_unconditional_after_work = any(
            i.condition is None for i in agent.handoffs.after_works
        )
        if has_unconditional_after_work:
            continue

        if idx < len(spec.agents) - 1:
            next_target = _resolve_transition_target(
                spec.agents[idx + 1].name,
                agents_by_name,
                ordered,
                user,
            )
            agent.handoffs.add_after_work(
                _build_after_work_condition(
                    AfterWorkCondition(target=spec.agents[idx + 1].name),
                    next_target,
                )
            )
        elif spec.terminate_after_last:
            terminate_target = _resolve_transition_target(
                "terminate", agents_by_name, ordered, user
            )
            agent.handoffs.add_after_work(
                _build_after_work_condition(
                    AfterWorkCondition(target="terminate"),
                    terminate_target,
                )
            )


def _build_handoff_condition(spec: HandoffCondition, target: Any):
    from autogen.agentchat.group import (
        ContextExpression,
        ExpressionContextCondition,
        OnCondition,
        OnContextCondition,
        StringContextCondition,
        StringLLMCondition,
    )

    available = _build_available_condition(spec.available, spec.available_type)
    if spec.condition_type == "llm":
        return OnCondition(
            target=target,
            condition=StringLLMCondition(prompt=spec.condition or ""),
            available=available,
            llm_function_name=spec.llm_function_name,
        )
    if spec.condition_type == "context_expression" and spec.condition:
        condition = ExpressionContextCondition(ContextExpression(spec.condition))
    elif spec.condition:
        condition = StringContextCondition(variable_name=spec.condition)
    else:
        condition = None
    return OnContextCondition(target=target, condition=condition, available=available)


def _build_after_work_condition(spec: AfterWorkCondition, target: Any):
    from autogen.agentchat.group import (
        ContextExpression,
        ExpressionContextCondition,
        OnContextCondition,
        StringContextCondition,
    )

    available = _build_available_condition(spec.available, spec.available_type)
    if spec.condition_type == "context_expression" and spec.condition:
        condition = ExpressionContextCondition(ContextExpression(spec.condition))
    elif spec.condition:
        condition = StringContextCondition(variable_name=spec.condition)
    else:
        condition = None
    return OnContextCondition(target=target, condition=condition, available=available)


def _build_available_condition(value: str | None, kind: AvailableKind):
    if not value:
        return None
    from autogen.agentchat.group import (
        ContextExpression,
        ExpressionAvailableCondition,
        StringAvailableCondition,
    )

    if kind == "context_expression":
        return ExpressionAvailableCondition(ContextExpression(value))
    return StringAvailableCondition(value)


def _resolve_transition_target(
    value: str | Any | None,
    agents_by_name: dict[str, Any],
    ordered: list[Any],
    user: Any,
):
    if value is None:
        return None

    from autogen.agentchat.group import (
        AgentTarget,
        AskUserTarget,
        RevertToUserTarget,
        StayTarget,
        TerminateTarget,
    )
    from autogen.agentchat.group.targets.transition_target import (
        RandomAgentTarget,
        TransitionTarget,
    )

    if isinstance(value, TransitionTarget):
        return value

    if not isinstance(value, str):
        return value

    normalized = value.lower()
    if normalized in {"terminate", "termination", "end", "stop"}:
        return TerminateTarget()
    if normalized in {"revert_to_user", "user", "user_proxy"}:
        return RevertToUserTarget()
    if normalized == "stay":
        return StayTarget()
    if normalized in {"ask_user", "manual"}:
        return AskUserTarget()
    if normalized == "random":
        return RandomAgentTarget(ordered)
    if normalized in {"group_manager", "auto"}:
        from autogen.agentchat.group.targets.group_manager_target import (
            GroupManagerTarget,
        )

        return GroupManagerTarget()
    if value == getattr(user, "name", None):
        return AgentTarget(user)
    if value in agents_by_name:
        return AgentTarget(agents_by_name[value])
    return None


def _build_selection_message(value: str | dict[str, Any] | None):
    if value is None:
        return None
    from autogen.agentchat.group.targets.group_manager_target import (
        GroupManagerSelectionMessageContextStr,
        GroupManagerSelectionMessageString,
    )

    if isinstance(value, str):
        return GroupManagerSelectionMessageString(message=value)

    kind = value.get("type", "string")
    message = value.get("message") or value.get("template") or ""
    if kind in {"context", "context_str", "context_template"}:
        return GroupManagerSelectionMessageContextStr(context_str_template=message)
    return GroupManagerSelectionMessageString(message=message)


def _build_nlip_agent(agent_spec: AgentSpec, llm_config: Any):
    from ..nlip.models import NlipRemoteAgentSpec, build_nlip_remote_agent

    if agent_spec.nlip is not None:
        spec = agent_spec.nlip.model_copy(
            update={
                "name": agent_spec.nlip.name or agent_spec.name,
                "role": agent_spec.nlip.role or agent_spec.role or "remote agent",
                "description": agent_spec.nlip.description
                or agent_spec.description
                or agent_spec.role
                or None,
            }
        )
    else:
        spec = NlipRemoteAgentSpec(
            url=agent_spec.nlip_url or "",
            name=agent_spec.name,
            role=agent_spec.role or "remote agent",
            description=agent_spec.description or agent_spec.role or None,
            silent=agent_spec.nlip_silent,
            timeout=agent_spec.nlip_timeout or 60.0,
            max_retries=agent_spec.nlip_max_retries or 3,
            context_variables=agent_spec.context_variables,
            client_tools=agent_spec.nlip_client_tools,
        )
    return build_nlip_remote_agent(spec)


def _resolve_function_map(
    function_map: dict[str, Any],
    tool_registry: dict[str, Callable],
) -> dict[str, Callable]:
    resolved: dict[str, Callable] = {}
    for name, value in function_map.items():
        if callable(value):
            resolved[name] = value
        elif isinstance(value, str) and value in tool_registry:
            resolved[name] = tool_registry[value]
        else:
            logger.warning("Function map entry %r could not be resolved; skipped", name)
    return resolved


def _resolve_functions(
    values: list[Any],
    tool_registry: dict[str, Callable],
) -> list[Callable]:
    resolved: list[Callable] = []
    for value in values:
        if callable(value):
            resolved.append(value)
        elif isinstance(value, str) and value in tool_registry:
            resolved.append(tool_registry[value])
        else:
            logger.warning("Function %r could not be resolved; skipped", value)
    return resolved


def _resolve_yield_events(values: list[Any]) -> list[type]:
    from autogen.events import agent_events

    aliases = {
        "text": "TextEvent",
        "tool_use": "ToolCallEvent",
        "tool_call": "ToolCallEvent",
        "tool_result": "ToolResponseEvent",
        "speaker": "GroupChatRunChatEvent",
        "speaker_turn": "GroupChatRunChatEvent",
        "select_speaker": "SelectSpeakerEvent",
        "completion": "RunCompletionEvent",
        "result": "RunCompletionEvent",
        "error": "ErrorEvent",
        "termination": "TerminationEvent",
    }

    resolved: list[type] = []
    for value in values:
        if isinstance(value, type):
            resolved.append(value)
            continue
        if not isinstance(value, str):
            logger.warning("Unknown AG2 yield_on value %r; skipped", value)
            continue
        event_name = aliases.get(value, value)
        if not event_name.endswith("Event"):
            event_name = f"{event_name}Event"
        event_cls = getattr(agent_events, event_name, None)
        if isinstance(event_cls, type):
            resolved.append(event_cls)
        else:
            logger.warning("Unknown AG2 event %r; skipped", value)
    return resolved


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_to_text(i) for i in content]
        return "\n".join(i for i in parts if i)
    if isinstance(content, dict):
        if "text" in content:
            return _content_to_text(content["text"])
        if "content" in content:
            return _content_to_text(content["content"])
    return str(content) if content is not None else ""


def _warn_missing_target(target: str, source: str) -> None:
    logger.warning(
        "Handoff target %r from agent %r not found in spec; skipped",
        target,
        source,
    )


async def _maybe_await(fn: Callable, *args: Any) -> Any:
    from lionagi.ln.concurrency.utils import is_coro_func

    if is_coro_func(fn):
        return await fn(*args)
    return fn(*args)
