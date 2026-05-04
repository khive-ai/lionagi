# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "NlipRemoteAgentSpec",
    "AG2NlipRequest",
    "build_nlip_remote_agent",
    "nlip_agent_config",
    "call_nlip_remote",
]


class NlipRemoteAgentSpec(BaseModel):
    """Reusable config for an AG2 NLIP remote ConversableAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = Field(description="Base URL of the remote NLIP server")
    name: str = Field(default="remote", description="Local AG2 agent name")
    role: str = Field(default="remote agent")
    description: str | None = Field(default=None)
    silent: bool | None = Field(default=None)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=1)
    context_variables: dict[str, Any] = Field(default_factory=dict)
    client_tools: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "agent_name" in data and "name" not in data:
            data["name"] = data.pop("agent_name")
        if "context" in data and "context_variables" not in data:
            data["context_variables"] = data.pop("context")
        return data

    def to_agent_config(self, **overrides: Any) -> dict[str, Any]:
        """Return a GroupChat AgentSpec-compatible remote-agent config."""

        spec = self.model_copy(
            update={k: v for k, v in overrides.items() if v is not None}
        )
        config: dict[str, Any] = {
            "name": spec.name,
            "role": spec.role,
            "nlip_url": spec.url,
            "nlip_timeout": spec.timeout,
            "nlip_max_retries": spec.max_retries,
        }
        if spec.description is not None:
            config["description"] = spec.description
        if spec.silent is not None:
            config["nlip_silent"] = spec.silent
        if spec.context_variables:
            config["context_variables"] = spec.context_variables
        if spec.client_tools:
            config["nlip_client_tools"] = spec.client_tools
        return config


class AG2NlipRequest(BaseModel):
    """Configuration + prompt for an AG2 NLIP remote call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str = ""
    url: str | None = None
    agent_name: str | None = None
    timeout: float | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=1)
    silent: bool | None = None
    context_variables: dict[str, Any] = Field(default_factory=dict)
    client_tools: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "name" in data and "agent_name" not in data:
            data["agent_name"] = data.pop("name")
        if "context" in data and "context_variables" not in data:
            data["context_variables"] = data.pop("context")
        return data

    @model_validator(mode="after")
    def _derive_prompt(self):
        if not self.prompt and self.messages:
            self.prompt = _prompt_from_messages(self.messages)
        return self

    def messages_for_call(self) -> list[dict[str, Any]]:
        return self.messages or [{"role": "user", "content": self.prompt}]

    def to_agent_spec(
        self,
        *,
        url: str | None = None,
        agent_name: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        silent: bool | None = None,
        role: str = "remote agent",
        description: str | None = None,
    ) -> NlipRemoteAgentSpec:
        """Return a reusable remote-agent spec for GroupChat composition."""

        resolved_url = url or self.url
        if not resolved_url:
            raise ValueError("AG2NlipRequest requires a url to build an agent spec")
        return NlipRemoteAgentSpec(
            url=resolved_url,
            name=agent_name or self.agent_name or "remote",
            role=role,
            description=description,
            silent=self.silent if silent is None else silent,
            timeout=timeout or self.timeout or 60.0,
            max_retries=max_retries or self.max_retries or 3,
            context_variables=self.context_variables,
            client_tools=self.client_tools,
        )


def nlip_agent_config(
    url: str,
    *,
    name: str = "remote",
    role: str = "remote agent",
    description: str | None = None,
    silent: bool | None = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    context_variables: dict[str, Any] | None = None,
    client_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convenience helper for creating a GroupChat AgentSpec config."""

    return NlipRemoteAgentSpec(
        url=url,
        name=name,
        role=role,
        description=description,
        silent=silent,
        timeout=timeout,
        max_retries=max_retries,
        context_variables=context_variables or {},
        client_tools=client_tools or [],
    ).to_agent_config()


def build_nlip_remote_agent(
    spec: NlipRemoteAgentSpec | dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Build an AG2-compatible NLIP remote ConversableAgent.

    Uses AG2's NlipRemoteAgent when the optional NLIP extras are installed and
    falls back to a lightweight ConversableAgent that calls ``call_nlip_remote``.
    """

    if spec is None:
        spec_obj = NlipRemoteAgentSpec(**kwargs)
    elif isinstance(spec, dict):
        spec_obj = NlipRemoteAgentSpec(**{**spec, **kwargs})
    elif kwargs:
        spec_obj = spec.model_copy(update=kwargs)
    else:
        spec_obj = spec

    try:
        from autogen.agentchat.contrib.nlip_agent import NlipRemoteAgent

        agent = NlipRemoteAgent(
            url=spec_obj.url,
            name=spec_obj.name,
            silent=spec_obj.silent,
            timeout=spec_obj.timeout,
            max_retries=spec_obj.max_retries,
        )
    except ImportError:
        agent = _FallbackNlipRemoteAgent(spec_obj)

    if spec_obj.description or spec_obj.role:
        agent.description = spec_obj.description or spec_obj.role
    if spec_obj.context_variables:
        agent.context_variables.update(spec_obj.context_variables)
    return agent


async def call_nlip_remote(
    url: str,
    messages: list[dict[str, Any]],
    agent_name: str = "remote",
    timeout: float = 60.0,
    max_retries: int = 3,
    context: dict[str, Any] | None = None,
    client_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call a remote NLIP endpoint using AG2-compatible NLIP wire format."""

    try:
        return await _call_nlip_sdk(
            url,
            messages,
            timeout,
            max_retries,
            context=context,
            client_tools=client_tools,
        )
    except ImportError:
        logger.info("nlip_sdk not installed, using direct HTTP")
        return await _call_direct(
            url,
            messages,
            timeout,
            max_retries,
            context=context,
            client_tools=client_tools,
        )


async def _call_nlip_sdk(
    url: str,
    messages: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
    context: dict[str, Any] | None = None,
    client_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Use AG2's NLIP converters for the remote wire format."""

    import httpx
    from autogen.agentchat.contrib.nlip_agent import (
        request_message_to_nlip,
        response_message_from_nlip,
    )
    from autogen.agentchat.remote import RequestMessage
    from nlip_sdk.nlip import NLIP_Message

    nlip_msg = request_message_to_nlip(
        RequestMessage(
            messages=messages,
            context=context or {},
            client_tools=client_tools or [],
        )
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{url.rstrip('/')}/nlip/",
                    json=nlip_msg.model_dump(exclude_none=True),
                )
                response.raise_for_status()

                nlip_response = NLIP_Message.model_validate(response.json())
                response_msg = response_message_from_nlip(nlip_response)
                content = ""
                if response_msg.messages:
                    content = response_msg.messages[-1].get("content", "")

                return {
                    "content": content,
                    "context": response_msg.context,
                    "input_required": response_msg.input_required,
                }

            except httpx.TimeoutException:
                if attempt == max_retries - 1:
                    raise
                logger.warning("NLIP timeout (attempt %d/%d)", attempt + 1, max_retries)
            except httpx.ConnectError:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "NLIP connect failed (attempt %d/%d)", attempt + 1, max_retries
                )

    return {"content": "", "context": None, "input_required": None}


async def _call_direct(
    url: str,
    messages: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
    context: dict[str, Any] | None = None,
    client_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Direct HTTP fallback when nlip_sdk is not installed."""

    import httpx

    payload: dict[str, Any] = {
        "format": "text",
        "subformat": "english",
        "content": _prompt_from_messages(messages),
    }
    if context:
        payload["context"] = context
    if client_tools:
        payload["client_tools"] = client_tools

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{url.rstrip('/')}/nlip/",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                content = ""
                response_context = None
                input_required = None
                if isinstance(data, dict):
                    content = data.get("content", "")
                    if isinstance(content, dict):
                        content = content.get("content", str(content))
                    response_context = data.get("context")
                    input_required = data.get("input_required")

                return {
                    "content": content,
                    "context": response_context,
                    "input_required": input_required,
                }

            except httpx.TimeoutException:
                if attempt == max_retries - 1:
                    raise
            except httpx.ConnectError:
                if attempt == max_retries - 1:
                    raise

    return {"content": "", "context": None, "input_required": None}


class _FallbackNlipRemoteAgent:
    """Factory for a remote ConversableAgent fallback."""

    def __new__(cls, spec: NlipRemoteAgentSpec):
        from autogen import ConversableAgent

        class LionNlipRemoteAgent(ConversableAgent):
            def __init__(self, agent_spec: NlipRemoteAgentSpec):
                super().__init__(
                    name=agent_spec.name,
                    llm_config=False,
                    human_input_mode="NEVER",
                    silent=agent_spec.silent,
                    description=agent_spec.description or agent_spec.role,
                )
                self.url = agent_spec.url.rstrip("/")
                self._timeout = agent_spec.timeout
                self._max_retries = agent_spec.max_retries
                self._client_tools = agent_spec.client_tools
                self.replace_reply_func(
                    ConversableAgent.generate_oai_reply,
                    LionNlipRemoteAgent.generate_remote_reply,
                )
                self.replace_reply_func(
                    ConversableAgent.a_generate_oai_reply,
                    LionNlipRemoteAgent.a_generate_remote_reply,
                )

            def generate_remote_reply(self, *_, **__):
                raise NotImplementedError(
                    f"{self.__class__.__name__} only supports async communication."
                )

            async def a_generate_remote_reply(
                self,
                messages: list[dict[str, Any]] | None = None,
                sender: ConversableAgent | None = None,
                config: Any | None = None,
            ) -> tuple[bool, dict[str, Any] | None]:
                if messages is None:
                    messages = self._oai_messages[sender]
                result = await call_nlip_remote(
                    url=self.url,
                    messages=messages,
                    agent_name=self.name,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                    context=self.context_variables.data,
                    client_tools=self._client_tools,
                )
                if result.get("context"):
                    self.context_variables.update(result["context"])
                    if sender:
                        sender.context_variables.update(result["context"])
                return True, {
                    "role": "assistant",
                    "content": result.get("content", ""),
                }

        return LionNlipRemoteAgent(spec)


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        content = _content_to_text(msg.get("content", ""))
        if msg.get("role") != "tool" and content and content != "None":
            return content
    return ""


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
