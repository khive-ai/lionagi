# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Mixin for per-provider config enums.

Tuple format (positional by index)::

    MEMBER = (endpoint, aliases, type, options, base_url, auth_type, content_type)
    # idx:     0        1        2      3        4         5          6

    # Short form (agentic — no HTTP config needed):
    GROUP_CHAT = ("group_chat", ["chat"], EndpointType.AGENTIC, LazyType("...:Request"))

    # Full form (API — HTTP config included):
    CHAT = ("chat/completions", ["chat"], EndpointType.API, LazyType("...:Request"),
            "https://api.openai.com/v1", "bearer", "application/json")

Usage::

    class OpenAIConfigs(ProviderConfig, Enum):
        _ignore_ = ["_PROVIDER", "_PROVIDER_ALIASES"]
        _PROVIDER = "openai"
        _PROVIDER_ALIASES = []

        CHAT = ("chat/completions", ["chat"], EndpointType.API,
                LazyType("lionagi.providers.openai.chat.models:OpenAIChatCompletionsRequest"),
                "https://api.openai.com/v1", "bearer")

    @OpenAIConfigs.CHAT.register
    class OpenaiChatEndpoint(Endpoint): ...
    # config auto-created — no _get_config() needed
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel

from .registry import EndpointType, register_endpoint


class LazyType:
    """Deferred type import. Resolves on first access.

    Usage in config enum::

        CHAT = (..., LazyType("lionagi.providers.openai.chat.models:OpenAIChatCompletionsRequest"))

    Hashable (required for Enum member tuples).
    """

    __slots__ = ("_ref", "_resolved")

    def __init__(self, ref: str) -> None:
        if ":" not in ref:
            raise ValueError(f"LazyType ref must be 'module:Class', got {ref!r}")
        self._ref = ref
        self._resolved: type | None = None

    def resolve(self) -> type[BaseModel]:
        if self._resolved is None:
            module_path, class_name = self._ref.rsplit(":", 1)
            mod = importlib.import_module(module_path)
            self._resolved = getattr(mod, class_name)
        return self._resolved

    def __hash__(self) -> int:
        return hash(self._ref)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LazyType):
            return self._ref == other._ref
        return NotImplemented

    def __repr__(self) -> str:
        status = "resolved" if self._resolved is not None else "pending"
        return f"LazyType({self._ref!r}, {status})"


def _get(value: tuple, idx: int, default=None):
    return value[idx] if len(value) > idx else default


class ProviderConfig:
    """Mixin for provider endpoint Enum classes.

    Concrete classes extend ``(ProviderConfig, Enum)`` and declare
    members as tuples with 4-7 elements (see module docstring).
    """

    # --- Tuple accessors ---

    @property
    def endpoint_path(self) -> str:
        return self.value[0]

    @property
    def aliases(self) -> list[str]:
        return self.value[1]

    @property
    def endpoint_type(self) -> EndpointType:
        return self.value[2]

    @property
    def options(self) -> type[BaseModel] | None:
        raw = _get(self.value, 3)
        if isinstance(raw, LazyType):
            return raw.resolve()
        return raw

    @property
    def base_url(self) -> str | None:
        return _get(self.value, 4)

    @property
    def auth_type(self) -> str | None:
        return _get(self.value, 5)

    @property
    def content_type(self) -> str | None:
        return _get(self.value, 6, "application/json")

    # --- Provider info (set via _ignore_ or post-definition) ---

    @property
    def provider(self) -> str:
        return type(self)._PROVIDER

    @property
    def provider_aliases(self) -> list[str]:
        return type(self)._PROVIDER_ALIASES

    # --- Registry integration ---

    def as_registry_kwargs(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_aliases": self.provider_aliases,
            "endpoint": self.endpoint_path,
            "aliases": self.aliases,
            "endpoint_type": self.endpoint_type,
            "options": self.options,
            "base_url": self.base_url,
            "auth_type": self.auth_type,
            "content_type": self.content_type,
        }

    def register(self, cls=None):
        """Decorator: register an endpoint class for this config member.

        Usage::

            @OpenAIConfigs.CHAT.register
            class OpenaiChatEndpoint(Endpoint): ...
        """
        decorator = register_endpoint(**self.as_registry_kwargs())
        if cls is not None:
            return decorator(cls)
        return decorator

    @classmethod
    def available(cls) -> frozenset[str]:
        return frozenset(m.endpoint_path for m in cls)
