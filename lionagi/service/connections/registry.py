# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel

from lionagi.ln.types import Enum

__all__ = (
    "EndpointType",
    "EndpointMeta",
    "EndpointRegistry",
    "register_endpoint",
)


class EndpointType(Enum):
    API = "api"
    AGENTIC = "agentic"


@dataclass(frozen=True, slots=True)
class EndpointMeta:
    """Injected onto endpoint classes as ``_ENDPOINT_META``.

    Carries enough info to auto-generate ``EndpointConfig`` —
    individual endpoints no longer need ``_get_config()``.
    """

    provider: str
    endpoint: str
    endpoint_type: EndpointType
    aliases: tuple[str, ...] = ()
    provider_aliases: tuple[str, ...] = ()
    options: type[BaseModel] | None = None
    base_url: str | None = None
    auth_type: str | None = None
    content_type: str | None = None

    def create_config(self, **overrides: Any):
        from .endpoint_config import EndpointConfig

        is_agentic = self.endpoint_type == EndpointType.AGENTIC
        defaults = dict(
            name=f"{self.provider}_{self.endpoint}",
            provider=self.provider,
            base_url=self.base_url or ("internal" if is_agentic else ""),
            endpoint=self.endpoint,
            api_key="internal" if is_agentic else None,
            request_options=self.options,
            timeout=3600 if is_agentic else 600,
            auth_type=self.auth_type or ("bearer" if not is_agentic else "bearer"),
            content_type=self.content_type or "application/json",
            method="POST",
        )
        defaults.update(overrides)
        return EndpointConfig(**defaults)


class _RegistryEntry:
    __slots__ = ("meta", "cls")

    def __init__(self, meta: EndpointMeta, cls: type):
        self.meta = meta
        self.cls = cls


class EndpointRegistry:
    _entries: ClassVar[list[_RegistryEntry]] = []
    _loaded: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        provider: str,
        endpoint: str,
        aliases: list[str] | None = None,
        endpoint_type: EndpointType = EndpointType.API,
        provider_aliases: list[str] | None = None,
        options: type[BaseModel] | None = None,
        base_url: str | None = None,
        auth_type: str | None = None,
        content_type: str | None = None,
    ):
        """Decorator that registers an endpoint and injects _ENDPOINT_META."""

        def decorator(endpoint_cls: type) -> type:
            meta = EndpointMeta(
                provider=provider,
                endpoint=endpoint,
                endpoint_type=endpoint_type,
                aliases=tuple(aliases or ()),
                provider_aliases=tuple(provider_aliases or ()),
                options=options,
                base_url=base_url,
                auth_type=auth_type,
                content_type=content_type,
            )
            endpoint_cls._ENDPOINT_META = meta
            cls._entries.append(_RegistryEntry(meta=meta, cls=endpoint_cls))
            return endpoint_cls

        return decorator

    @classmethod
    def match(cls, provider: str, endpoint: str = "", **kwargs) -> Any:
        """Find and instantiate the best matching endpoint."""
        cls._ensure_loaded()

        for entry in cls._entries:
            m = entry.meta
            if not (provider == m.provider or provider in m.provider_aliases):
                continue
            if (
                not endpoint
                or endpoint == m.endpoint
                or endpoint in m.aliases
            ):
                return entry.cls(None, **kwargs)

        from .endpoint import Endpoint, EndpointConfig

        config = EndpointConfig(
            provider=provider,
            endpoint=endpoint or "chat/completions",
            name="openai_compatible_chat",
            auth_type="bearer",
            content_type="application/json",
            method="POST",
            requires_tokens=True,
        )
        return Endpoint(config, **kwargs)

    @classmethod
    def _ensure_loaded(cls):
        if cls._loaded:
            return
        cls._loaded = True
        _import_all_providers()

    @classmethod
    def list_providers(cls) -> list[dict[str, Any]]:
        cls._ensure_loaded()
        return [
            {
                "provider": e.meta.provider,
                "endpoint": e.meta.endpoint,
                "aliases": list(e.meta.aliases),
                "type": e.meta.endpoint_type.value,
                "class": e.cls.__name__,
                "options": e.meta.options.__name__ if e.meta.options else None,
            }
            for e in cls._entries
        ]


def register_endpoint(
    provider: str,
    endpoint: str,
    aliases: list[str] | None = None,
    endpoint_type: EndpointType = EndpointType.API,
    provider_aliases: list[str] | None = None,
    options: type[BaseModel] | None = None,
    base_url: str | None = None,
    auth_type: str | None = None,
    content_type: str | None = None,
):
    """Decorator that registers an endpoint and injects ``_ENDPOINT_META``."""
    return EndpointRegistry.register(
        provider=provider,
        endpoint=endpoint,
        aliases=aliases,
        endpoint_type=endpoint_type,
        provider_aliases=provider_aliases,
        options=options,
        base_url=base_url,
        auth_type=auth_type,
        content_type=content_type,
    )


def _import_all_providers():
    """Import all provider modules to trigger registration decorators."""
    import importlib

    _modules = [
        # OpenAI family
        "lionagi.providers.openai.chat.endpoint",
        "lionagi.providers.openai.codex.endpoint",
        "lionagi.providers.openai.audio.endpoint",
        "lionagi.providers.openai.images.endpoint",
        # Anthropic
        "lionagi.providers.anthropic.messages.endpoint",
        "lionagi.providers.anthropic.claude_code.endpoint",
        # Ollama
        "lionagi.providers.ollama.chat.endpoint",
        "lionagi.providers.ollama.embed.endpoint",
        "lionagi.providers.ollama.generate.endpoint",
        # Search & scraping
        "lionagi.providers.tavily.endpoint",
        "lionagi.providers.exa.endpoint",
        "lionagi.providers.exa.contents.endpoint",
        "lionagi.providers.exa.find_similar.endpoint",
        "lionagi.providers.firecrawl.endpoint",
        "lionagi.providers.firecrawl.crawl.endpoint",
        # Chat / LLM providers
        "lionagi.providers.perplexity.chat.endpoint",
        "lionagi.providers.nvidia_nim.endpoint",
        "lionagi.providers.deepseek.endpoint",
        "lionagi.providers.gemini.gemini_code.endpoint",
        "lionagi.providers.pi.endpoint",
        "lionagi.providers.openrouter.endpoint",
        # Agentic
        "lionagi.providers.ag2.groupchat.endpoint",
    ]
    for mod in _modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
