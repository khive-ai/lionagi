# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI and OpenAI-compatible endpoint configurations.

This module provides endpoint configurations for:
- OpenAI (chat, response, embedding)
- OpenRouter (OpenAI-compatible)
- Groq (OpenAI-compatible)
- Gemini (OpenAI-compatible)

Each provider has a helper function (_get_*_config) that creates
configurations with sensible defaults that can be overridden.
"""

from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)


def _get_oai_config(**kw):
    config = {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "auth_type": "bearer",
        "content_type": "application/json",
        "method": "POST",
        "api_key": settings.OPENAI_API_KEY or "dummy-key-for-testing",
    }
    config.update(kw)
    return EndpointConfig(**config)


def _get_openrouter_config(**kwargs):
    """Create OpenRouter endpoint configuration with defaults."""
    config = dict(
        name="openrouter_chat",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        endpoint="chat/completions",
        kwargs={"model": "google/gemini-2.5-flash"},
        api_key=settings.OPENROUTER_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        request_options=OpenAIChatCompletionsRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


def _get_groq_config(**kwargs):
    """Create Groq endpoint configuration with defaults."""
    config = dict(
        name="groq_chat",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        endpoint="chat/completions",
        kwargs={"model": "llama-3.3-70b-versatile"},
        api_key=settings.GROQ_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        request_options=OpenAIChatCompletionsRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


class OpenaiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_oai_config(
            name="oai_chat",
            endpoint="chat/completions",
            request_options=OpenAIChatCompletionsRequest,
            kwargs={"model": settings.OPENAI_DEFAULT_MODEL},
            requires_tokens=True,
        )
        super().__init__(config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """Override to handle model-specific parameter filtering."""
        payload, headers = super().create_payload(
            request, extra_headers, **kwargs
        )
        # Convert system role to developer role for reasoning models
        if "messages" in payload and payload["messages"]:
            if payload["messages"][0].get("role") == "system":
                payload["messages"][0]["role"] = "developer"

        return (payload, headers)


class OpenaiResponseEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_oai_config(
            name="openai_response",
            endpoint="responses",
            requires_tokens=True,
        )
        super().__init__(config, **kwargs)


class OpenrouterChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_openrouter_config()
        super().__init__(config, **kwargs)


class GroqChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_groq_config()
        super().__init__(config, **kwargs)


def _get_gemini_config(**kwargs):
    """Create Gemini endpoint configuration with defaults."""
    config = dict(
        name="gemini_chat",
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        endpoint="chat/completions",
        kwargs={"model": "gemini-2.5-flash"},
        api_key=settings.GEMINI_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        request_options=OpenAIChatCompletionsRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


class GeminiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_gemini_config()
        super().__init__(config, **kwargs)


class OpenaiEmbedEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_oai_config(
            name="openai_embed",
            endpoint="embeddings",
            kwargs={"model": "text-embedding-3-small"},
            requires_tokens=True,
        )
        super().__init__(config, **kwargs)
