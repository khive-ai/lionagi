# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.openai_models import (
    CreateChatCompletionRequest,
    CreateResponse,
)

__all__ = (
    "OpenaiChatEndpoint",
    "OpenaiResponseEndpoint",
    "OpenrouterChatEndpoint",
    "OPENROUTER_GEMINI_ENDPOINT_CONFIG",
)


OPENAI_CHAT_ENDPOINT_CONFIG = EndpointConfig(
    name="openai_chat",
    provider="openai",
    base_url="https://api.openai.com/v1",
    endpoint="chat/completions",
    kwargs={"model": "gpt-4o"},
    api_key=settings.OPENAI_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    method="POST",
    requires_tokens=True,
    request_options=CreateChatCompletionRequest,
)

OPENAI_RESPONSE_ENDPOINT_CONFIG = EndpointConfig(
    name="openai_response",
    provider="openai",
    base_url="https://api.openai.com/v1",
    endpoint="chat/completions",  # OpenAI responses API uses same endpoint
    kwargs={"model": "gpt-4o"},
    api_key=settings.OPENAI_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    method="POST",
    requires_tokens=True,
    request_options=CreateResponse,
)

OPENROUTER_CHAT_ENDPOINT_CONFIG = EndpointConfig(
    name="openrouter_chat",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    endpoint="chat/completions",
    kwargs={"model": "google/gemini-2.5-flash-preview-05-20"},
    api_key=settings.OPENROUTER_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    method="POST",
    request_options=CreateChatCompletionRequest,
)

OPENROUTER_GEMINI_ENDPOINT_CONFIG = EndpointConfig(
    name="openrouter_gemini",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    endpoint="chat/completions",
    kwargs={"model": "google/gemini-2.5-flash-preview-05-20"},
    api_key=settings.OPENROUTER_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    method="POST",
)

OPENAI_EMBEDDING_ENDPOINT_CONFIG = EndpointConfig(
    name="openai_embed",
    provider="openai",
    base_url="https://api.openai.com/v1",
    endpoint="embeddings",
    kwargs={"model": "text-embedding-3-small"},
    api_key=settings.OPENAI_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    method="POST",
)

GROQ_CHAT_ENDPOINT_CONFIG = EndpointConfig(
    name="groq_chat",
    provider="groq",
    base_url="https://api.groq.com/openai/v1",
    endpoint="chat/completions",
    api_key=settings.GROQ_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    method="POST",
)


REASONING_MODELS = (
    "o3-mini-2025-01-31",
    "o3-mini",
    "o1",
    "o1-2024-12-17",
)

REASONING_NOT_SUPPORT_PARAMS = (
    "temperature",
    "top_p",
    "logit_bias",
    "logprobs",
    "top_logprobs",
)


class OpenaiChatEndpoint(Endpoint):
    def __init__(self, config=OPENAI_CHAT_ENDPOINT_CONFIG, **kwargs):
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

        # Handle reasoning models
        model = payload.get("model")
        if model in REASONING_MODELS:
            # Remove unsupported parameters for reasoning models
            for param in REASONING_NOT_SUPPORT_PARAMS:
                payload.pop(param, None)

            # Convert system role to developer role for reasoning models
            if "messages" in payload and payload["messages"]:
                if payload["messages"][0].get("role") == "system":
                    payload["messages"][0]["role"] = "developer"
        else:
            # Remove reasoning_effort for non-reasoning models
            payload.pop("reasoning_effort", None)

        return (payload, headers)


class OpenaiResponseEndpoint(Endpoint):
    def __init__(self, config=OPENAI_RESPONSE_ENDPOINT_CONFIG, **kwargs):
        super().__init__(config, **kwargs)


class OpenrouterChatEndpoint(Endpoint):
    def __init__(self, config=OPENROUTER_CHAT_ENDPOINT_CONFIG, **kwargs):
        super().__init__(config, **kwargs)


class GroqChatEndpoint(Endpoint):
    def __init__(self, config=GROQ_CHAT_ENDPOINT_CONFIG, **kwargs):
        super().__init__(config, **kwargs)


class OpenaiEmbedEndpoint(Endpoint):
    def __init__(self, config=OPENAI_EMBEDDING_ENDPOINT_CONFIG, **kwargs):
        super().__init__(config, **kwargs)
