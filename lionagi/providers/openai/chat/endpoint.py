# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI and OpenAI-compatible endpoint configurations.

This module provides endpoint configurations for:
- OpenAI (chat, response, embedding)
- Groq (OpenAI-compatible)
- Gemini OpenAI-compatible API (chat via generativelanguage endpoint)
"""

import io

from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from .._config import GeminiOAIConfigs, GroqConfigs, OpenAIConfigs

CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.5": 1_000_000,
    "gpt-5.4-mini": 1_000_000,
    "gpt-5.4": 1_048_576,
    "gpt-5": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4.1": 1_000_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 128_000,
    "o4-mini": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "o1-pro": 200_000,
    "o1-mini": 128_000,
    "o1": 200_000,
}


@OpenAIConfigs.CHAT.register
class OpenaiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.OPENAI_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": settings.OPENAI_DEFAULT_MODEL})
            kwargs.setdefault("requires_tokens", True)
        super().__init__(config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """Override to handle model-specific parameter filtering."""
        payload, headers = super().create_payload(request, extra_headers, **kwargs)
        # Convert system role to developer role for reasoning models
        if "messages" in payload and payload["messages"]:
            if payload["messages"][0].get("role") == "system":
                payload["messages"][0]["role"] = "developer"

        return (payload, headers)


@OpenAIConfigs.RESPONSE.register
class OpenaiResponseEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.OPENAI_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("requires_tokens", True)
        super().__init__(config, **kwargs)


@GroqConfigs.CHAT.register
class GroqChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.GROQ_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "llama-3.3-70b-versatile"})
        super().__init__(config, **kwargs)


@GeminiOAIConfigs.CHAT.register
class GeminiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.GEMINI_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "gemini-2.5-flash"})
        super().__init__(config, **kwargs)


@OpenAIConfigs.EMBED.register
class OpenaiEmbedEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.OPENAI_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "text-embedding-3-small"})
            kwargs.setdefault("requires_tokens", True)
        super().__init__(config, **kwargs)


@GroqConfigs.AUDIO_TRANSCRIPTION.register
class GroqAudioTranscriptionEndpoint(Endpoint):
    """Groq Whisper transcription endpoint.

    Groq supports the Whisper model for fast audio transcription.
    Uses multipart/form-data — pass ``file`` bytes and ``filename`` via kwargs.

    Usage::

        endpoint = GroqAudioTranscriptionEndpoint()
        with open("audio.mp3", "rb") as f:
            result = await endpoint.call(
                {"model": "whisper-large-v3"},
                file=f.read(),
                filename="audio.mp3",
            )
    """

    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.GROQ_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("timeout", 120)
            kwargs.setdefault("max_retries", 3)
        super().__init__(config, **kwargs)

    async def _call(self, payload: dict, headers: dict, **kwargs):
        """Encode audio as multipart/form-data."""
        import aiohttp

        file_data: bytes | None = kwargs.pop("file", None)
        filename: str = kwargs.pop("filename", "audio.mp3")

        form = aiohttp.FormData()
        for key, value in payload.items():
            if value is not None:
                form.add_field(key, str(value))

        if file_data is not None:
            form.add_field(
                "file",
                (
                    io.BytesIO(file_data)
                    if isinstance(file_data, (bytes, bytearray))
                    else file_data
                ),
                filename=filename,
                content_type="application/octet-stream",
            )

        multipart_headers = {
            k: v for k, v in headers.items() if k.lower() != "content-type"
        }

        async with self._create_http_session() as session:
            async with session.post(
                url=self.config.full_url,
                headers=multipart_headers,
                data=form,
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Groq transcription failed ({response.status}): {error_body}",
                        headers=response.headers,
                    )
                return await response.json()
