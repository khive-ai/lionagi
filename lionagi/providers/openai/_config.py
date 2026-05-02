# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from lionagi.service.connections.provider_config import LazyType, ProviderConfig
from lionagi.service.connections.registry import EndpointType


class OpenAIConfigs(ProviderConfig, Enum):

    CHAT = (
        "chat/completions",
        ["chat"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.chat.models:OpenAIChatCompletionsRequest"),
        "https://api.openai.com/v1",
        "bearer",
    )
    RESPONSE = (
        "responses",
        ["response"],
        EndpointType.API,
        None,
        "https://api.openai.com/v1",
        "bearer",
    )
    EMBED = (
        "embeddings",
        ["embed"],
        EndpointType.API,
        None,
        "https://api.openai.com/v1",
        "bearer",
    )
    AUDIO_SPEECH = (
        "audio/speech",
        ["tts"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.audio.models:AudioSpeechRequest"),
        "https://api.openai.com/v1",
        "bearer",
    )
    AUDIO_TRANSCRIPTION = (
        "audio/transcriptions",
        ["stt", "whisper"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.audio.models:AudioTranscriptionRequest"),
        "https://api.openai.com/v1",
        "bearer",
    )
    IMAGE_GENERATION = (
        "images/generations",
        ["dalle", "image"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.images.models:ImageGenerationRequest"),
        "https://api.openai.com/v1",
        "bearer",
    )
    IMAGE_EDIT = (
        "images/edits",
        ["image_edit"],
        EndpointType.API,
        None,
        "https://api.openai.com/v1",
        "bearer",
    )


OpenAIConfigs._PROVIDER = "openai"
OpenAIConfigs._PROVIDER_ALIASES = []


class CodexConfigs(ProviderConfig, Enum):

    CLI = (
        "query_cli",
        ["cli", "code"],
        EndpointType.AGENTIC,
        LazyType("lionagi.providers.openai.codex.models:CodexCodeRequest"),
    )


CodexConfigs._PROVIDER = "codex"
CodexConfigs._PROVIDER_ALIASES = ["openai-codex"]


class GroqConfigs(ProviderConfig, Enum):

    CHAT = (
        "chat/completions",
        ["chat"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.chat.models:OpenAIChatCompletionsRequest"),
        "https://api.groq.com/openai/v1",
        "bearer",
    )
    AUDIO_TRANSCRIPTION = (
        "audio/transcriptions",
        ["whisper", "stt"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.audio.models:AudioTranscriptionRequest"),
        "https://api.groq.com/openai/v1",
        "bearer",
    )


GroqConfigs._PROVIDER = "groq"
GroqConfigs._PROVIDER_ALIASES = []


class GeminiOAIConfigs(ProviderConfig, Enum):

    CHAT = (
        "chat/completions",
        ["chat"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.chat.models:OpenAIChatCompletionsRequest"),
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "bearer",
    )


GeminiOAIConfigs._PROVIDER = "gemini"
GeminiOAIConfigs._PROVIDER_ALIASES = ["gemini-api"]

__all__ = (
    "OpenAIConfigs",
    "CodexConfigs",
    "GroqConfigs",
    "GeminiOAIConfigs",
)
