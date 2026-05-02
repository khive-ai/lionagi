# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from lionagi.service.connections.provider_config import LazyType, ProviderConfig
from lionagi.service.connections.registry import EndpointType



class GeminiConfigs(ProviderConfig, Enum):
    CHAT = (
        "chat/completions",
        ["chat"],
        EndpointType.API,
        LazyType("lionagi.providers.openai.chat.models:OpenAIChatCompletionsRequest"),
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "bearer",
    )
    CLI = (
        "query_cli",
        ["cli"],
        EndpointType.AGENTIC,
        LazyType("lionagi.providers.google.gemini_code.models:GeminiCodeRequest"),
    )

GeminiConfigs._PROVIDER = "gemini"
GeminiConfigs._PROVIDER_ALIASES = ["gemini-api", "gemini", "gemini-code",  "gemini_code", "gemini_cli", "gemini-cli"]


__all__ = ("GeminiConfigs",)
