# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

# Eager imports for core functionality
from .connections.api_calling import APICalling
from .connections.endpoint import Endpoint, EndpointConfig
from .connections.providers.types import *
from .hooks import *
from .imodel import iModel
from .manager import iModelManager
from .rate_limited_processor import RateLimitedAPIExecutor

# Lazy loading cache
_lazy_imports = {}


def __getattr__(name: str):
    """Lazy loading for heavy service imports."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "TokenCalculator":
        from .token_calculator import TokenCalculator

        _lazy_imports["TokenCalculator"] = TokenCalculator
        return TokenCalculator

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
    "APICalling",
    "Endpoint",
    "EndpointConfig",
    "RateLimitedAPIExecutor",
    "TokenCalculator",
    "iModel",
    "iModelManager",
    "AnthropicMessagesEndpoint",
    "ClaudeCodeEndpoint",
    "ClaudeCodeRequest",
    "ClaudeCodeCLIEndpoint",
    "ExaSearchEndpoint",
    "ExaSearchRequest",
    "OpenaiChatEndpoint",
    "OpenaiEmbedEndpoint",
    "OpenaiResponseEndpoint",
    "OpenrouterChatEndpoint",
    "GroqChatEndpoint",
    "OllamaChatEndpoint",
    "PerplexityChatEndpoint",
    "PerplexityChatRequest",
    "HookEventTypes",
    "HookDict",
    "AssosiatedEventInfo",
    "HookEvent",
    "HookRegistry",
)
