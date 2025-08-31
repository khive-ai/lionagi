# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from lionagi.service.connections.endpoint_config import EndpointConfig

from .endpoint import Endpoint


def match_endpoint(
    provider: str,
    endpoint: str,
    **kwargs,
) -> Endpoint:
    if provider == "openai":
        if "chat" in endpoint:
            from .providers.oai_ import OpenaiChatEndpoint

            return OpenaiChatEndpoint(None, **kwargs)
        if "response" in endpoint:
            from .providers.oai_ import OpenaiResponseEndpoint

            return OpenaiResponseEndpoint(None, **kwargs)
    if provider == "openrouter" and "chat" in endpoint:
        from .providers.oai_ import OpenrouterChatEndpoint

        return OpenrouterChatEndpoint(None, **kwargs)
    if provider == "ollama" and "chat" in endpoint:
        from .providers.ollama_ import OllamaChatEndpoint

        return OllamaChatEndpoint(None, **kwargs)
    if provider == "exa" and "search" in endpoint:
        from .providers.exa_ import ExaSearchEndpoint

        return ExaSearchEndpoint(None, **kwargs)
    if provider == "anthropic" and (
        "messages" in endpoint or "chat" in endpoint
    ):
        from .providers.anthropic_ import AnthropicMessagesEndpoint

        return AnthropicMessagesEndpoint(None, **kwargs)
    if provider == "groq" and "chat" in endpoint:
        from .providers.oai_ import GroqChatEndpoint

        return GroqChatEndpoint(None, **kwargs)
    if provider == "perplexity" and "chat" in endpoint:
        from .providers.perplexity_ import PerplexityChatEndpoint

        return PerplexityChatEndpoint(None, **kwargs)
    if provider == "nvidia_nim":
        if "embed" in endpoint:
            from .providers.nvidia_nim_ import NvidiaNimEmbedEndpoint

            return NvidiaNimEmbedEndpoint(None, **kwargs)
        if "chat" in endpoint or "completion" in endpoint:
            from .providers.nvidia_nim_ import NvidiaNimChatEndpoint

            return NvidiaNimChatEndpoint(None, **kwargs)
    if provider == "claude_code":
        if "cli" in endpoint:
            from .providers.claude_code_cli import ClaudeCodeCLIEndpoint

            return ClaudeCodeCLIEndpoint(None, **kwargs)

        if "query" in endpoint or "code" in endpoint:
            from lionagi.service.connections.providers.claude_code_ import (
                ClaudeCodeEndpoint,
            )

            return ClaudeCodeEndpoint(None, **kwargs)

    from .providers.oai_ import OpenaiChatEndpoint

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
