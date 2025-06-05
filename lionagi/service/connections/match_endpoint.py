# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from .endpoint import Endpoint


def match_endpoint(
    provider: str,
    endpoint: str,
    **kwargs,
) -> Endpoint:
    if provider == "openai":
        if "chat" in endpoint:
            from .providers.oai_ import OpenaiChatEndpoint

            return OpenaiChatEndpoint(**kwargs)
        if "response" in endpoint:
            from .providers.oai_ import OpenaiResponseEndpoint

            return OpenaiResponseEndpoint(**kwargs)
    if provider == "openrouter" and "chat" in endpoint:
        from .providers.oai_ import OpenrouterChatEndpoint

        return OpenrouterChatEndpoint(**kwargs)
    if provider == "ollama" and "chat" in endpoint:
        from .providers.ollama_ import OllamaChatEndpoint

        return OllamaChatEndpoint(**kwargs)
    if provider == "exa" and "search" in endpoint:
        from .providers.exa_ import ExaSearchEndpoint

        return ExaSearchEndpoint(**kwargs)
    if provider == "anthropic" and (
        "messages" in endpoint or "chat" in endpoint
    ):
        from .providers.anthropic_ import AnthropicMessagesEndpoint

        return AnthropicMessagesEndpoint(**kwargs)
    if provider == "groq" and "chat" in endpoint:
        from .providers.oai_ import GroqChatEndpoint

        return GroqChatEndpoint(**kwargs)
    if provider == "perplexity" and "chat" in endpoint:
        from .providers.perplexity_ import PerplexityChatEndpoint

        return PerplexityChatEndpoint(**kwargs)

    return None
