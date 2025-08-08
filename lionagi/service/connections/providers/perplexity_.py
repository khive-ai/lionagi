# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Perplexity AI endpoint configuration.

Perplexity provides real-time web search and Q&A capabilities through their Sonar API.
This module configures endpoints for different Sonar model tiers.
"""

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.pplx_models import PerplexityChatRequest

__all__ = (
    "PerplexityChatEndpoint",
    "PERPLEXITY_CHAT_ENDPOINT_CONFIG",
)


def _get_perplexity_config(**kwargs):
    """Create Perplexity endpoint configuration with defaults."""
    config = dict(
        name="perplexity_chat",
        provider="perplexity",
        base_url="https://api.perplexity.ai",
        endpoint="chat/completions",
        method="POST",
        kwargs={"model": "sonar"},  # Default to base sonar model
        api_key=settings.PERPLEXITY_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        request_options=PerplexityChatRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


# Default configuration (users can specify model at runtime)
PERPLEXITY_CHAT_ENDPOINT_CONFIG = _get_perplexity_config()

# Legacy naming for backward compatibility
ENDPOINT_CONFIG = PERPLEXITY_CHAT_ENDPOINT_CONFIG


class PerplexityChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_perplexity_config()
        super().__init__(config, **kwargs)
