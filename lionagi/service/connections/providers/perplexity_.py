# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

# Re-export for backward compatibility
from lionagi.service.third_party.pplx_models import PerplexityChatRequest

__all__ = ("PerplexityChatEndpoint", "PerplexityChatRequest")


class PerplexityChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        from ...third_party.pplx_models import PerplexityChatRequest

        # Check if api_key is provided in kwargs
        api_key = kwargs.get(
            "api_key", settings.PERPLEXITY_API_KEY or "dummy-key-for-testing"
        )

        _config = {
            "name": "perplexity_chat",
            "provider": "perplexity",
            "base_url": "https://api.perplexity.ai",
            "endpoint": "chat/completions",
            "method": "POST",
            "kwargs": {"model": "sonar"},
            "api_key": api_key,
            "auth_type": "bearer",
            "content_type": "application/json",
            "context_window": 100_000,  # Perplexity context window
            "request_options": PerplexityChatRequest,
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)
