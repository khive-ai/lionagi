# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.pplx_models import PerplexityChatRequest

__all__ = ("PerplexityChatEndpoint",)


ENDPOINT_CONFIG = EndpointConfig(
    name="perplexity_chat",
    provider="perplexity",
    base_url="https://api.perplexity.ai",
    endpoint="chat/completions",
    method="POST",
    kwargs={"model": "sonar"},
    api_key=settings.PERPLEXITY_API_KEY or "dummy-key-for-testing",
    auth_type="bearer",
    content_type="application/json",
    request_options=PerplexityChatRequest,
)


class PerplexityChatEndpoint(Endpoint):
    def __init__(self, config=ENDPOINT_CONFIG, **kwargs):
        super().__init__(config, **kwargs)
