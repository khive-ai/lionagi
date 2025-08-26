# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.exa_models import ExaSearchRequest

__all__ = ("ExaSearchEndpoint",)

_get_config = lambda: EndpointConfig(
    name="exa_search",
    provider="exa",
    base_url="https://api.exa.ai",
    endpoint="search",
    method="POST",
    request_options=ExaSearchRequest,
    api_key=settings.EXA_API_KEY or "dummy-key-for-testing",
    timeout=120,
    max_retries=3,
    auth_type="x-api-key",
    transport_type="http",
    content_type="application/json",
)

ENDPOINT_CONFIG = _get_config()  # backward compatibility


class ExaSearchEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_config()
        super().__init__(config=config, **kwargs)
