# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.tavily_models import (
    TavilyExtractRequest,
    TavilySearchRequest,
)

__all__ = ("TavilySearchEndpoint", "TavilyExtractEndpoint")


def _get_search_config() -> EndpointConfig:
    return EndpointConfig(
        name="tavily_search",
        provider="tavily",
        base_url="https://api.tavily.com",
        endpoint="search",
        method="POST",
        request_options=TavilySearchRequest,
        api_key=settings.TAVILY_API_KEY or "dummy-key-for-testing",
        timeout=120,
        max_retries=3,
        auth_type="bearer",
        content_type="application/json",
    )


def _get_extract_config() -> EndpointConfig:
    return EndpointConfig(
        name="tavily_extract",
        provider="tavily",
        base_url="https://api.tavily.com",
        endpoint="extract",
        method="POST",
        request_options=TavilyExtractRequest,
        api_key=settings.TAVILY_API_KEY or "dummy-key-for-testing",
        timeout=120,
        max_retries=3,
        auth_type="bearer",
        content_type="application/json",
    )


class TavilySearchEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_search_config()
        super().__init__(config=config, **kwargs)


class TavilyExtractEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_extract_config()
        super().__init__(config=config, **kwargs)
