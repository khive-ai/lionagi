# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.firecrawl_models import (
    FirecrawlMapRequest,
    FirecrawlScrapeRequest,
)

__all__ = ("FirecrawlScrapeEndpoint", "FirecrawlMapEndpoint")


def _get_scrape_config() -> EndpointConfig:
    return EndpointConfig(
        name="firecrawl_scrape",
        provider="firecrawl",
        base_url="https://api.firecrawl.dev",
        endpoint="v1/scrape",
        method="POST",
        request_options=FirecrawlScrapeRequest,
        api_key=settings.FIRECRAWL_API_KEY or "dummy-key-for-testing",
        timeout=120,
        max_retries=3,
        auth_type="bearer",
        transport_type="http",
        content_type="application/json",
    )


def _get_map_config() -> EndpointConfig:
    return EndpointConfig(
        name="firecrawl_map",
        provider="firecrawl",
        base_url="https://api.firecrawl.dev",
        endpoint="v1/map",
        method="POST",
        request_options=FirecrawlMapRequest,
        api_key=settings.FIRECRAWL_API_KEY or "dummy-key-for-testing",
        timeout=120,
        max_retries=3,
        auth_type="bearer",
        transport_type="http",
        content_type="application/json",
    )


class FirecrawlScrapeEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_scrape_config()
        super().__init__(config=config, **kwargs)


class FirecrawlMapEndpoint(Endpoint):
    def __init__(self, config: EndpointConfig = None, **kwargs):
        config = config or _get_map_config()
        super().__init__(config=config, **kwargs)
