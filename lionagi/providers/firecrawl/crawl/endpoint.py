# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Firecrawl crawl endpoint — asynchronous full-site crawl.

Endpoint: POST https://api.firecrawl.dev/v1/crawl
Docs: https://docs.firecrawl.dev/api-reference/endpoint/crawl-post

Returns a ``{"id": "<jobId>", "url": "..."}`` response.  Poll
GET /v1/crawl/{jobId} to retrieve results when status == "completed".
"""

from __future__ import annotations

from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from .._config import FirecrawlConfigs
from .models import FirecrawlCrawlRequest

__all__ = ("FirecrawlCrawlEndpoint",)


@FirecrawlConfigs.CRAWL.register
class FirecrawlCrawlEndpoint(Endpoint):
    """Firecrawl crawl endpoint — kick off an async full-site crawl job.

    Usage::

        endpoint = FirecrawlCrawlEndpoint()
        result = await endpoint.call({"url": "https://docs.example.com", "limit": 50})
        job_id = result["id"]
        # Then poll: GET https://api.firecrawl.dev/v1/crawl/{job_id}
    """

    def __init__(self, config: EndpointConfig = None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.FIRECRAWL_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("timeout", 120)
            kwargs.setdefault("max_retries", 3)
        super().__init__(config=config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        payload, headers = super().create_payload(request, extra_headers, **kwargs)
        # Re-serialize with camelCase aliases for the Firecrawl API.
        if self.config.request_options is not None:
            model_cls = self.config.request_options
            try:
                obj = model_cls.model_validate(payload)
                payload = obj.model_dump(by_alias=True, exclude_none=True)
            except Exception:
                pass
        return payload, headers
