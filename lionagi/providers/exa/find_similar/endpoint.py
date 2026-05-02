# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Exa findSimilar endpoint — find pages similar to a URL.

Endpoint: POST https://api.exa.ai/findSimilar
Docs: https://docs.exa.ai/reference/find-similar-links
"""

from __future__ import annotations

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from .._config import ExaConfigs
from .models import ExaFindSimilarRequest

__all__ = ("ExaFindSimilarEndpoint",)


@ExaConfigs.FIND_SIMILAR.register
class ExaFindSimilarEndpoint(Endpoint):
    """Exa findSimilar endpoint — discover pages semantically similar to a URL.

    Usage::

        endpoint = ExaFindSimilarEndpoint()
        result = await endpoint.call({
            "url": "https://arxiv.org/abs/2303.08774",
            "numResults": 5,
        })
    """

    def __init__(self, config: EndpointConfig = None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.EXA_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("timeout", 120)
            kwargs.setdefault("max_retries", 3)
        super().__init__(config=config, **kwargs)
