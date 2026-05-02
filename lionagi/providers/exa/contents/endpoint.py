# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Exa contents extraction endpoint.

Endpoint: POST https://api.exa.ai/contents
Docs: https://docs.exa.ai/reference/contents
"""

from __future__ import annotations

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from .._config import ExaConfigs
from .models import ExaContentsRequest

__all__ = ("ExaContentsEndpoint",)


@ExaConfigs.CONTENTS.register
class ExaContentsEndpoint(Endpoint):
    """Exa contents endpoint — extract page text/highlights/summaries from URLs.

    Usage::

        endpoint = ExaContentsEndpoint()
        result = await endpoint.call({
            "ids": ["https://example.com"],
            "contents": {"text": {"maxCharacters": 1000}}
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
