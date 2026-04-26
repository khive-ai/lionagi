# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.third_party.deepseek_models import (
    DeepseekChatCompletionsRequest,
    normalize_deepseek_usage,
)

from ..endpoint import Endpoint, EndpointConfig

CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4": 1_000_000,
    "deepseek-coder-v2": 128_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-v3": 128_000,
    "deepseek-r1": 64_000,
}


def _get_deepseek_config(**kwargs):
    """Create DeepSeek endpoint configuration with defaults."""
    config = dict(
        name="deepseek_chat",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        endpoint="chat/completions",
        kwargs={"model": "deepseek-chat"},
        api_key=settings.DEEPSEEK_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        request_options=DeepseekChatCompletionsRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


class DeepseekChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_deepseek_config()
        super().__init__(config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        payload, headers = super().create_payload(request, extra_headers, **kwargs)
        original_messages = payload.get("messages")
        req = DeepseekChatCompletionsRequest.model_validate(payload)
        payload = req.model_dump(exclude_none=True, mode="json")
        if original_messages is not None:
            payload["messages"] = original_messages
        return payload, headers

    async def _call(self, payload: dict, headers: dict, **kwargs):
        response = await super()._call(payload, headers, **kwargs)
        return normalize_deepseek_usage(response)
