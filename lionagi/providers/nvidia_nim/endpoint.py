# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

from ._config import NvidiaNimConfigs

CONTEXT_WINDOWS: dict[str, int] = {
    "nemotron-ultra": 131_072,
    "nemotron-super": 131_072,
    "llama-4-scout": 10_485_760,
    "llama-4-maverick": 1_048_576,
    "llama-3": 128_000,
}

__all__ = (
    "NvidiaNimChatEndpoint",
    "NvidiaNimEmbedEndpoint",
)


@NvidiaNimConfigs.CHAT.register
class NvidiaNimChatEndpoint(Endpoint):
    """NVIDIA NIM chat completion endpoint.

    Get your API key from: https://build.nvidia.com/
    API Documentation: https://docs.nvidia.com/nim/
    """

    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.NVIDIA_NIM_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "meta/llama3-8b-instruct"})
            kwargs.setdefault("requires_tokens", True)
        super().__init__(config, **kwargs)


@NvidiaNimConfigs.EMBED.register
class NvidiaNimEmbedEndpoint(Endpoint):
    """NVIDIA NIM embedding endpoint.

    Note: Verify available embedding models at https://build.nvidia.com/
    """

    def __init__(self, config=None, **kwargs):
        if config is None:
            from lionagi.config import settings

            kwargs.setdefault(
                "api_key", settings.NVIDIA_NIM_API_KEY or "dummy-key-for-testing"
            )
            kwargs.setdefault("kwargs", {"model": "nvidia/nv-embed-v1"})
        super().__init__(config, **kwargs)
