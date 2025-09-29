# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

__all__ = (
    "NvidiaNimChatEndpoint",
    "NvidiaNimEmbedEndpoint",
    "NVIDIA_NIM_CHAT_ENDPOINT_CONFIG",
    "NVIDIA_NIM_EMBED_ENDPOINT_CONFIG",
)


def _get_nvidia_nim_config(**kwargs):
    """Create NVIDIA NIM endpoint configuration with defaults.

    NVIDIA NIM uses the integrate.api.nvidia.com endpoint for cloud-hosted models.
    Authentication is via bearer token (API key from build.nvidia.com).
    """
    config = dict(
        name="nvidia_nim_chat",
        provider="nvidia_nim",
        base_url="https://integrate.api.nvidia.com/v1",
        endpoint="chat/completions",
        kwargs={"model": "meta/llama3-8b-instruct"},  # Default model
        api_key=settings.NVIDIA_NIM_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        requires_tokens=True,
        # OpenAI-compatible format
    )
    config.update(kwargs)
    return EndpointConfig(**config)


# Chat endpoint configuration
NVIDIA_NIM_CHAT_ENDPOINT_CONFIG = _get_nvidia_nim_config()

# Embedding endpoint configuration
# Note: You'll need to verify which embedding models are available on NVIDIA NIM
NVIDIA_NIM_EMBED_ENDPOINT_CONFIG = _get_nvidia_nim_config(
    name="nvidia_nim_embed",
    endpoint="embeddings",
    kwargs={"model": "nvidia/nv-embed-v1"},  # Example embedding model
)


class NvidiaNimChatEndpoint(Endpoint):
    """NVIDIA NIM chat completion endpoint.

    Get your API key from: https://build.nvidia.com/
    API Documentation: https://docs.nvidia.com/nim/
    """

    def __init__(self, config=None, **kwargs):
        config = config or _get_nvidia_nim_config()
        super().__init__(config, **kwargs)


class NvidiaNimEmbedEndpoint(Endpoint):
    """NVIDIA NIM embedding endpoint.

    Note: Verify available embedding models at https://build.nvidia.com/
    """

    def __init__(self, config=None, **kwargs):
        config = config or _get_nvidia_nim_config(
            name="nvidia_nim_embed",
            endpoint="embeddings",
            kwargs={"model": "nvidia/nv-embed-v1"},
        )
        super().__init__(config, **kwargs)
