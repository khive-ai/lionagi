# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA NIM endpoint configurations.

This module provides endpoint configurations for NVIDIA NIM (NVIDIA Inference Microservices),
which offers GPU-accelerated inference for various AI models through an OpenAI-compatible API.

NVIDIA NIM features:
- OpenAI-compatible API endpoints
- GPU-accelerated inference
- Support for various open-source models (Llama, Mistral, etc.)
- Both cloud-hosted and self-hosted options
- Free tier with 1000 credits for development

API Documentation: https://docs.nvidia.com/nim/
Build Portal: https://build.nvidia.com/
"""

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

    Supports various open-source models including:
    - meta/llama3-8b-instruct
    - meta/llama3-70b-instruct
    - meta/llama3.1-405b-instruct
    - mistralai/mixtral-8x7b-instruct-v0.1
    - google/gemma-7b
    - And many more...

    Get your API key from: https://build.nvidia.com/
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
