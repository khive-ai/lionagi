# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from ..core import Service
from ..endpoint import ChatRequestModel
from ..providers.openai import create_generic_service
from ..providers.provider_registry import ProviderAdapter


def _host_rights(url: str | None, default: str) -> set[str]:
    host = urlparse(url or default).netloc or default
    return {f"net.out:{host}"}


class NvidiaNimAdapter(ProviderAdapter):
    """Adapter for NVIDIA NIM (NVIDIA Inference Microservice).
    
    Provides access to GPU-accelerated inference through NVIDIA's OpenAI-compatible API.
    
    Supports:
      - provider="nvidia_nim" or "nvidia"
      - model="nvidia/<model>" or "nvidia_nim/<model>"  
      - base_url containing 'integrate.api.nvidia.com'
      
    Available models include:
      - meta/llama-3.1-8b-instruct
      - meta/llama-3.1-70b-instruct
      - meta/llama-3.2-3b-instruct
      - mistralai/mistral-7b-instruct-v0.3
      - microsoft/phi-3-mini-4k-instruct
      - And many more open-source models
    """

    name = "nvidia_nim"
    default_base_url = "https://integrate.api.nvidia.com/v1"
    request_model = ChatRequestModel
    requires = _host_rights(default_base_url, "integrate.api.nvidia.com")

    # Optional config validator for pydantic ergonomics
    class ConfigModel(BaseModel):
        api_key: str
        base_url: str | None = None

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        # Support explicit provider specification
        if provider and provider.lower() in ("nvidia_nim", "nvidia"):
            return True
        
        # Support model prefix specification
        if model:
            model_lower = model.lower()
            if model_lower.startswith(("nvidia/", "nvidia_nim/")):
                return True
        
        # Support NVIDIA NIM base URL
        if base_url and "integrate.api.nvidia.com" in base_url.lower():
            return True
        
        return False

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        api_key = kwargs.pop("api_key", "")
        url = base_url or self.default_base_url
        
        # Use generic OpenAI-compatible service for NVIDIA NIM
        return create_generic_service(
            api_key=api_key,
            base_url=url,
            name="nvidia_nim",
            host_capability=f"net.out:{urlparse(url).netloc}",
            **kwargs
        )

    def required_rights(self, *, base_url: str | None, **_: Any) -> set[str]:
        return _host_rights(base_url, "integrate.api.nvidia.com")