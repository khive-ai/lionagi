# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from ..core import Service
from ..endpoint import ChatRequestModel
from ..openai import create_generic_service, create_openai_service
from ..provider_registry import ProviderAdapter


def _host_rights(url: str | None, default: str) -> set[str]:
    host = urlparse(url or default).netloc or default
    return {f"net.out:{host}"}


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI & compatible OpenAI-like endpoints.

    Supports:
      - provider="openai"
      - model="openai/<model>"
      - base_url containing 'api.openai.com'
      - Generic OpenAI-compatible hosts (use create_generic_service)
    """

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    request_model = ChatRequestModel
    requires = _host_rights(default_base_url, "api.openai.com")

    # Optional config validator (keeps your pydantic ergonomics)
    class ConfigModel(BaseModel):
        api_key: str
        organization: str | None = None
        base_url: str | None = None

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        if (provider or "").lower() == "openai":
            return True
        if (model or "").lower().startswith("openai/"):
            return True
        return (base_url or "").lower().find("api.openai.com") >= 0

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        api_key = kwargs.pop("api_key", "")
        organization = kwargs.pop("organization", None)
        url = base_url or self.default_base_url

        # If the host is OpenAI, use native factory; otherwise generic OpenAI-compat path
        if urlparse(url).netloc.endswith("openai.com"):
            return create_openai_service(api_key=api_key, organization=organization, **kwargs)
        return create_generic_service(
            api_key=api_key, base_url=url, name="openai-compatible", **kwargs
        )

    def required_rights(self, *, base_url: str | None, **_: Any) -> set[str]:
        return _host_rights(base_url, "api.openai.com")
