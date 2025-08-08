# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Ollama endpoint configuration.

Ollama provides local model hosting with both native and OpenAI-compatible APIs.
This module configures the OpenAI-compatible endpoint for consistency.
"""

from pydantic import BaseModel

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.utils import is_import_installed

__all__ = (
    "OllamaChatEndpoint",
    "OLLAMA_CHAT_ENDPOINT_CONFIG",
)

_HAS_OLLAMA = is_import_installed("ollama")


def _get_ollama_config(**kwargs):
    """Create Ollama endpoint configuration with defaults."""
    config = dict(
        name="ollama_chat",
        provider="ollama",
        base_url="http://localhost:11434/v1",  # OpenAI-compatible endpoint
        endpoint="chat/completions",
        kwargs={},  # Model will be provided at runtime
        openai_compatible=False,  # Use HTTP transport
        api_key=None,  # No API key needed
        method="POST",
        content_type="application/json",
        auth_type="none",  # No authentication
        default_headers={},  # No auth headers needed
        # NOTE: Not using request_options due to OpenAI model role literal issues
        # request_options=CreateChatCompletionRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


# Default OpenAI-compatible configuration
OLLAMA_CHAT_ENDPOINT_CONFIG = _get_ollama_config()


class OllamaChatEndpoint(Endpoint):
    """
    Documentation: https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(self, config=None, **kwargs):
        if not _HAS_OLLAMA:
            raise ModuleNotFoundError(
                "ollama is not installed, please install it with `pip install lionagi[ollama]`"
            )

        # Override api_key for Ollama (not needed)
        if "api_key" in kwargs:
            kwargs.pop("api_key")

        config = config or _get_ollama_config()
        super().__init__(config, **kwargs)

        from ollama import list as ollama_list  # type: ignore[import]
        from ollama import pull as ollama_pull  # type: ignore[import]

        self._pull = ollama_pull
        self._list = ollama_list

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """Override to handle Ollama-specific needs."""
        payload, headers = super().create_payload(
            request, extra_headers, **kwargs
        )

        # Ollama doesn't support reasoning_effort
        payload.pop("reasoning_effort", None)

        return (payload, headers)

    async def call(
        self, request: dict | BaseModel, cache_control: bool = False, **kwargs
    ):
        payload, _ = self.create_payload(request, **kwargs)

        # Check if model exists and pull if needed
        model = payload["model"]
        self._check_model(model)

        # The parent call method will handle headers internally
        return await super().call(
            payload, cache_control=cache_control, **kwargs
        )

    def _pull_model(self, model: str):
        from tqdm import tqdm

        current_digest, bars = "", {}
        for progress in self._pull(model, stream=True):
            digest = progress.get("digest", "")
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get("status"))
                continue

            if digest not in bars and (total := progress.get("total")):
                bars[digest] = tqdm(
                    total=total,
                    desc=f"pulling {digest[7:19]}",
                    unit="B",
                    unit_scale=True,
                )

            if completed := progress.get("completed"):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest

    def _check_model(self, model: str):
        try:
            available_models = [i.model for i in self._list().models]

            if model not in available_models:
                print(
                    f"Model '{model}' not found locally. Pulling from Ollama registry..."
                )
                self._pull_model(model)
                print(f"Model '{model}' successfully pulled.")
        except Exception as e:
            print(f"Warning: Could not check/pull model '{model}': {e}")
