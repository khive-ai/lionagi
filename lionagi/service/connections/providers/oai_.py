# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from .model_specs import (
    get_model_context_window, 
    is_reasoning_model,
    get_model_capabilities,
    is_deprecated_model
)

__all__ = (
    "OpenaiChatEndpoint",
    "OpenaiResponseEndpoint", 
    "OpenaiStructuredOutputsEndpoint",
    "OpenrouterChatEndpoint",
    "GroqChatEndpoint",
    "OpenaiEmbedEndpoint",
)

# Transport types
TransportType = Literal["http", "sdk", "mcp", "event"]


REASONING_MODELS = (
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-pro",
    "o1-pro-2025-03-19",
    "o3-pro",
    "o3-pro-2025-06-10",
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1-mini",
    "o1-mini-2024-09-12",
)

REASONING_NOT_SUPPORT_PARAMS = (
    "temperature",
    "top_p",
    "logit_bias",
    "logprobs",
    "top_logprobs",
)


class OpenaiChatEndpoint(Endpoint):
    def __init__(self, config=None, transport: TransportType = "http", **kwargs):
        from ...third_party.openai_models import CreateChatCompletionRequest

        # Handle headless scenarios - if api_key is explicitly None, use no auth
        if "api_key" in kwargs and kwargs["api_key"] is None:
            api_key = None
            auth_type = "none"
        else:
            # Use provided api_key, fall back to settings, then dummy key
            api_key = kwargs.get("api_key")
            if api_key is None:
                api_key = settings.OPENAI_API_KEY or "dummy-key-for-testing"
            auth_type = "bearer"

        # Get model from config or kwargs to determine context window
        model_name = None
        if config and isinstance(config, dict):
            model_name = config.get("kwargs", {}).get("model")
        elif config and isinstance(config, EndpointConfig):
            model_name = config.kwargs.get("model") if config.kwargs else None
        if not model_name:
            model_name = kwargs.get("model", "gpt-4.1-nano")

        # Get dynamic context window for the model
        context_window = get_model_context_window(model_name)
        
        # Check for deprecated models and warn
        if is_deprecated_model(model_name):
            import warnings
            warnings.warn(
                f"Model '{model_name}' is deprecated and may be retired soon. "
                f"Consider using a newer model.", 
                DeprecationWarning, 
                stacklevel=2
            )

        _config = {
            "name": "openai_chat",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat/completions",
            "kwargs": {"model": model_name},
            "api_key": api_key,
            "auth_type": auth_type,
            "content_type": "application/json",
            "method": "POST",
            "requires_tokens": True,
            "openai_compatible": True,
            "context_window": context_window,
            "transport": transport,
            "request_options": CreateChatCompletionRequest,
            "model_capabilities": get_model_capabilities(model_name),
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """Override to handle model-specific parameter filtering."""
        payload, headers = super().create_payload(
            request, extra_headers, **kwargs
        )

        # Handle reasoning models - use dynamic detection
        model = payload.get("model")
        if is_reasoning_model(model):
            # Remove unsupported parameters for reasoning models
            for param in REASONING_NOT_SUPPORT_PARAMS:
                payload.pop(param, None)

            # Convert system role to developer role for reasoning models
            if "messages" in payload and payload["messages"]:
                if payload["messages"][0].get("role") == "system":
                    payload["messages"][0]["role"] = "developer"
        else:
            # Remove reasoning_effort for non-reasoning models
            payload.pop("reasoning_effort", None)

        return (payload, headers)

    async def call_api(self, payload: dict, **kwargs) -> dict:
        """Call API using configured transport method."""
        transport = getattr(self.config, 'transport', 'http')
        
        if transport == "sdk":
            return await self._call_via_sdk(payload, **kwargs)
        elif transport == "http":
            return await super().call_api(payload, **kwargs)
        elif transport == "mcp":
            return await self._call_via_mcp(payload, **kwargs)
        elif transport == "event":
            return await self._call_via_event(payload, **kwargs)
        else:
            raise ValueError(f"Unsupported transport type: {transport}")

    async def _call_via_sdk(self, payload: dict, **kwargs) -> dict:
        """Call OpenAI API using official Python SDK."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI Python SDK not installed. Install with: pip install openai"
            )

        # Configure OpenAI client
        client_kwargs = {
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
        }
        if hasattr(self.config, 'timeout'):
            client_kwargs["timeout"] = self.config.timeout
        if hasattr(self.config, 'max_retries'):
            client_kwargs["max_retries"] = self.config.max_retries

        client = openai.AsyncOpenAI(**client_kwargs)

        try:
            # Use OpenAI SDK to make the call
            response = await client.chat.completions.create(**payload)
            
            # Convert response to dict format - much cleaner with Pydantic model_dump()
            return response.model_dump()

        except openai.APIError as e:
            # Convert OpenAI SDK errors to lionagi format
            raise Exception(f"OpenAI API error: {e}")

    async def _call_via_mcp(self, payload: dict, **kwargs) -> dict:
        """Call via MCP transport (to be implemented)."""
        raise NotImplementedError("MCP transport not yet implemented")

    async def _call_via_event(self, payload: dict, **kwargs) -> dict:
        """Call via event-driven transport (to be implemented)."""
        raise NotImplementedError("Event transport not yet implemented")


class OpenaiStructuredOutputsEndpoint(Endpoint):
    """OpenAI Structured Outputs endpoint for guaranteed JSON schema compliance."""
    
    def __init__(self, config=None, transport: TransportType = "sdk", **kwargs):
        # Handle headless scenarios - if api_key is explicitly None, use no auth
        if "api_key" in kwargs and kwargs["api_key"] is None:
            api_key = None
            auth_type = "none"
        else:
            # Use provided api_key, fall back to settings, then dummy key
            api_key = kwargs.get("api_key")
            if api_key is None:
                api_key = settings.OPENAI_API_KEY or "dummy-key-for-testing"
            auth_type = "bearer"

        # Get model from config or kwargs to determine context window
        model_name = None
        if config and isinstance(config, dict):
            model_name = config.get("kwargs", {}).get("model")
        elif config and isinstance(config, EndpointConfig):
            model_name = config.kwargs.get("model") if config.kwargs else None
        if not model_name:
            model_name = kwargs.get("model", "gpt-4o-2024-08-06")  # Default to first structured outputs model

        # Validate model supports structured outputs
        capabilities = get_model_capabilities(model_name)
        if not capabilities.structured_outputs:
            import warnings
            warnings.warn(
                f"Model '{model_name}' may not support structured outputs. "
                f"Consider using 'gpt-4o-2024-08-06' or 'gpt-4o-mini'.",
                UserWarning,
                stacklevel=2
            )

        # Get dynamic context window for the model
        context_window = get_model_context_window(model_name)

        _config = {
            "name": "openai_structured_outputs",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat/completions",
            "kwargs": {"model": model_name},
            "api_key": api_key,
            "auth_type": auth_type,
            "content_type": "application/json",
            "method": "POST",
            "requires_tokens": True,
            "openai_compatible": True,
            "context_window": context_window,
            "transport": transport,
            "supports_structured_outputs": True,
            "model_capabilities": capabilities,
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        response_schema: Optional[BaseModel] = None,
        **kwargs,
    ):
        """Create payload with structured outputs configuration."""
        payload, headers = super().create_payload(
            request, extra_headers, **kwargs
        )

        # Add structured outputs configuration if schema provided
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema", 
                "json_schema": {
                    "name": response_schema.__name__.lower(),
                    "strict": True,
                    "schema": response_schema.model_json_schema()
                }
            }

        return payload, headers

    async def call_api(
        self, 
        payload: dict, 
        response_schema: Optional[BaseModel] = None,
        **kwargs
    ) -> dict:
        """Call API with structured outputs parsing."""
        transport = getattr(self.config, 'transport', 'sdk')
        
        if transport == "sdk":
            return await self._call_via_sdk(payload, response_schema, **kwargs)
        elif transport == "http":
            return await self._call_via_http(payload, response_schema, **kwargs)
        else:
            raise ValueError(f"Transport {transport} not supported for structured outputs")

    async def _call_via_sdk(
        self, 
        payload: dict, 
        response_schema: Optional[BaseModel] = None,
        **kwargs
    ) -> dict:
        """Call structured outputs using OpenAI SDK."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI Python SDK not installed. Install with: pip install openai"
            )

        # Configure OpenAI client
        client_kwargs = {
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
        }
        if hasattr(self.config, 'timeout'):
            client_kwargs["timeout"] = getattr(self.config, 'timeout', 60.0)  # Longer timeout for first request
        if hasattr(self.config, 'max_retries'):
            client_kwargs["max_retries"] = self.config.max_retries

        client = openai.AsyncOpenAI(**client_kwargs)

        try:
            if response_schema:
                # Use beta parse endpoint for Pydantic integration
                response = await client.beta.chat.completions.parse(
                    model=payload["model"],
                    messages=payload["messages"],
                    response_format=response_schema,
                    **{k: v for k, v in payload.items() 
                       if k not in ["model", "messages", "response_format"]}
                )
                
                # Handle refusals
                message = response.choices[0].message
                if message.refusal:
                    raise ValueError(f"Model refused request: {message.refusal}")

                # Convert response with parsed data - use model_dump() for clean conversion
                response_dict = response.model_dump()
                
                # Note: The parsed Pydantic object is automatically included in model_dump()
                # but may need special handling depending on lionagi's expectations
                return response_dict
            else:
                # Fall back to regular completion
                response = await client.chat.completions.create(**payload)
                return self._convert_openai_response_to_dict(response)

        except openai.APIError as e:
            # Convert OpenAI SDK errors to lionagi format
            if "rate_limit" in str(e).lower():
                raise Exception(f"Rate limit error: {e}")
            elif "timeout" in str(e).lower():
                raise Exception("Request timeout - schema may be too complex for first-time processing")
            else:
                raise Exception(f"OpenAI API error: {e}")

    async def _call_via_http(
        self, 
        payload: dict, 
        response_schema: Optional[BaseModel] = None,
        **kwargs
    ) -> dict:
        """Call structured outputs via HTTP with manual schema handling."""
        # Add response_format to payload if schema provided
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__.lower(),
                    "strict": True,
                    "schema": response_schema.model_json_schema()
                }
            }

        # Call HTTP endpoint
        response = await super().call_api(payload, **kwargs)

        # Handle refusals and parse structured outputs
        if response.get("choices", [{}])[0].get("message", {}).get("refusal"):
            refusal = response["choices"][0]["message"]["refusal"]
            raise ValueError(f"Model refused request: {refusal}")

        # Parse structured output if schema provided
        if response_schema and response.get("choices"):
            content = response["choices"][0]["message"]["content"]
            try:
                import json
                parsed_data = response_schema.model_validate_json(content)
                response["choices"][0]["message"]["parsed"] = parsed_data
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to parse structured output: {e}")
                raise ValueError(f"Failed to parse structured output: {e}")

        return response

    def _convert_openai_response_to_dict(self, response) -> dict:
        """Convert OpenAI SDK response to dict format."""
        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "function_call": choice.message.function_call,
                        "tool_calls": choice.message.tool_calls,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            "system_fingerprint": getattr(response, 'system_fingerprint', None),
        }


class OpenaiResponseEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        from ...third_party.openai_models import CreateResponse

        _config = dict(
            name="openai_response",
            provider="openai",
            base_url="https://api.openai.com/v1",
            endpoint="chat/completions",  # OpenAI responses API uses same endpoint
            kwargs={"model": "gpt-4.1-nano"},
            api_key=settings.OPENAI_API_KEY or "dummy-key-for-testing",
            auth_type="bearer",
            content_type="application/json",
            method="POST",
            requires_tokens=True,
            request_options=CreateResponse,
        )
        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump(exclude_none=True)
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)


class OpenrouterChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        _config = dict(
            name="openrouter_chat",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            endpoint="chat/completions",
            kwargs={"model": "google/gemini-2.5-flash"},
            api_key=settings.OPENROUTER_API_KEY or "dummy-key-for-testing",
            auth_type="bearer",
            content_type="application/json",
            method="POST",
        )
        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump(exclude_none=True)
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)


class GroqChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        _config = {
            "name": "groq_chat",
            "provider": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "endpoint": "chat/completions",
            "api_key": settings.GROQ_API_KEY or "dummy-key-for-testing",
            "auth_type": "bearer",
            "content_type": "application/json",
            "method": "POST",
            "context_window": 128_000,  # Groq context window
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)


class OpenaiEmbedEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        from ...third_party.openai_models import CreateEmbeddingRequest

        _config = {
            "name": "openai_embed",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "embeddings",
            "kwargs": {"model": "text-embedding-3-small"},
            "api_key": settings.OPENAI_API_KEY or "dummy-key-for-testing",
            "auth_type": "bearer",
            "content_type": "application/json",
            "method": "POST",
            "request_options": CreateEmbeddingRequest,
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)
