# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Sophisticated iModel - v1 architecture with v0 feature depth."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from lionagi.ln._types import IDType
from lionagi.ln._utils import ID
from lionagi.ln.concurrency import create_task_group, effective_deadline, fail_at

from .core import CallContext, Service, ServiceError
from .endpoint import ChatRequestModel, RequestModel
from .executor import ExecutorConfig, RateLimitedExecutor, ServiceCall
from .hooks import HookedMiddleware, HookRegistry, HookType
from .middleware import MetricsMW, PolicyGateMW, RedactionMW
from .openai import (
    OpenAICompatibleService,
    create_anthropic_service,
    create_ollama_service,
    create_openai_service,
)
from .provider_detection import detect_provider_from_model, infer_provider_config

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetadata:
    """Provider-specific metadata and session state."""

    session_id: str | None = None
    rate_limits: dict[str, Any] = field(default_factory=dict)
    custom_headers: dict[str, str] = field(default_factory=dict)
    last_used: float = field(default_factory=time.time)


class iModel:
    """Sophisticated AI model interface with v0 feature depth in v1 architecture.

    Features from v0:
    - Rate limiting and intelligent queuing
    - Hook system for lifecycle events
    - Provider auto-detection and intelligence
    - Event-driven architecture with state tracking
    - Concurrency control and streaming management
    - Full serialization support

    Enhanced for v1:
    - Structured concurrency with AnyIO
    - Capability-based security
    - Clean middleware composition
    - Better error handling and observability
    """

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str = "chat",
        # Rate limiting and queuing
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60.0,
        interval: float | None = None,
        limit_requests: int | None = None,
        limit_tokens: int | None = None,
        concurrency_limit: int | None = None,
        # Hooks and middleware
        hook_registry: HookRegistry | None = None,
        enable_policy: bool = True,
        enable_metrics: bool = True,
        enable_redaction: bool = True,
        # Metadata
        provider_metadata: dict | None = None,
        id: IDType | str | None = None,
        created_at: float | None = None,
        **kwargs,
    ):
        """Initialize sophisticated iModel.

        Args:
            provider: Provider name or auto-detect from model
            model: Model name, can include provider prefix (e.g., "openai/gpt-4")
            api_key: API key or auto-detect from environment
            base_url: Custom base URL
            endpoint: Endpoint type (chat, completion, etc.)
            queue_capacity: Maximum queued calls
            limit_requests: Request rate limit per refresh period
            limit_tokens: Token rate limit per refresh period
            concurrency_limit: Max concurrent streaming calls
            hook_registry: Custom hook registry
            **kwargs: Additional provider-specific parameters
        """

        # Identity and timestamps (v0 style)
        self.id = ID.get_id(id) if id is not None else IDType.create()
        self.created_at = created_at or time.time()

        # Provider intelligence - auto-detect if not specified
        if model and not provider:
            if "/" in model:
                provider, model = model.split("/", 1)
            else:
                provider = detect_provider_from_model(model)

        if not provider:
            raise ValueError("Provider must be specified or detectable from model")

        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.endpoint_name = endpoint

        # Auto-detect API key if not provided
        if api_key is None:
            api_key = self._detect_api_key(provider)

        # Build service using provider intelligence
        self.service = self._create_service(
            provider=provider, api_key=api_key, base_url=base_url, model=model, **kwargs
        )

        # Rate limiting and queuing (v0 feature depth)
        executor_config = ExecutorConfig(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            interval=interval,
            limit_requests=limit_requests,
            limit_tokens=limit_tokens,
            concurrency_limit=concurrency_limit,
        )
        self.executor = RateLimitedExecutor(executor_config)

        # Hook system (v0 style but v1 implementation)
        self.hook_registry = hook_registry or HookRegistry()
        self._setup_middleware(enable_policy, enable_metrics, enable_redaction)

        # Provider metadata and session state
        self.provider_metadata = ProviderMetadata()
        if provider_metadata:
            for key, value in provider_metadata.items():
                setattr(self.provider_metadata, key, value)

    def _detect_api_key(self, provider: str) -> str | None:
        """Auto-detect API key from environment."""
        env_vars = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "together": ["TOGETHER_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
        }

        for env_var in env_vars.get(provider, []):
            api_key = os.getenv(env_var)
            if api_key:
                logger.debug(f"Auto-detected API key for {provider} from {env_var}")
                return api_key

        # Generic fallback
        generic_key = os.getenv(f"{provider.upper()}_API_KEY")
        if generic_key:
            logger.debug(f"Auto-detected API key for {provider} from generic env var")
            return generic_key

        return None

    def _create_service(
        self, provider: str, api_key: str | None, base_url: str | None, model: str | None, **kwargs
    ) -> Service:
        """Create service based on provider intelligence."""

        # Provider-specific service creation
        if provider == "openai":
            return create_openai_service(api_key=api_key or "", base_url=base_url, **kwargs)
        elif provider == "anthropic":
            return create_anthropic_service(api_key=api_key or "", base_url=base_url, **kwargs)
        elif provider == "ollama":
            return create_ollama_service(base_url=base_url or "http://localhost:11434/v1", **kwargs)
        else:
            # Generic OpenAI-compatible service
            from .openai import create_generic_service

            return create_generic_service(
                api_key=api_key or "",
                base_url=base_url or f"https://api.{provider}.com/v1",
                name=provider,
                **kwargs,
            )

    def _setup_middleware(self, enable_policy: bool, enable_metrics: bool, enable_redaction: bool):
        """Setup middleware stack with hooks integration."""
        # Hook middleware (always included)
        hook_middleware = HookedMiddleware(self.hook_registry)

        # Build middleware stack (order matters - hooks should be innermost)
        middleware_stack = [hook_middleware]
        stream_middleware_stack = [hook_middleware.stream]

        if enable_redaction:
            redaction_mw = RedactionMW()
            middleware_stack.insert(0, redaction_mw)
            stream_middleware_stack.insert(0, redaction_mw.stream)

        if enable_metrics:
            metrics_mw = MetricsMW()
            middleware_stack.insert(0, metrics_mw)
            stream_middleware_stack.insert(0, metrics_mw.stream)

        if enable_policy:
            policy_mw = PolicyGateMW()
            middleware_stack.insert(0, policy_mw)
            stream_middleware_stack.insert(0, policy_mw.stream)

        # Update service middleware
        if hasattr(self.service, "call_mw"):
            self.service.call_mw = tuple(middleware_stack)
        if hasattr(self.service, "stream_mw"):
            self.service.stream_mw = tuple(stream_middleware_stack)

    async def invoke(self, request: RequestModel | dict | None = None, **kwargs) -> Any:
        """Invoke API call with sophisticated queuing and state management.

        This is the main method that matches v0's invoke() behavior but with v1 architecture.
        """
        # Build request from kwargs if not provided
        if request is None:
            request = self._build_request(**kwargs)
        elif isinstance(request, dict):
            request = self._build_request(**request)

        # Create call context with deadline awareness
        context = self._build_context(**kwargs)

        # Add service name to context for hooks
        context.attrs = dict(context.attrs) if context.attrs else {}
        context.attrs["service_name"] = self.service.name

        # Submit to executor for rate limiting and queuing
        call = await self.executor.submit_call(self.service, request, context)

        # Wait for completion (this handles the queuing/processing)
        result = await call.wait_completion()

        # Handle provider-specific post-processing (like v0's session management)
        await self._post_process_result(call, result)

        return result

    async def stream(
        self, request: RequestModel | dict | None = None, **kwargs
    ) -> AsyncIterator[Any]:
        """Stream API call with sophisticated concurrency control."""
        # Build request
        if request is None:
            request = self._build_request(stream=True, **kwargs)
        elif isinstance(request, dict):
            request = self._build_request(stream=True, **request)
        else:
            # Ensure streaming is enabled
            if hasattr(request, "model_copy"):
                request = request.model_copy(update={"stream": True})
            else:
                request.stream = True

        # Create context
        context = self._build_context(**kwargs)
        context.attrs = dict(context.attrs) if context.attrs else {}
        context.attrs["service_name"] = self.service.name

        # Stream through executor (handles concurrency limiting)
        async for chunk in self.executor.submit_stream(self.service, request, context):
            yield chunk

    def _build_request(self, **kwargs) -> RequestModel:
        """Build request model from parameters."""
        # Use model from instance if not specified
        if "model" not in kwargs and self.model:
            kwargs["model"] = self.model

        # Default to chat request
        return ChatRequestModel(**kwargs)

    def _build_context(self, **kwargs) -> CallContext:
        """Build call context with deadline awareness."""
        # Extract context-specific parameters
        timeout_s = kwargs.pop("timeout_s", None)
        deadline_s = kwargs.pop("deadline_s", None)
        capabilities = kwargs.pop("capabilities", None)
        branch_id = kwargs.pop("branch_id", None)

        # Use effective deadline if available
        if deadline_s is None and timeout_s is not None:
            import anyio

            deadline_s = anyio.current_time() + timeout_s
        elif deadline_s is None:
            deadline_s = effective_deadline()  # Use ambient deadline if exists

        # Generate branch_id if not provided
        if branch_id is None:
            branch_id = uuid4()

        # Build capabilities from service requirements
        service_capabilities = getattr(self.service, "requires", set())
        all_capabilities = service_capabilities.copy()
        if capabilities:
            all_capabilities.update(capabilities)

        return CallContext(
            call_id=uuid4(),
            branch_id=branch_id,
            deadline_s=deadline_s,
            capabilities=all_capabilities,
            attrs=kwargs,  # Pass remaining kwargs as attrs
        )

    async def _post_process_result(self, call: ServiceCall, result: Any) -> None:
        """Handle provider-specific post-processing like session management."""
        # Claude Code session management (like v0)
        if self.provider == "claude_code" and isinstance(result, dict) and "session_id" in result:
            self.provider_metadata.session_id = result["session_id"]
            logger.debug(f"Updated Claude Code session_id: {result['session_id']}")

        # Update last used timestamp
        self.provider_metadata.last_used = time.time()

    async def start(self) -> None:
        """Start the executor (like v0's processor)."""
        await self.executor.start()

    async def stop(self) -> None:
        """Stop the executor and cleanup."""
        await self.executor.stop()

    # Properties for compatibility with v0
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model or ""

    @property
    def stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return self.executor.stats

    # Serialization support (v0 compatibility)
    def to_dict(self) -> dict[str, Any]:
        """Serialize iModel to dictionary."""
        return {
            "id": str(self.id),
            "created_at": self.created_at,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": self.endpoint_name,
            "executor_config": {
                "queue_capacity": self.executor.config.queue_capacity,
                "capacity_refresh_time": self.executor.config.capacity_refresh_time,
                "limit_requests": self.executor.config.limit_requests,
                "limit_tokens": self.executor.config.limit_tokens,
                "concurrency_limit": self.executor.config.concurrency_limit,
            },
            "provider_metadata": {
                "session_id": self.provider_metadata.session_id,
                "rate_limits": self.provider_metadata.rate_limits,
                "custom_headers": self.provider_metadata.custom_headers,
                "last_used": self.provider_metadata.last_used,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> iModel:
        """Deserialize iModel from dictionary."""
        executor_config = data.get("executor_config", {})
        provider_metadata = data.get("provider_metadata", {})

        return cls(
            provider=data.get("provider"),
            model=data.get("model"),
            base_url=data.get("base_url"),
            endpoint=data.get("endpoint", "chat"),
            id=data.get("id"),
            created_at=data.get("created_at"),
            provider_metadata=provider_metadata,
            **executor_config,
        )

    # Context manager support
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Convenience functions (keeping v1 ergonomics but with v0 power)


async def quick_chat(
    messages: list[dict],
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    timeout_s: float = 30.0,
    **kwargs,
) -> dict:
    """Quick chat with full v0 sophistication under the hood."""
    async with iModel(provider=provider, model=model, api_key=api_key, **kwargs) as im:
        return await im.invoke(
            messages=messages,
            timeout_s=timeout_s,
        )


async def quick_stream(
    messages: list[dict],
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    timeout_s: float = 30.0,
    **kwargs,
) -> AsyncIterator[dict]:
    """Quick streaming with full sophistication."""
    async with iModel(provider=provider, model=model, api_key=api_key, **kwargs) as im:
        async for chunk in im.stream(
            messages=messages,
            timeout_s=timeout_s,
        ):
            yield chunk
