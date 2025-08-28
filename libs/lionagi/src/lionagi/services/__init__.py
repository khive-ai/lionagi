# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Lion v1 Services - Clean, composable service architecture.

This module implements the v1 services architecture with:
- Single mental model: structured operations with deadlines and explicit capabilities
- Strict boundaries: separate concerns (request shaping, transport, resilience, scheduling)
- Composability: middleware-style hooks
- Deterministic lifecycle: clear phases with consistent cancellation/timeout behavior
- First-class streaming: proper backpressure and cancellation
- Observability: per-call IDs, metrics, traces
- Ergonomics: delightful defaults
"""

from lionagi.errors import PolicyError, ServiceError, TimeoutError

from .adapters import GenericJSONAdapter, OpenAIAdapter
from .core import CallContext, Service
from .endpoint import (
    ChatRequestModel,
    CompletionRequestModel,
    EmbeddingRequestModel,
    Endpoint,
    RequestModel,
)
from .executor import CallStatus, ExecutorConfig, RateLimitedExecutor, ServiceCall
from .hooks import (
    HookedMiddleware,
    HookEvent,
    HookRegistry,
    HookType,
    get_global_hooks,
    hook,
    stream_hook,
)
from .imodel import ProviderMetadata, iModel, quick_chat, quick_stream
from .middleware import CallMW, MetricsMW, PolicyGateMW, RedactionMW, StreamMW
from .openai import (
    OpenAICompatibleService,
    create_anthropic_service,
    create_generic_service,
    create_ollama_service,
    create_openai_service,
)
from .provider_registry import get_provider_registry, register_builtin_adapters
from .resilience import CircuitBreakerConfig, RetryConfig, create_resilience_mw
from .transport import HTTPXTransport, Transport

# Register built-in adapters once on import
register_builtin_adapters()

__all__ = [
    # Core interfaces
    "CallContext",
    "Service",
    "Endpoint",
    "Transport",
    "RequestModel",
    "ChatRequestModel",
    "CompletionRequestModel",
    "EmbeddingRequestModel",
    # Errors
    "ServiceError",
    "TimeoutError",
    "PolicyError",
    # Executor and rate limiting
    "RateLimitedExecutor",
    "ExecutorConfig",
    "CallStatus",
    "ServiceCall",
    # Hook system
    "HookRegistry",
    "HookType",
    "HookEvent",
    "hook",
    "stream_hook",
    "get_global_hooks",
    # Middleware
    "CallMW",
    "StreamMW",
    "PolicyGateMW",
    "MetricsMW",
    "RedactionMW",
    "HookedMiddleware",
    # OpenAI-compatible services
    "OpenAICompatibleService",
    "create_openai_service",
    "create_anthropic_service",
    "create_ollama_service",
    "create_generic_service",
    # Transport implementations
    "HTTPXTransport",
    # Resilience
    "RetryConfig",
    "CircuitBreakerConfig",
    "create_resilience_mw",
    # Sophisticated iModel (main interface)
    "iModel",
    "ProviderMetadata",
    "quick_chat",
    "quick_stream",
    # Provider registry
    "get_provider_registry",
    "register_builtin_adapters",
]
