diff proposed for services/


-----GEMINI------

This solution provides the concrete implementation diffs and new files required to transition LION v1 Services to a hybrid provider configuration architecture. This architecture eliminates brittle regex detection while ensuring maximum modularity, performance, and security by implementing a dynamic registry and adapter pattern.

### 1\. File Deletion

The file `lionagi/services/provider_detection.py` is deprecated and should be deleted.

### 2\. New File: `lionagi/services/registry.py`

This new file introduces the `ServiceRegistry`.

```python
# lionagi/services/registry.py (New File)
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Service Registry for pluggable service implementations."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

# Assuming these modules are in the same directory based on context
from .core import Service
from .provider_config import ProviderConfig

logger = logging.getLogger(__name__)

# Standardized Service Factory signature
# Factories take the config, optional API key, and kwargs dict, returning a Service.
ServiceFactory = Callable[[ProviderConfig, str | None, Dict[str, Any]], Service]

class ServiceRegistry:
    """Registry for managing service factories based on adapter types or provider names."""
    
    def __init__(self):
        # Maps AdapterType or specific Provider names (strings) to factories
        self._factories: Dict[str, ServiceFactory] = {}
        self._initialized = False

    def initialize_defaults(self):
        """Register default adapters."""
        if self._initialized:
            return
        
        # Avoid circular imports by importing factories here
        try:
            # This requires the factory function to be implemented in openai.py
            from .openai import factory_openai_compatible
            self.register("openai_compatible", factory_openai_compatible)
        except ImportError:
            logger.warning("Could not import built-in OpenAI compatible adapter.")
        
        # Future defaults: self.register("anthropic_native", ...)
        self._initialized = True

    def register(self, key: str, factory: ServiceFactory) -> None:
        """Register a custom service factory.
        
        Args:
            key: The adapter type (e.g., "openai_compatible") or provider name (e.g., "database_query").
            factory: The function that creates the Service instance.
        """
        if not self._initialized:
            self.initialize_defaults()

        if key in self._factories:
            logger.warning(f"Overwriting existing factory registration for: {key}")
        self._factories[key] = factory
        logger.info(f"Registered service factory for: {key}")

    def create_service(self, config: ProviderConfig, api_key: str | None, factory_kwargs: Dict[str, Any]) -> Service:
        """Create a Service instance using the registered factory."""
        if not self._initialized:
            self.initialize_defaults()

        # Determine the lookup key:
        # 1. Prioritize custom registration by provider name (enables v0 modularity like ClaudeCodeCLI).
        # 2. Fall back to the adapter type (standardized implementation).
        lookup_key = config.name if config.name in self._factories else config.adapter_type

        factory = self._factories.get(lookup_key)

        if not factory:
            raise ValueError(
                f"No factory found for provider '{config.name}' or adapter '{config.adapter_type}'. "
                f"Ensure the factory is registered."
            )

        logger.debug(f"Creating service '{config.name}' using factory for '{lookup_key}'")
        
        # Call the factory with standardized arguments
        return factory(config, api_key, factory_kwargs)

# Global service registry instance
registry = ServiceRegistry()
```

### 3\. Refactoring `lionagi/services/provider_config.py`

We refactor `provider_config.py` to introduce `AdapterType`, allow flexible provider identifiers, and implement strict, explicit resolution logic. We also centralize capability inference within the `ProviderConfig` struct.

```python
# lionagi/services/provider_config.py (Complete Replacement)
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Explicit provider configuration system and resolution logic."""

from __future__ import annotations

import msgspec
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

# Define supported wire protocols (Adapters)
AdapterType = Literal[
    "openai_compatible",
    # Future extensions: "anthropic_native", "google_vertex", "custom"
]

# Allows any string for dynamic providers
ProviderIdentifier: TypeAlias = str

# Known provider names for built-in configurations
KnownProviderType = Literal[
    "openai", "anthropic", "google", "cohere", "together", "openrouter",
    "groq", "fireworks", "ollama", "perplexity", "claude_code"
]

AuthType = Literal["bearer", "x-api-key", "none"]

@msgspec.Struct
class ProviderConfig:
    """Explicit provider configuration - supports known and dynamic providers."""
    
    name: ProviderIdentifier
    base_url: str
    adapter_type: AdapterType = "openai_compatible"
    
    auth_type: AuthType = "bearer"
    auth_prefix: str = "Bearer"
    supports_streaming: bool = True
    supports_functions: bool = True
    max_tokens_field: str = "max_tokens"
    temperature_field: str = "temperature" 
    custom_headers: dict[str, str] | None = None
    default_model: str | None = None
    
    # Crucial for security: Capability requirements
    requires: set[str] = msgspec.field(default_factory=set)

    def __post_init__(self):
        # Auto-infer network capability if not explicitly set (Security Requirement)
        if not self.requires and self.base_url:
            try:
                parsed = urlparse(self.base_url)
                if parsed.netloc:
                    self.requires.add(f"net.out:{parsed.netloc}")
                elif self.base_url.startswith(('http', 'https')):
                     logger.warning(f"Could not infer network capabilities for '{self.name}' from base_url: {self.base_url}")
            except Exception as e:
                logger.error(f"Error inferring capabilities for '{self.name}': {e}")

# Factory functions (requires is handled by __post_init__)

def _create_openai_config(**overrides: Any) -> ProviderConfig:
    return ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        adapter_type="openai_compatible",
        default_model="gpt-4o-mini",
        **overrides
    )

def _create_anthropic_config(**overrides: Any) -> ProviderConfig:
    # Using OpenAI compatibility adapter as per the context provided in openai.py
    return ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1/messages",
        adapter_type="openai_compatible",
        auth_type="x-api-key",
        auth_prefix="",
        default_model="claude-3-5-sonnet-20241022",
        **overrides
    )

# ... (Implementations for Google, Cohere, Together, OpenRouter, Groq, Fireworks, Perplexity, ClaudeCode omitted for brevity but follow the pattern)

def _create_ollama_config(**overrides: Any) -> ProviderConfig:
    return ProviderConfig(
        name="ollama",
        base_url="http://localhost:11434/v1",
        adapter_type="openai_compatible",
        auth_type="none",
        default_model="llama3.2:latest",
        **overrides
    )

# Explicit provider configuration registry
KNOWN_PROVIDER_CONFIGS: dict[KnownProviderType, ProviderConfig] = {
    "openai": _create_openai_config(),
    "anthropic": _create_anthropic_config(),
    # ... (Add all known providers from the original file)
    "ollama": _create_ollama_config(),
}

def get_provider_config(
    provider: ProviderIdentifier, 
    base_url: str | None = None,
    adapter_type: AdapterType | None = None,
    **overrides: Any
) -> ProviderConfig:
    """Get or dynamically create provider configuration."""

    # 1. Check Known Providers
    if provider in KNOWN_PROVIDER_CONFIGS:
        config = KNOWN_PROVIDER_CONFIGS[provider] # type: ignore
        is_dynamic = False
    else:
        # 2. Dynamic Provider Creation
        if not base_url:
            # If base_url is missing and provider is unknown, we cannot proceed.
            raise ValueError(f"base_url is required for unknown/custom provider: '{provider}'")
        
        config = ProviderConfig(
            name=provider,
            base_url=base_url,
            adapter_type=adapter_type or "openai_compatible",
        )
        is_dynamic = True

    # Apply overrides (msgspec Structs are immutable, create a copy if needed)
    overrides_dict = {}
    # Apply base_url/adapter_type overrides only if not dynamically creating based on them
    if base_url and not is_dynamic: 
        overrides_dict["base_url"] = base_url
    if adapter_type and not is_dynamic:
        overrides_dict["adapter_type"] = adapter_type
    overrides_dict.update(overrides)

    if not overrides_dict:
        return config

    # Create new config with overrides
    config_dict = msgspec.to_builtins(config)
    config_dict.update(overrides_dict)
    # Re-instantiate to trigger __post_init__ for capability updates if base_url changed
    return ProviderConfig(**config_dict)


def parse_provider_from_model(model: str) -> tuple[ProviderIdentifier | None, str]:
    """Parse provider from model name prefix (e.g., "openai/gpt-4" or "custom_api/v1"). Flexible."""
    if not model or "/" not in model:
        return None, model
        
    parts = model.split("/", 1)
    potential_provider = parts[0].lower().strip()
    
    # We accept any non-empty string as a potential provider prefix
    if potential_provider:
        return potential_provider, parts[1]
    
    return None, model

def resolve_provider_and_model(
    provider: str | None = None, 
    model: str | None = None
) -> tuple[ProviderIdentifier, str | None]:
    """Resolve provider and model robustly and explicitly. No guessing (replaces regex)."""
    
    # 1. Handle cases without a model
    if not model:
        if provider:
            return provider.lower().strip(), None
        else:
            raise ValueError("Either 'provider' or 'model' (with prefix) must be specified.")

    # 2. Try parsing provider prefix from model (Flexible parsing)
    parsed_provider, clean_model = parse_provider_from_model(model)
    
    explicit_provider = provider.lower().strip() if provider else None

    # 3. Resolution Logic
    if explicit_provider and parsed_provider:
        if explicit_provider != parsed_provider:
            raise ValueError(
                f"Provider conflict: Explicitly provided '{explicit_provider}' does not match model prefix '{parsed_provider}'."
            )
        return explicit_provider, clean_model
            
    if explicit_provider:
        # Use explicit provider and the original model name
        return explicit_provider, model

    if parsed_provider:
        # Use parsed provider and clean model name
        return parsed_provider, clean_model
    
    # 4. Ambiguity: No explicit provider AND no prefix found.
    raise ValueError(
        f"Provider cannot be determined for model '{model}'. "
        f"Specify the provider explicitly (e.g., provider='my_api') or use the prefix format (e.g., model='my_api/{model}')."
    )
```

### 4\. Refactoring `lionagi/services/openai.py`

We refactor `openai.py` to implement the standardized `factory_openai_compatible` function used by the registry. This centralizes authentication logic for this adapter type.

```python
# lionagi/services/openai.py (Diff)

# ... (standard imports)
import asyncio
+import os # Added import
 from dataclasses import dataclass
-from typing import Any, AsyncIterator
+from typing import Any, AsyncIterator, Dict # Added Dict

# ... (openai SDK imports)

from lionagi.errors import NonRetryableError, RetryableError, ServiceError, TimeoutError, RateLimitError
 from .core import CallContext
 from .endpoint import ChatRequestModel, RequestModel
 from .middleware import CallMW, StreamMW
+from .provider_config import ProviderConfig # Added import

@dataclass(slots=True)
class OpenAICompatibleService:
    # ... (The Service implementation (OpenAICompatibleService class) remains exactly the same as provided in the context file)
    # ... (call, stream, _build_call_kwargs methods remain the same)
    pass


# --- NEW: Adapter Factory Implementation ---
# This replaces all previous factory functions (create_openai_service, create_anthropic_service, etc.)

def factory_openai_compatible(
    config: ProviderConfig,
    api_key: str | None,
    factory_kwargs: Dict[str, Any],
) -> OpenAICompatibleService:
    """Factory function for the 'openai_compatible' adapter. Registered with ServiceRegistry."""
    
    client_kwargs = factory_kwargs.copy()
    headers = config.custom_headers.copy() if config.custom_headers else {}

    # 1. Determine API Key (Centralized Auth Logic)
    # This replaces iModel._detect_api_key
    
    final_api_key = api_key
    
    if not final_api_key and config.auth_type != "none":
        # Attempt auto-detection from environment if not provided
        env_key_name = f"{config.name.upper()}_API_KEY"
        final_api_key = os.getenv(env_key_name)
        
        # Specific common fallbacks (convenience for known providers)
        if not final_api_key:
            if config.name == "openai":
                final_api_key = os.getenv("OPENAI_API_KEY")
            elif config.name == "anthropic":
                 final_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")

    # 2. Configure Auth based on ProviderConfig.auth_type
    if config.auth_type == "x-api-key":
        if not final_api_key:
             raise ValueError(f"API key required for provider '{config.name}' (auth_type='x-api-key') but not found.")
        # e.g., Anthropic style auth when using their OpenAI endpoint
        headers["x-api-key"] = final_api_key
        sdk_api_key = final_api_key 
    elif config.auth_type == "none":
        # Local services like Ollama often don't need a real key
        sdk_api_key = final_api_key or "local_dummy_key"
    else:
        # Standard Bearer auth (default)
        if not final_api_key:
             raise ValueError(f"API key required for provider '{config.name}' (auth_type='bearer') but not found.")
        sdk_api_key = final_api_key

    # 3. Initialize the Client
    client = AsyncOpenAI(
        api_key=sdk_api_key,
        base_url=config.base_url,
        default_headers=headers if headers else None,
        **client_kwargs,
    )

    # 4. Create the Service Instance
    # Capabilities (requires) are derived from ProviderConfig (ensuring security)
    return OpenAICompatibleService(
        client=client,
        name=config.name,
        requires=config.requires,
        # Middleware (call_mw/stream_mw) will be injected later by iModel
    )
```

### 5\. Refactoring `lionagi/services/imodel.py`

We update `iModel` to integrate the new configuration and registry systems, ensuring correct handling of dynamic providers and security contexts.

```python
# lionagi/services/imodel.py (Diff)

import logging
-import os
import time
# ... (other imports)

from .core import CallContext, Service, ServiceError
from .endpoint import ChatRequestModel, RequestModel
from .executor import ExecutorConfig, RateLimitedExecutor, ServiceCall
from .hooks import HookedMiddleware, HookRegistry, HookType
from .middleware import MetricsMW, PolicyGateMW, RedactionMW
-from .openai import (
-    OpenAICompatibleService,
-    create_anthropic_service,
-    create_ollama_service,
-    create_openai_service,
-)
-from .provider_detection import detect_provider_from_model, infer_provider_config
+
+# New Imports
+from .provider_config import ProviderConfig, resolve_provider_and_model, get_provider_config, AdapterType
+from .registry import registry as service_registry

logger = logging.getLogger(__name__)

# ... (ProviderMetadata remains the same)

class iModel:
    # ...
    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
+       adapter_type: AdapterType | None = None, # Added adapter_type support
        endpoint: str = "chat",
        # ... (rest of args)
        **kwargs,
    ):
        # Identity and timestamps
        self.id = ID.get_id(id) if id is not None else IDType.create()
        self.created_at = created_at or time.time()

-        # Provider intelligence - auto-detect if not specified
-        if model and not provider:
-            if "/" in model:
-                provider, model = model.split("/", 1)
-            else:
-                provider = detect_provider_from_model(model)
-
-        if not provider:
-            raise ValueError("Provider must be specified or detectable from model")
-
-        self.provider = provider
-        self.model = model
-        self.base_url = base_url
+        # --- NEW Provider Resolution and Configuration Logic ---
+
+        # 1. Robust Provider Resolution (Explicit Only)
+        try:
+            # This handles explicit provider, prefixed model, or ambiguity.
+            resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
+        except ValueError as e:
+            # If resolution failed (e.g. ambiguous model name), check if we have enough info for a dynamic provider (explicit provider + base_url)
+            # This supports iModel(provider="custom", base_url="http://...", model="unstructured/name")
+            if provider and base_url:
+                 resolved_provider = provider
+                 resolved_model = model
+            else:
+                # If not enough info for dynamic provider and resolution failed, raise error.
+                raise ValueError(f"Cannot initialize iModel: {e}") from e
+
+        self.provider = resolved_provider
+        self.model = resolved_model
         self.endpoint_name = endpoint
 
-        # Auto-detect API key if not provided
-        if api_key is None:
-            api_key = self._detect_api_key(provider)
-
-        # Build service using provider intelligence
-        self.service = self._create_service(
-            provider=provider, api_key=api_key, base_url=base_url, model=model, **kwargs
-        )
+        # 2. Configuration Retrieval (Handles known and dynamic providers)
+        try:
+            # Pass user inputs for overrides or dynamic creation.
+            self.config: ProviderConfig = get_provider_config(
+                provider=resolved_provider,
+                base_url=base_url,
+                adapter_type=adapter_type,
+                # Pass relevant kwargs that might override config defaults (e.g., auth_type)
+                **{k: v for k, v in kwargs.items() if k in ProviderConfig.__struct_fields__}
+            )
+        except ValueError as e:
+            raise ValueError(f"Failed to configure provider '{resolved_provider}': {e}") from e
+
+        self.base_url = self.config.base_url
+
+        # 3. Service Creation via Registry
+        try:
+            # Pass remaining kwargs to the service factory
+            factory_kwargs = {k: v for k, v in kwargs.items() if k not in ProviderConfig.__struct_fields__}
+            # API Key detection is handled within the factory (adapter implementation)
+            self.service = service_registry.create_service(
+                config=self.config, 
+                api_key=api_key, 
+                factory_kwargs=factory_kwargs
+            )
+        except ValueError as e:
+            raise ValueError(f"Failed to create service for provider '{self.provider}' (Adapter: {self.config.adapter_type}): {e}") from e
+
+        # --- End of New Logic ---
 
         # Rate limiting and queuing (v0 feature depth)
         executor_config = ExecutorConfig(
@@ -168,63 +180,13 @@
         if provider_metadata:
             for key, value in provider_metadata.items():
                 setattr(self.provider_metadata, key, value)
-
-    def _detect_api_key(self, provider: str) -> str | None:
-        """Auto-detect API key from environment."""
-        env_vars = {
-            "openai": ["OPENAI_API_KEY"],
-            "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
-            "together": ["TOGETHER_API_KEY"],
-            "openrouter": ["OPENROUTER_API_KEY"],
-            "groq": ["GROQ_API_KEY"],
-            "fireworks": ["FIREWORKS_API_KEY"],
-        }
-
-        for env_var in env_vars.get(provider, []):
-            api_key = os.getenv(env_var)
-            if api_key:
-                logger.debug(f"Auto-detected API key for {provider} from {env_var}")
-                return api_key
-
-        # Generic fallback
-        generic_key = os.getenv(f"{provider.upper()}_API_KEY")
-        if generic_key:
-            logger.debug(f"Auto-detected API key for {provider} from generic env var")
-            return generic_key
-
-        return None
-
-    def _create_service(
-        self, provider: str, api_key: str | None, base_url: str | None, model: str | None, **kwargs
-    ) -> Service:
-        """Create service based on provider intelligence."""
-
-        # Provider-specific service creation
-        if provider == "openai":
-            return create_openai_service(api_key=api_key or "", base_url=base_url, **kwargs)
-        elif provider == "anthropic":
-            return create_anthropic_service(api_key=api_key or "", base_url=base_url, **kwargs)
-        elif provider == "ollama":
-            return create_ollama_service(base_url=base_url or "http://localhost:11434/v1", **kwargs)
-        else:
-            # Generic OpenAI-compatible service
-            from .openai import create_generic_service
-
-            return create_generic_service(
-                api_key=api_key or "",
-                base_url=base_url or f"https://api.{provider}.com/v1",
-                name=provider,
-                **kwargs,
-            )
-
+    
+    # REMOVED: _detect_api_key (Moved to adapter factories, e.g., openai.py)
+    # REMOVED: _create_service (Replaced by ServiceRegistry)
+    
     def _setup_middleware(self, enable_policy: bool, enable_metrics: bool, enable_redaction: bool):
-        """Setup middleware stack with hooks integration."""
-        # Hook middleware (always included)
-        hook_middleware = HookedMiddleware(self.hook_registry)
-
-        # Build middleware stack (order matters - hooks should be innermost)
-        middleware_stack = [hook_middleware]
-        stream_middleware_stack = [hook_middleware.stream]
-
-        if enable_redaction:
-            redaction_mw = RedactionMW()
-            middleware_stack.insert(0, redaction_mw)
-            stream_middleware_stack.insert(0, redaction_mw.stream)
-
-        if enable_metrics:
-            metrics_mw = MetricsMW()
-            middleware_stack.insert(0, metrics_mw)
-            stream_middleware_stack.insert(0, metrics_mw.stream)
-
-        if enable_policy:
-            policy_mw = PolicyGateMW()
-            middleware_stack.insert(0, policy_mw)
-            stream_middleware_stack.insert(0, policy_mw.stream)
-
-        # Update service middleware
-        if hasattr(self.service, "call_mw"):
-            self.service.call_mw = tuple(middleware_stack)
-        if hasattr(self.service, "stream_mw"):
-            self.service.stream_mw = tuple(stream_middleware_stack)
+        # ... (Implementation remains the same)
 
     async def invoke(self, request: RequestModel | dict | None = None, **kwargs) -> Any:
-        """Invoke API call with sophisticated queuing and state management.
-
-        This is the main method that matches v0's invoke() behavior but with v1 architecture.
-        """
-        # Build request from kwargs if not provided
-        if request is None:
-            request = self._build_request(**kwargs)
-        elif isinstance(request, dict):
-            request = self._build_request(**request)
-
-        # Create call context with deadline awareness
-        context = self._build_context(**kwargs)
-
-        # Add service name to context for hooks
-        context.attrs = dict(context.attrs) if context.attrs else {}
-        context.attrs["service_name"] = self.service.name
-
-        # Submit to executor for rate limiting and queuing
-        call = await self.executor.submit_call(self.service, request, context)
-
-        # Wait for completion (this handles the queuing/processing)
-        result = await call.wait_completion()
-
-        # Handle provider-specific post-processing (like v0's session management)
-        await self._post_process_result(call, result)
-
-        return result
+        # ... (Implementation remains the same)
 
     async def stream(
         self, request: RequestModel | dict | None = None, **kwargs
     ) -> AsyncIterator[Any]:
-        """Stream API call with sophisticated concurrency control."""
-        # Build request
-        if request is None:
-            request = self._build_request(stream=True, **kwargs)
-        elif isinstance(request, dict):
-            request = self._build_request(stream=True, **request)
-        else:
-            # Ensure streaming is enabled
-            if hasattr(request, "model_copy"):
-                request = request.model_copy(update={"stream": True})
-            else:
-                request.stream = True
-
-        # Create context
-        context = self._build_context(**kwargs)
-        context.attrs = dict(context.attrs) if context.attrs else {}
-        context.attrs["service_name"] = self.service.name
-
-        # Stream through executor (handles concurrency limiting)
-        async for chunk in self.executor.submit_stream(self.service, request, context):
-            yield chunk
+        # ... (Implementation remains the same)
 
     def _build_request(self, **kwargs) -> RequestModel:
         """Build request model from parameters."""
         # Use model from instance if not specified
         if "model" not in kwargs and self.model:
             kwargs["model"] = self.model
+        
+        # If model is still missing, check if config has a default
+        if "model" not in kwargs and self.config.default_model:
+             kwargs["model"] = self.config.default_model
 
         # Default to chat request
         return ChatRequestModel(**kwargs)
@@ -278,20 +240,22 @@
         if branch_id is None:
             branch_id = uuid4()
 
-        # Build capabilities from service requirements
-        service_capabilities = getattr(self.service, "requires", set())
-        all_capabilities = service_capabilities.copy()
+        # Build capabilities (what the caller IS allowed to do)
+        all_capabilities = set()
         if capabilities:
             all_capabilities.update(capabilities)
+
+        # CRITICAL: Pass service requirements into context attrs for PolicyGateMW
+        # The requirements are derived from the ProviderConfig (which the service uses)
+        # This ensures capability-based security (Security Requirement).
+        service_requires = self.config.requires
+
+        attrs = kwargs.copy()
+        # Inject requirements and metadata for middleware access
+        attrs["service_requires"] = service_requires
 
         return CallContext(
             call_id=uuid4(),
             branch_id=branch_id,
             deadline_s=deadline_s,
             capabilities=all_capabilities,
-            attrs=kwargs,  # Pass remaining kwargs as attrs
+            attrs=attrs,
         )
 
     async def _post_process_result(self, call: ServiceCall, result: Any) -> None:
@@ -320,6 +284,7 @@
             "provider": self.provider,
             "model": self.model,
             "base_url": self.base_url,
+            "adapter_type": self.config.adapter_type, # Add adapter type
             "endpoint": self.endpoint_name,
             "executor_config": {
 # ... (rest of serialization)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> iModel:
        # ... (Update deserialization to include adapter_type)
        return cls(
            provider=data.get("provider"),
            model=data.get("model"),
            base_url=data.get("base_url"),
+           adapter_type=data.get("adapter_type"),
            endpoint=data.get("endpoint", "chat"),
            # ...
        )
# ... (rest of file)
```


-----CHATGPT------


