Below are **drop‑in diffs + new files** that implement the hybrid provider‑configuration you requested:

* **Robust & explicit resolution** (no brittle regex)
* **Maximum modularity** (plug any JSON endpoint as an iModel)
* **Type‑safe, msgspec‑based descriptors**
* **Security**: adapters expose `requires` → enforced by your `PolicyGateMW`
* **Streaming‑first**: no buffering added

I include targeted notes and a mini migration guide at the end (with citations to the earlier resilience/transport behavior for context  ).

---

## 0) New: `provider_registry.py`

**Path:** `libs/lionagi/src/lionagi/services/provider_registry.py`

```python
# Copyright (c) 2025, HaiyangLi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import msgspec

from .endpoint import RequestModel
from .core import Service

__all__ = [
    "ProviderAdapter",
    "ProviderRegistry",
    "get_default_registry",
    "resolve_provider_and_model",
]


@runtime_checkable
class ProviderAdapter(Protocol):
    """Adapter contract for wiring a provider into a Service."""

    #: canonical provider name, e.g. "openai", "anthropic", "generic_http"
    name: str
    #: default base URL (None for non-HTTP or when not applicable)
    default_base_url: str | None

    #: the default RequestModel type this adapter expects
    request_model: type[RequestModel]

    #: static capabilities (may be an empty set); 
    #: adapters may compute dynamic rights using base_url in create_service()
    requires: set[str]

    def supports(
        self,
        *,
        provider: str | None,
        model: str | None,
        base_url: str | None,
    ) -> bool:
        """Return True if this adapter can handle the given tuple."""
        ...

    def create_service(
        self,
        *,
        base_url: str | None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Service:
        """Instantiate a Service for this provider."""
        ...


class ProviderRegistry(msgspec.Struct, kw_only=True):
    """Runtime registry that resolves provider/model → adapter → Service."""
    _adapters: dict[str, ProviderAdapter] = msgspec.field(default_factory=dict)

    def register(self, adapter: ProviderAdapter) -> None:
        key = adapter.name.lower()
        if key in self._adapters:
            raise ValueError(f"Adapter already registered for provider '{key}'")
        self._adapters[key] = adapter

    def get(self, provider: str) -> ProviderAdapter | None:
        return self._adapters.get(provider.lower())

    def known(self) -> set[str]:
        return set(self._adapters.keys())

    def resolve(
        self,
        *,
        provider: str | None,
        model: str | None,
        base_url: str | None,
    ) -> ProviderAdapter:
        """Return a matching adapter or raise ValueError."""
        if provider:
            # direct lookup first
            found = self.get(provider)
            if found:
                return found
            # unknown provider → fall back to generic HTTP if base_url is HTTP(S)
            if base_url and base_url.startswith(("http://", "https://")):
                found = self.get("generic_http")
                if found and found.supports(provider=provider, model=model, base_url=base_url):
                    return found
            raise ValueError(
                f"Unknown provider '{provider}'. Registered: {sorted(self.known())}. "
                f"Provide a custom adapter or use base_url with 'generic_http'."
            )

        # provider not given → try inferring from model "provider/model"
        if model and "/" in model:
            pfx = model.split("/", 1)[0].strip().lower()
            found = self.get(pfx)
            if found:
                return found
            # Unknown prefixed provider → fall back to generic HTTP w/ base_url
            if base_url and base_url.startswith(("http://", "https://")):
                found = self.get("generic_http")
                if found and found.supports(provider=pfx, model=model, base_url=base_url):
                    return found
            raise ValueError(
                f"Model implies unknown provider '{pfx}'. Registered: {sorted(self.known())}."
            )

        # fully ambiguous → only valid if explicit base_url (+ generic_http)
        if base_url and base_url.startswith(("http://", "https://")):
            found = self.get("generic_http")
            if found:
                return found

        raise ValueError(
            "Provider must be specified or derivable from model 'provider/model', "
            "or an HTTP(S) base_url must be provided to use 'generic_http'."
        )


# -- parse helpers (regex-free) ------------------------------------------------
def _parse_provider_model(
    provider: str | None, model: str | None
) -> tuple[str | None, str | None]:
    """Normalize (provider, model). If model is 'p/m', strip prefix into provider."""
    if model and "/" in model:
        pfx, rest = model.split("/", 1)
        pfx = pfx.strip()
        if provider and provider.lower() != pfx.lower():
            raise ValueError(
                f"Provider mismatch: provider='{provider}' vs model prefix='{pfx}'."
            )
        return pfx, rest
    return provider, model


def resolve_provider_and_model(
    registry: ProviderRegistry,
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, str | None, ProviderAdapter]:
    """Return (provider, model, adapter). Raises ValueError on ambiguity."""
    provider, model = _parse_provider_model(provider, model)
    adapter = registry.resolve(provider=provider, model=model, base_url=base_url)
    if provider is None:
        # if adapter chosen via model prefix or generic, set provider = adapter.name
        provider = adapter.name
    return provider, model, adapter


# -- global default registry ---------------------------------------------------
_default_registry = ProviderRegistry()

def get_default_registry() -> ProviderRegistry:
    return _default_registry
```

---

## 1) New: generic JSON descriptor (HTTP) for **any** service

**Path:** `libs/lionagi/src/lionagi/services/generic_descriptor.py`  *(top‑level helper used by the adapter)*

```python
# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Mapping

import msgspec


class HttpDescriptor(msgspec.Struct, kw_only=True):
    """Minimal shape to drive a generic HTTP JSON call."""
    method: str = "POST"
    path: str = "/"
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    query: dict[str, str] = msgspec.field(default_factory=dict)
    # Optional key to pull JSON payload from RequestModel (default: entire request)
    payload_field: str | None = None

    def build_url(self, base_url: str) -> str:
        if self.query:
            from urllib.parse import urlencode
            return f"{base_url.rstrip('/')}{self.path}?{urlencode(self.query)}"
        return f"{base_url.rstrip('/')}{self.path}"
```

---

## 2) New adapters

### 2.a `adapters/openai_adapter.py`

**Path:** `libs/lionagi/src/lionagi/services/adapters/openai_adapter.py`

```python
from __future__ import annotations

from typing import Any

from ..endpoint import ChatRequestModel, RequestModel
from ..openai import (
    create_openai_service,
    create_anthropic_service,  # still OpenAI-compatible pathway
)
from ..core import Service
from ..provider_registry import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    request_model: type[RequestModel] = ChatRequestModel
    requires = {"net.out:api.openai.com"}

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        return (provider or "").lower() == "openai"

    def create_service(self, *, base_url: str | None, name: str | None = None, **kwargs: Any) -> Service:
        # We ignore "name" here; Service internally has a `name`.
        api_key = kwargs.get("api_key") or ""
        org = kwargs.get("organization")
        return create_openai_service(
            api_key=api_key,
            organization=org,
            call_mw=kwargs.get("call_mw", ()),
            stream_mw=kwargs.get("stream_mw", ()),
            **{k: v for k, v in kwargs.items() if k not in {"api_key", "organization", "call_mw", "stream_mw"}}
        )


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1/messages"  # via OpenAI-compatible SDK
    request_model: type[RequestModel] = ChatRequestModel
    requires = {"net.out:api.anthropic.com"}

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        return (provider or "").lower() == "anthropic"

    def create_service(self, *, base_url: str | None, name: str | None = None, **kwargs: Any) -> Service:
        api_key = kwargs.get("api_key") or ""
        return create_anthropic_service(
            api_key=api_key,
            call_mw=kwargs.get("call_mw", ()),
            stream_mw=kwargs.get("stream_mw", ()),
            **{k: v for k, v in kwargs.items() if k not in {"api_key", "call_mw", "stream_mw"}}
        )
```

### 2.b `adapters/generic_adapter.py`  (HTTP/JSON catch‑all)

**Path:** `libs/lionagi/src/lionagi/services/adapters/generic_adapter.py`

```python
from __future__ import annotations

from typing import Any, AsyncIterator

from ..core import CallContext, Service
from ..endpoint import RequestModel, ChatRequestModel
from ..provider_registry import ProviderAdapter
from ..transport import HTTPXTransport
from ..generic_descriptor import HttpDescriptor


def _host_to_capability(base_url: str | None) -> set[str]:
    if not base_url:
        return set()
    from urllib.parse import urlparse
    host = urlparse(base_url).netloc
    return {f"net.out:{host}"} if host else set()


class GenericJSONService(Service[RequestModel, dict, bytes]):
    """Thin pass-through that sends RequestModel as JSON to any HTTP endpoint."""
    name: str
    requires: set[str]

    def __init__(self, *, name: str, base_url: str, http: HttpDescriptor):
        self._base_url = base_url
        self._http = http
        self.name = name
        self.requires = _host_to_capability(base_url)

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict:
        payload = self._extract_payload(req)
        url = self._http.build_url(self._base_url)
        timeout = ctx.remaining_time

        async with HTTPXTransport() as t:
            return await t.send_json(
                self._http.method, url, headers=self._http.headers, json=payload, timeout_s=timeout
            )

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[bytes]:
        payload = self._extract_payload(req)
        url = self._http.build_url(self._base_url)
        timeout = ctx.remaining_time

        async with HTTPXTransport() as t:
            async for chunk in t.stream_json(
                self._http.method, url, headers=self._http.headers, json=payload, timeout_s=timeout
            ):
                yield chunk

    def _extract_payload(self, req: RequestModel) -> dict:
        if self._http.payload_field:
            val = getattr(req, self._http.payload_field, None)
            return val if isinstance(val, dict) else {"value": val}
        # default: serialize RequestModel → dict
        try:
            return req.__dict__  # msgspec.Struct has fast attrs
        except Exception:
            # fallback: msgspec to dict
            import msgspec
            return msgspec.to_builtins(req)


class GenericHTTPAdapter(ProviderAdapter):
    """Catch-all HTTP adapter. Works whenever base_url is HTTP(S)."""
    name = "generic_http"
    default_base_url = None
    request_model: type[RequestModel] = ChatRequestModel  # safe default
    requires: set[str] = set()

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        return bool(base_url and base_url.startswith(("http://", "https://")))

    def create_service(self, *, base_url: str | None, name: str | None = None, **kwargs: Any) -> Service:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("generic_http requires a valid HTTP(S) base_url")

        http = kwargs.get("http")
        if http is None:
            http = HttpDescriptor()  # POST "/"
        elif isinstance(http, dict):
            http = HttpDescriptor(**http)

        svc_name = (name or "generic_http").lower()
        return GenericJSONService(name=svc_name, base_url=base_url, http=http)
```

---

## 3) Wire up built‑ins at import time

**Path:** `libs/lionagi/src/lionagi/services/__init__.py` *(create if absent)*

```python
from __future__ import annotations

from .provider_registry import get_default_registry
from .adapters.openai_adapter import OpenAIAdapter, AnthropicAdapter
from .adapters.generic_adapter import GenericHTTPAdapter

# Register built-in adapters once on import
_registry = get_default_registry()
for _adapter in (OpenAIAdapter(), AnthropicAdapter(), GenericHTTPAdapter()):
    try:
        _registry.register(_adapter)
    except ValueError:
        # already registered in this process; ignore
        pass

__all__ = [
    "get_default_registry",
]
```

> **Why import‑time?** Built‑ins should “just work.” For plugins, register explicitly (or via entry‑points) at app bootstrap—keeps startup deterministic and avoids magic.

---

## 4) Replace brittle detection with a resolver that uses the registry

**REPLACE**: `libs/lionagi/src/lionagi/services/provider_detection.py`

```diff
--- a/libs/lionagi/src/lionagi/services/provider_detection.py
+++ b/libs/lionagi/src/lionagi/services/provider_detection.py
@@ -1,232 +1,110 @@
-# (previous regex-heavy detection removed)
-from __future__ import annotations
-
-import re
-from dataclasses import dataclass
-from typing import Any
-
-# Provider patterns for model name detection
-PROVIDER_PATTERNS = {
-    ... many regexes ...
-}
-
-@dataclass
-class ProviderConfig:
-    ...
-
-PROVIDER_CONFIGS = {...}
-
-def detect_provider_from_model(model: str) -> str | None:
-    ...
-
-def infer_provider_config(...): ...
-
-def get_model_info(...): ...
-
-def normalize_model_name(...): ...
-
-def get_capability_requirements(...): ...
-
-def validate_model_provider_compatibility(...): ...
-
-def suggest_alternative_models(...): ...
+from __future__ import annotations
+
+"""Provider resolution helpers (regex-free).
+
+This module delegates to the runtime ProviderRegistry for resolution.
+It only provides convenience functions to parse "provider/model" and
+validate explicit provider+model combinations.
+"""
+
+from typing import Tuple
+
+from .provider_registry import (
+    get_default_registry,
+    resolve_provider_and_model,
+    ProviderRegistry,
+)
+
+
+def parse_provider_model(model: str | None) -> tuple[str | None, str | None]:
+    """Return (provider, model) if model has prefix 'provider/model'."""
+    if model and "/" in model:
+        pfx, rest = model.split("/", 1)
+        return pfx.strip(), rest
+    return None, model
+
+
+def normalize_and_resolve(
+    *,
+    provider: str | None,
+    model: str | None,
+    base_url: str | None,
+    registry: ProviderRegistry | None = None,
+) -> tuple[str, str | None]:
+    """Resolve provider+model using the global registry."""
+    reg = registry or get_default_registry()
+    prov, mdl, _adapter = resolve_provider_and_model(
+        reg, provider=provider, model=model, base_url=base_url
+    )
+    return prov, mdl
```

---

## 5) Integrate the registry in `imodel.py`

**Path:** `libs/lionagi/src/lionagi/services/imodel.py`

```diff
--- a/libs/lionagi/src/lionagi/services/imodel.py
+++ b/libs/lionagi/src/lionagi/services/imodel.py
@@ -1,33 +1,26 @@
 from __future__ import annotations
 ...
-from .openai import (
-    OpenAICompatibleService,
-    create_anthropic_service,
-    create_ollama_service,
-    create_openai_service,
-)
-from .provider_detection import detect_provider_from_model, infer_provider_config
+from .provider_registry import (
+    get_default_registry,
+    resolve_provider_and_model,
+)
 ...
 class iModel:
@@ -79,24 +72,21 @@
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
-        self.endpoint_name = endpoint
+        # Resolve provider/model via registry (regex-free)
+        registry = get_default_registry()
+        resolved_provider, resolved_model, adapter = resolve_provider_and_model(
+            registry,
+            provider=provider,
+            model=model,
+            base_url=base_url,
+        )
+        self.provider = resolved_provider
+        self.model = resolved_model
+        self.base_url = base_url
+        self.endpoint_name = endpoint
 
         # Auto-detect API key if not provided
         if api_key is None:
-            api_key = self._detect_api_key(provider)
+            api_key = self._detect_api_key(self.provider)
 
-        # Build service using provider intelligence
-        self.service = self._create_service(
-            provider=provider, api_key=api_key, base_url=base_url, model=model, **kwargs
-        )
+        # Build service via adapter
+        self.service = adapter.create_service(
+            base_url=base_url, name=self.provider, api_key=api_key, model=self.model, **kwargs
+        )
@@ -169,6 +159,9 @@
     def _build_context(self, **kwargs) -> CallContext:
         """Build call context with deadline awareness."""
         ...
-        return CallContext(
+        ctx = CallContext(
             call_id=uuid4(),
             branch_id=branch_id,
             deadline_s=deadline_s,
             capabilities=all_capabilities,
-            attrs=kwargs,  # Pass remaining kwargs as attrs
+            attrs=kwargs,  # Pass remaining kwargs as attrs
         )
+        # NEW: carry service requires explicitly for PolicyGateMW
+        ctx.attrs["service_requires"] = getattr(self.service, "requires", set())
+        return ctx
@@ -177,51 +170,6 @@
         """Handle provider-specific post-processing like session management."""
         # Claude Code session management (like v0)
         if self.provider == "claude_code" and isinstance(result, dict) and "session_id" in result:
             self.provider_metadata.session_id = result["session_id"]
             logger.debug(f"Updated Claude Code session_id: {result['session_id']}")
 ...
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
```

> **Why this change?** `iModel` no longer needs hard‑coded per‑provider factories. It simply asks the **registry** to resolve and then **lets the adapter** construct the right `Service`. This makes *any* JSON/HTTP service pluggable via the `GenericHTTPAdapter` (or custom adapters).

---

## 6) (Optional) small, defensive improvement in `middleware.py`

**Path:** `libs/lionagi/src/lionagi/services/middleware.py`

```diff
--- a/libs/lionagi/src/lionagi/services/middleware.py
+++ b/libs/lionagi/src/lionagi/services/middleware.py
@@ -92,9 +92,12 @@ class PolicyGateMW:
     def _get_required_capabilities(self, req: RequestModel, ctx: CallContext) -> set[str]:
         """Get required capabilities from service declaration and optional request additions.
         
-        Service requirements are authoritative (from ctx.attrs["service_requires"]).
+        Service requirements are authoritative (from ctx.attrs["service_requires"]).
         Request can add extra requirements but cannot replace service requirements.
         """
-        service_requires = set(ctx.attrs.get("service_requires", set()))
+        service_requires = set()
+        if isinstance(ctx.attrs, dict):
+            service_requires = set(ctx.attrs.get("service_requires", set()))
+
         request_extras = set(getattr(req, "_extra_requires", set()))
         return service_requires | request_extras
```

> *This keeps PolicyGate robust even if `attrs` was not dict‑like in some edge test.*

---

## 7) Minimal changes to `openai.py` (none required)

Your current `openai.py` already exposes `create_openai_service` and `create_anthropic_service` that the adapter uses directly, so **no diffs are needed**.

---

## 8) Tests (examples)

```python
# tests/test_provider_resolution.py
import pytest
from lionagi.services.provider_registry import get_default_registry, resolve_provider_and_model
from lionagi.services.imodel import iModel

def test_explicit_provider_resolution():
    reg = get_default_registry()
    p, m, _ = resolve_provider_and_model(reg, provider="openai", model="gpt-4", base_url=None)
    assert p == "openai"
    assert m == "gpt-4"

def test_prefixed_model_resolution():
    reg = get_default_registry()
    p, m, _ = resolve_provider_and_model(reg, provider=None, model="anthropic/claude-3-5-sonnet", base_url=None)
    assert p == "anthropic"
    assert m == "claude-3-5-sonnet"

def test_custom_provider_registration_and_use():
    class DummyAdapter:
        name = "my_custom"
        default_base_url = "https://api.example.com"
        request_model = None
        requires = {"net.out:api.example.com"}
        def supports(self, **kw): return (kw.get("provider") or "").lower() == "my_custom"
        def create_service(self, **kw):
            from lionagi.services.adapters.generic_adapter import GenericJSONService
            from lionagi.services.generic_descriptor import HttpDescriptor
            return GenericJSONService(name="my_custom", base_url=kw["base_url"], http=HttpDescriptor())

    reg = get_default_registry()
    reg.register(DummyAdapter())
    im = iModel(provider="my_custom", base_url="https://api.example.com/v1")
    assert im.service.name == "my_custom"

def test_error_ambiguous():
    reg = get_default_registry()
    with pytest.raises(ValueError, match="Provider must be specified"):
        resolve_provider_and_model(reg, provider=None, model="gpt-4", base_url=None)

def test_error_conflict():
    reg = get_default_registry()
    with pytest.raises(ValueError, match="Provider mismatch"):
        resolve_provider_and_model(reg, provider="openai", model="anthropic/claude-3", base_url=None)
```

---

## 9) Migration Guide (from `provider_detection.py` & regex approach)

1. **Replace imports**

   * Old: `from lionagi.services.provider_detection import detect_provider_from_model`
   * New:

     ```python
     from lionagi.services.provider_registry import get_default_registry, resolve_provider_and_model
     ```
   * Or convenience helper:

     ```python
     from lionagi.services.provider_detection import normalize_and_resolve
     ```

2. **iModel callers**

   * Continue to pass `provider=...` or `model="provider/model"` or just `base_url="https://..."` for custom JSON services.
   * For **custom HTTP** services, you can pass an optional `http` descriptor (see `HttpDescriptor`) to control method/path/headers. If omitted, default is `POST /`.

3. **Custom provider adapters (non‑HTTP too)**

   * Implement `ProviderAdapter` and `Service` (e.g., a CLI/subprocess service).
   * Register at app start:

     ```python
     from lionagi.services.provider_registry import get_default_registry
     get_default_registry().register(MyAdapter())
     ```

4. **Security**

   * Each adapter must set `Service.requires` (static or base\_url‑derived); this propagates to `ctx.attrs['service_requires']` by the `iModel` change above → enforced by `PolicyGateMW`.

5. **Observability**

   * Your resilience/transport fixes remain intact. Recall why **streaming must never buffer** (the old code buffered chunks inside the circuit breaker, now removed), and why **Transport** must classify errors precisely (rate-limit vs client vs server). See prior behavior in your earlier files for contrast: resilience buffering and transport JSON parsing choices. &#x20;

---

## 10) Developer Experience & Validation Strategy

* **Registration timing**: Built‑ins register at import; plugins register explicitly at boot (you can also wire Python `entry_points` to auto‑discover if desired).

* **Error messages**:

  * Unknown provider: “Unknown provider 'X'. Registered: \[...]”
  * Ambiguous: “Provider must be specified … or an HTTP(S) base\_url must be provided …”
  * Mismatch: “Provider mismatch: provider='X' vs model prefix='Y'.”

* **Validation**:

  * **Registration-time**: Prevent duplicate provider names.
  * **Call-time**: Adapters may raise if required `base_url`/`api_key` is missing.
  * **Generic HTTP** validation: checks base\_url scheme; optional `HttpDescriptor` schema is msgspec‑validated.

* **Non‑HTTP transports**: Provide a custom adapter that returns a `Service` using your own IO (subprocess/gRPC). Because we depend only on the `Service` protocol (`call`/`stream`), the rest of the stack (executor, middleware, policy) remains unchanged and fully works.

---

### Why this meets your 4 goals

1. **Robust explicit configuration**
   No regex in hot paths; explicit registry resolution and `"provider/model"` parsing.

2. **Maximum modularity**
   `GenericHTTPAdapter` + `HttpDescriptor` makes *any* JSON endpoint an iModel. Non‑HTTP is just a custom adapter that returns a `Service`.

3. **Type safety & msgspec perf**
   `ProviderRegistry` is `msgspec.Struct`. Descriptors are `msgspec.Struct`. No reflection or regex loops.

4. **Security integration**
   Each `Service` sets `requires` (e.g., from `base_url`). `iModel` propagates it into `ctx.attrs['service_requires']` → `PolicyGateMW` enforces capability gates before transport.

---

If you want me to also generate **stubs for a subprocess adapter** (e.g., `claude_code_cli`) or **entry‑points discovery**, say the word and I’ll add those files with diffs in the same style.
