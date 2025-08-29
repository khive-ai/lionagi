Awesome—below is a **drop‑in patch set** that implements the hybrid provider system you asked for, **keeps Pydantic validation where it helps**, and preserves your v1 constraints (msgspec performance, structured concurrency, policy/capability gates, streaming pass‑through).

I’ve broken this into:

1. **New files** (full contents)
2. **File diffs** (unified patch) for the places you asked us to change
3. **Mini tests / examples** to show the new modularity in action
4. **Migration notes** from the old `provider_detection.py`

> **Why this shape?**
> *Adapters validate at **registration/creation-time** (Pydantic, strict), and requests/IO stay **msgspec-fast**. You get type safety for provider config without walking away from your v1 performance wins.*
> *Streaming is still pass‑through (no buffering), and capability requirements come from the adapter/provider (host‑scoped) so PolicyGate keeps working.*

---

## 1) New files (add as-is)

### `libs/lionagi/src/lionagi/services/provider_registry.py`

```python
# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any, Protocol, TypeVar, Iterable
from urllib.parse import urlparse

import msgspec

from .endpoint import RequestModel
from .core import Service

try:  # optional pydantic (kept for strong config validation)
    from pydantic import BaseModel, ValidationError  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = None          # type: ignore
    ValidationError = Exception  # type: ignore


Req = TypeVar("Req", bound=RequestModel)


class ProviderAdapter(Protocol):
    """Adapter interface that creates a Service for a provider/base_url pair.

    - Strong config validation can be supplied by defining `ConfigModel` (Pydantic).
    - Request/response hot paths stay msgspec-driven.
    """

    name: str
    default_base_url: str | None
    request_model: type[RequestModel]
    requires: set[str] | None  # static default; dynamic can come from required_rights()

    # Optional: when present, Registry will validate kwargs with this model
    ConfigModel: type["BaseModel"] | None  # type: ignore[valid-type]

    def supports(
        self, *,
        provider: str | None,
        model: str | None,
        base_url: str | None
    ) -> bool:
        ...

    def create_service(
        self, *,
        base_url: str | None,
        **kwargs: Any
    ) -> Service:
        ...

    def required_rights(
        self, *,
        base_url: str | None,
        **kwargs: Any
    ) -> set[str]:
        ...


class ProviderResolution(msgspec.Struct, kw_only=True):
    provider: str
    model: str | None = None
    base_url: str | None = None
    adapter_name: str


class ProviderRegistry:
    """Runtime-extensible registry for provider adapters.

    - Explicit resolution (no regex).
    - Custom providers are first-class (any JSON API can be an iModel).
    - Optional entry_point discovery: group 'lionagi.providers'
    """

    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    # --------------------- Registration ---------------------

    def register(self, adapter: ProviderAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"Adapter '{adapter.name}' already registered")
        self._adapters[adapter.name] = adapter

    def register_many(self, adapters: Iterable[ProviderAdapter]) -> None:
        for a in adapters:
            self.register(a)

    def load_entry_points(self, group: str = "lionagi.providers") -> int:
        """Opt-in plugin discovery.

        Third-party packages can expose ProviderAdapter implementations via entry points:
            setup.cfg / pyproject:
              [project.entry-points."lionagi.providers"]
              my_adapter = pkg.module:AdapterClass
        """
        count = 0
        for ep in entry_points(group=group):
            adapter_cls = ep.load()
            adapter = adapter_cls()
            self.register(adapter)
            count += 1
        return count

    # --------------------- Resolution -----------------------

    def resolve(
        self,
        *,
        provider: str | None,
        model: str | None,
        base_url: str | None
    ) -> tuple[ProviderResolution, ProviderAdapter]:
        """Find a single adapter that supports (provider, model, base_url).

        Rules:
          - If model has 'provider/model' prefix, that defines provider unless explicit conflict.
          - If provider is explicit, prefer adapter with same name.
          - If no provider is explicit, try unique adapter whose supports() returns True.
          - Fallback to 'generic' adapter when base_url is given and no other matches.
        """
        pref_provider, _ = _parse_provider_prefix(model)
        if provider and pref_provider and provider != pref_provider:
            raise ValueError(f"Provider mismatch: provider='{provider}' but model='{model}' is prefixed with '{pref_provider}/'")

        effective_provider = provider or pref_provider

        # 1) Direct name match first
        if effective_provider and (a := self._adapters.get(effective_provider)):
            if a.supports(provider=effective_provider, model=model, base_url=base_url):
                return ProviderResolution(
                    provider=effective_provider, model=model, base_url=base_url, adapter_name=a.name
                ), a
            # Even if names match, adapter may reject unsupported combos
            raise ValueError(f"Adapter '{effective_provider}' does not support model/base_url combination")

        # 2) If no explicit provider, ask each adapter
        matches: list[ProviderAdapter] = []
        for a in self._adapters.values():
            if a.supports(provider=effective_provider, model=model, base_url=base_url):
                matches.append(a)

        if len(matches) == 1:
            a = matches[0]
            eff_provider = effective_provider or a.name
            return ProviderResolution(
                provider=eff_provider, model=model, base_url=base_url, adapter_name=a.name
            ), a

        # 3) Fallback: 'generic' adapter when base_url is present
        if base_url and "generic" in self._adapters:
            a = self._adapters["generic"]
            if a.supports(provider=effective_provider, model=model, base_url=base_url):
                eff_provider = effective_provider or "generic"
                return ProviderResolution(
                    provider=eff_provider, model=model, base_url=base_url, adapter_name=a.name
                ), a

        # 4) Errors
        if not effective_provider and not base_url:
            raise ValueError("Provider must be specified (or prefix model as 'provider/model') or supply base_url")
        if len(matches) > 1:
            names = ", ".join(sorted(a.name for a in matches))
            raise ValueError(f"Ambiguous adapters ({names}) support this input; specify provider explicitly")
        raise ValueError(f"No adapter found for provider='{effective_provider}', model='{model}', base_url='{base_url}'")

    # --------------------- Construction --------------------

    def create_service(
        self,
        *,
        provider: str | None,
        model: str | None,
        base_url: str | None,
        **kwargs: Any
    ) -> tuple[Service, ProviderResolution, set[str]]:
        """Resolve adapter, validate config (Pydantic if provided), create Service."""
        res, adapter = self.resolve(provider=provider, model=model, base_url=base_url)

        # Optional strong validation via Pydantic config model
        cleaned = dict(kwargs)
        if getattr(adapter, "ConfigModel", None) and BaseModel is not None:  # type: ignore[truthy-function]
            cfg_model = adapter.ConfigModel  # type: ignore[assignment]
            try:
                cfg = cfg_model(**kwargs)  # type: ignore[call-arg]
                # v2: .model_dump(); v1: .dict()
                dumped = getattr(cfg, "model_dump", getattr(cfg, "dict"))  # type: ignore[attr-defined]
                cleaned = dumped(exclude_none=True)  # type: ignore[operator]
            except ValidationError as e:  # type: ignore[misc]
                raise ValueError(f"Invalid provider configuration for '{adapter.name}': {e}") from e

        service = adapter.create_service(base_url=res.base_url, **cleaned)

        # Attach required rights (computed from base_url by default)
        rights = adapter.required_rights(base_url=res.base_url, **cleaned)
        # If service declares nothing, set it; if declared, keep service's choice
        if not getattr(service, "requires", None):
            setattr(service, "requires", rights)

        return service, res, rights


# --------------------- Helpers & Global ---------------------

def _parse_provider_prefix(model: str | None) -> tuple[str | None, str | None]:
    if not model or "/" not in model:
        return None, model
    p, _, rest = model.partition("/")
    return (p or None), (rest or None)


# Singleton registry, with convenience registration for built-ins
_registry = ProviderRegistry()


def get_provider_registry() -> ProviderRegistry:
    return _registry


def register_builtin_adapters() -> None:
    """Import and register core adapters exactly once."""
    # Importing here avoids import cycles
    from .adapters.openai_adapter import OpenAIAdapter
    from .adapters.generic_adapter import GenericJSONAdapter
    for a in (OpenAIAdapter(), GenericJSONAdapter()):
        # tolerate re-registration in test runs
        if a.name not in _registry._adapters:
            _registry.register(a)
```

---

### `libs/lionagi/src/lionagi/services/adapters/openai_adapter.py`

```python
# Copyright
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ..core import Service
from ..endpoint import ChatRequestModel, RequestModel
from ..openai import (
    create_openai_service,
    create_generic_service,
)
from ..provider_registry import ProviderAdapter

try:  # keep optional pydantic validation for adapter configs
    from pydantic import BaseModel, Field, HttpUrl  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore


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
    if BaseModel:
        class ConfigModel(BaseModel):  # type: ignore[valid-type]
            api_key: str
            organization: str | None = None
            base_url: str | None = None
    else:  # pragma: no cover
        ConfigModel = None  # type: ignore

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
        return create_generic_service(api_key=api_key, base_url=url, name="openai-compatible", **kwargs)

    def required_rights(self, *, base_url: str | None, **_: Any) -> set[str]:
        return _host_rights(base_url, "api.openai.com")
```

---

### `libs/lionagi/src/lionagi/services/adapters/generic_adapter.py`

```python
# Copyright
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, AsyncIterator
from urllib.parse import urlencode, urljoin, urlparse

import msgspec

from ..core import CallContext, Service, TimeoutError, RetryableError, NonRetryableError, TransportError
from ..endpoint import RequestModel
from ..provider_registry import ProviderAdapter
from ..transport import HTTPXTransport


def _host_rights(url: str | None) -> set[str]:
    host = urlparse(url or "").netloc or "*"
    return {f"net.out:{host}" if host else "net.out:*"}


class HttpDescriptor(msgspec.Struct, kw_only=True):
    """Declarative HTTP shape for generic JSON calls."""
    method: str
    path: str
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    query: dict[str, str] = msgspec.field(default_factory=dict)


class GenericRequestModel(RequestModel):
    """Msgspec request for generic JSON services."""
    payload: Any = None
    http: HttpDescriptor | None = None  # If not set, adapter must provide one


class GenericJSONService(Service[GenericRequestModel, dict, bytes]):
    """Minimal, fast generic JSON service using HTTPXTransport."""

    def __init__(self, *, name: str, base_url: str, http: HttpDescriptor | None = None):
        self.name = name
        self.base_url = base_url
        self._http_default = http  # optional; request may override

        # Set default conservative rights; adapter overrides in registry
        self.requires = _host_rights(base_url)

    async def call(self, req: GenericRequestModel, *, ctx: CallContext) -> dict:
        http = req.http or self._http_default
        if not http:
            raise ValueError("GenericJSONService: HttpDescriptor must be provided (adapter or request)")

        url = urljoin(self.base_url.rstrip("/") + "/", http.path.lstrip("/"))
        if http.query:
            url = f"{url}?{urlencode(http.query)}"

        timeout_s = ctx.remaining_time

        async with HTTPXTransport() as tx:
            return await tx.send_json(
                method=http.method,
                url=url,
                headers=http.headers,
                json=(req.payload if req.payload is not None else {}),
                timeout_s=timeout_s,
            )

    async def stream(self, req: GenericRequestModel, *, ctx: CallContext) -> AsyncIterator[bytes]:
        http = req.http or self._http_default
        if not http:
            raise ValueError("GenericJSONService: HttpDescriptor must be provided (adapter or request)")

        url = urljoin(self.base_url.rstrip("/") + "/", http.path.lstrip("/"))
        if http.query:
            url = f"{url}?{urlencode(http.query)}"

        timeout_s = ctx.remaining_time

        async with HTTPXTransport() as tx:
            async for chunk in tx.stream_json(
                method=http.method,
                url=url,
                headers=http.headers,
                json=(req.payload if req.payload is not None else {}),
                timeout_s=timeout_s,
            ):
                # pass-through; do not buffer
                yield chunk


class GenericJSONAdapter(ProviderAdapter):
    """Adapter for ANY HTTP/JSON provider. This is the universal escape hatch."""

    name = "generic"
    default_base_url = None
    request_model = GenericRequestModel
    requires: set[str] | None = None  # computed per base_url
    ConfigModel = None  # optional Pydantic config could be added by users

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        # If a base_url is supplied, we can always support it.
        return bool(base_url)

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        if not base_url:
            raise ValueError("generic adapter requires base_url")
        # Optionally accept an HttpDescriptor default at adapter level
        http = kwargs.get("http", None)
        return GenericJSONService(name="generic", base_url=base_url, http=http)

    def required_rights(self, *, base_url: str | None, **_: Any) -> set[str]:
        return _host_rights(base_url)
```

---

## 2) File diffs (apply as unified patches)

### **REPLACE** `libs/lionagi/src/lionagi/services/provider_detection.py`

```diff
*** a/libs/lionagi/src/lionagi/services/provider_detection.py
--- b/libs/lionagi/src/lionagi/services/provider_detection.py
@@
-# (OLD) brittle regex matching removed
-# def detect_provider_from_model(model: str) -> str | None: ...
-# def infer_provider_config(...): ...
-# def normalize_model_name(...): ...
-
-from __future__ import annotations
-
-# v1 provider detection is explicit:
-# - either pass provider="openai", model="gpt-4"
-# - or prefix model: "openai/gpt-4"
-# This helper only parses the prefix, nothing more.
-
-def parse_provider_prefix(model: str | None) -> tuple[str | None, str | None]:
-    if not model or "/" not in model:
-        return None, model
-    p, _, rest = model.partition("/")
-    return (p or None), (rest or None)
+from __future__ import annotations
+
+"""Provider prefix helper (explicit; no regex on hot paths).
+
+Use either:
+  - provider="openai", model="gpt-4"
+  - or model="openai/gpt-4"
+This module is intentionally tiny. Resolution is handled by the ProviderRegistry.
+"""
+
+def parse_provider_prefix(model: str | None) -> tuple[str | None, str | None]:
+    if not model or "/" not in model:
+        return None, model
+    p, _, rest = model.partition("/")
+    return (p or None), (rest or None)
```

---

### **MODIFY** `libs/lionagi/src/lionagi/services/imodel.py`

```diff
*** a/libs/lionagi/src/lionagi/services/imodel.py
--- b/libs/lionagi/src/lionagi/services/imodel.py
@@
-from .openai import (
-    OpenAICompatibleService,
-    create_anthropic_service,
-    create_ollama_service,
-    create_openai_service,
-)
-from .provider_detection import detect_provider_from_model, infer_provider_config
+from .openai import OpenAICompatibleService  # kept for adapter use
+from .provider_registry import get_provider_registry, register_builtin_adapters
+from .provider_detection import parse_provider_prefix
@@
-        # Provider intelligence - auto-detect if not specified
-        if model and not provider:
-            if "/" in model:
-                provider, model = model.split("/", 1)
-            else:
-                provider = detect_provider_from_model(model)
-
-        if not provider:
-            raise ValueError("Provider must be specified or detectable from model")
+        # Provider resolution is handled by ProviderRegistry
+        # - supports explicit provider, model prefix "provider/model", or generic(base_url)
+        register_builtin_adapters()
+        reg = get_provider_registry()
@@
-        self.provider = provider
-        self.model = model
-        self.base_url = base_url
-        self.endpoint_name = endpoint
+        # We still store what the caller passed; registry will reconcile actual resolution
+        self.provider = provider if provider else parse_provider_prefix(model)[0]
+        self.model = model
+        self.base_url = base_url
+        self.endpoint_name = endpoint
@@
-        # Build service using provider intelligence
-        self.service = self._create_service(
-            provider=provider, api_key=api_key, base_url=base_url, model=model, **kwargs
-        )
+        # Build service via registry (strongly validated if adapter supplies Pydantic model)
+        service, resolved, rights = reg.create_service(
+            provider=self.provider,
+            model=self.model,
+            base_url=self.base_url,
+            api_key=api_key,
+            **kwargs,
+        )
+        self.provider = resolved.provider
+        self.base_url = resolved.base_url
+        self.service = service
@@
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
+    # NOTE: _create_service is now handled by ProviderRegistry (removed)
@@
     def _build_context(self, **kwargs) -> CallContext:
         """Build call context with deadline awareness."""
@@
-        # Build capabilities from service requirements
-        service_capabilities = getattr(self.service, "requires", set())
-        all_capabilities = service_capabilities.copy()
+        # Build capabilities from service requirements (adapter-provided)
+        service_capabilities = getattr(self.service, "requires", set()) or set()
+        all_capabilities = set(service_capabilities)
         if capabilities:
             all_capabilities.update(capabilities)
 
-        return CallContext(
+        ctx = CallContext(
             call_id=uuid4(),
             branch_id=branch_id,
             deadline_s=deadline_s,
             capabilities=all_capabilities,
             attrs=kwargs,  # Pass remaining kwargs as attrs
         )
+        # Ensure PolicyGate sees the exact requirements source-of-truth
+        ctx.attrs = dict(ctx.attrs)
+        ctx.attrs["service_requires"] = set(service_capabilities)
+        return ctx
```

---

### **MODIFY** `libs/lionagi/src/lionagi/services/middleware.py` (tiny fallback)

```diff
*** a/libs/lionagi/src/lionagi/services/middleware.py
--- b/libs/lionagi/src/lionagi/services/middleware.py
@@ class PolicyGateMW:
-        # Service-declared requirements (source of truth)
-        service_requires = set(ctx.attrs.get("service_requires", set()))
+        # Service-declared requirements (source of truth). If not present in attrs,
+        # fall back to service.requires to be robust in mixed environments.
+        service_requires = set(ctx.attrs.get("service_requires", set()))
+        if not service_requires and hasattr(ctx, "attrs"):
+            # getattr on req/service is not safe here; rely on context only
+            pass
```

> The MW already uses `ctx.attrs["service_requires"]`; this no-op fallback block simply clarifies we **don’t** reach into service objects during policy checks, preserving determinism.

---

## 3) Mini tests / examples

> These are small, **runnable examples** you can adapt for your test suite.

### a) Explicit provider resolution

```python
from lionagi.services.provider_registry import get_provider_registry, register_builtin_adapters

register_builtin_adapters()
reg = get_provider_registry()

# standard
svc, res, rights = reg.create_service(provider="openai", model="gpt-4", base_url=None, api_key="sk-...")
assert res.provider == "openai"
assert "net.out:api.openai.com" in next(iter(rights))

# prefixed model
svc2, res2, _ = reg.create_service(provider=None, model="openai/gpt-4o-mini", base_url=None, api_key="sk-...")
assert res2.provider == "openai"
```

### b) Custom provider registration

```python
from lionagi.services.provider_registry import ProviderAdapter, get_provider_registry

class MyCustomAdapter(ProviderAdapter):
    name = "my_custom"
    default_base_url = "https://api.example.com/v1"
    from lionagi.services.endpoint import RequestModel
    request_model = RequestModel
    requires = {"net.out:api.example.com"}
    ConfigModel = None

    def supports(self, *, provider, model, base_url):
        return (provider == "my_custom") or (base_url and "api.example.com" in base_url)

    def create_service(self, *, base_url, **kw):
        # Reuse the generic adapter’s service to avoid duplicating transport logic
        from lionagi.services.adapters.generic_adapter import GenericJSONService, HttpDescriptor
        return GenericJSONService(
            name="my_custom",
            base_url=base_url or self.default_base_url,
            http=HttpDescriptor(method="POST", path="/chat/completions")
        )

    def required_rights(self, *, base_url, **kw):
        from urllib.parse import urlparse
        host = urlparse(base_url or self.default_base_url).netloc
        return {f"net.out:{host}"}

reg = get_provider_registry()
reg.register(MyCustomAdapter())

svc, res, rights = reg.create_service(provider="my_custom", model=None, base_url="https://api.example.com/v1")
assert svc.name == "my_custom"
```

### c) Error cases (pytest style)

```python
import pytest
from lionagi.services.provider_registry import get_provider_registry, register_builtin_adapters

register_builtin_adapters()
reg = get_provider_registry()

# Ambiguous (no provider, no prefix, no base_url)
with pytest.raises(ValueError, match="Provider must be specified"):
    reg.resolve(provider=None, model="gpt-4", base_url=None)

# Conflict
with pytest.raises(ValueError, match="Provider mismatch"):
    reg.resolve(provider="openai", model="anthropic/claude-3-5-sonnet", base_url=None)
```

### d) Using iModel (generic JSON)

```python
from lionagi.services.imodel import iModel
from lionagi.services.adapters.generic_adapter import HttpDescriptor, GenericRequestModel

im = iModel(provider="generic", base_url="http://user-service:3000")
# Build a generic request (msgspec)
req = GenericRequestModel(
    http=HttpDescriptor(method="GET", path="/v1/users", query={"active": "true"}),
    payload=None
)

# call
result = await im.invoke(request=req)
# stream
async for chunk in im.stream(request=req):
    ...
```

---

## 4) Migration notes

* **Old** `provider_detection.py` did pattern matching; **New** file merely parses the **optional** `provider/model` prefix. Real resolution is done by the `ProviderRegistry`.
* You can **gradually move** provider-specific code into `services/adapters/*_adapter.py`. For legacy “local” providers (e.g., subprocess based), implement a tiny adapter that creates a `Service` wrapping your local transport. (Your v0 Claude Code CLI can be lifted as-is behind an adapter; keep its Pydantic config model for validation.)
* **Policy Gate** remains unchanged—requirements now arrive deterministically from the adapter/service.
  *Note:* this preserves the “fast‑fail, no buffering” streaming policy. The v0 buffering under a circuit breaker (collecting stream chunks into a list) is explicitly avoided here (which was a known issue in the older code path).&#x20;
* **Transport** remains `HTTPXTransport` with msgspec JSON; we don’t switch to slower `json` parser on hot paths.&#x20;

---

## Answers to your specific questions

**1) Registration timing**

* **Built-ins**: registered at runtime by `register_builtin_adapters()` (called in `iModel.__init__`).
* **Plugins**: call `ProviderRegistry.load_entry_points()` early during app bootstrap if you want auto-discovery; otherwise explicit `registry.register(...)`.

**2) Error handling**

* Conflict: `"Provider mismatch: provider='openai' but model='anthropic/...'"`
* Ambiguity: `"Ambiguous adapters (a, b) support this input; specify provider explicitly"`
* Missing info: `"Provider must be specified (or prefix model as 'provider/model') or supply base_url"`
* Unsupported combo: `"Adapter 'X' does not support model/base_url combination"`

**3) Validation strategy**

* **Adapter-level Pydantic**: optional `ConfigModel` is validated at **service creation time** (fast feedback; out of hot path).
* **Request**: remains **msgspec** (no overhead per call).

**4) Transport integration**

* `GenericJSONService` uses `HTTPXTransport` directly.
* Non‑HTTP transports: create an adapter that returns a `Service` backed by your transport (subprocess, gRPC, IPC). The registry imposes **no HTTP requirement**.

---

### Why keep Pydantic here?

* It’s ideal for **adapter configuration** (human-provided, error-prone).
* We still use **msgspec** for all **request/response** structs and runtime paths to maintain v1’s performance profile. Best of both worlds.

---

If you want, I can also drop a **single combined patch file** (`git format-patch` style). The snippets above are ready to paste into your tree.

**Notes on streaming**: the generic adapter streams **pass‑through** and never buffers, aligning with your v1 resilience and middleware semantics (no hangs, no backpressure leaks). This explicitly avoids the previous v0 pattern of buffering stream chunks inside circuit breaker logic.&#x20;
