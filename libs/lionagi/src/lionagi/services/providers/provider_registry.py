# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Hybrid provider registry: msgspec performance + Pydantic validation + LLM tool integration."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import entry_points
from typing import Any, Protocol, TypeVar

import msgspec
from pydantic import BaseModel, ValidationError

from ..core import Service
from ..endpoint import RequestModel

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
    ConfigModel: type[BaseModel] | None

    def supports(
        self, *, provider: str | None, model: str | None, base_url: str | None
    ) -> bool: ...

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service: ...

    def required_rights(self, *, base_url: str | None, **kwargs: Any) -> set[str]: ...


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
        self, *, provider: str | None, model: str | None, base_url: str | None
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
            raise ValueError(
                f"Provider mismatch: provider='{provider}' but model='{model}' is prefixed with '{pref_provider}/'"
            )

        effective_provider = provider or pref_provider

        # 1) Direct name match first
        if effective_provider and (a := self._adapters.get(effective_provider)):
            if a.supports(provider=effective_provider, model=model, base_url=base_url):
                return (
                    ProviderResolution(
                        provider=effective_provider,
                        model=model,
                        base_url=base_url,
                        adapter_name=a.name,
                    ),
                    a,
                )
            # Even if names match, adapter may reject unsupported combos
            raise ValueError(
                f"Adapter '{effective_provider}' does not support model/base_url combination"
            )

        # 2) If no explicit provider, ask each adapter
        matches: list[ProviderAdapter] = []
        for a in self._adapters.values():
            if a.supports(provider=effective_provider, model=model, base_url=base_url):
                matches.append(a)

        if len(matches) == 1:
            a = matches[0]
            eff_provider = effective_provider or a.name
            return (
                ProviderResolution(
                    provider=eff_provider,
                    model=model,
                    base_url=base_url,
                    adapter_name=a.name,
                ),
                a,
            )

        # 3) Fallback: 'generic' adapter when base_url is present
        if base_url and "generic" in self._adapters:
            a = self._adapters["generic"]
            if a.supports(provider=effective_provider, model=model, base_url=base_url):
                eff_provider = effective_provider or "generic"
                return (
                    ProviderResolution(
                        provider=eff_provider,
                        model=model,
                        base_url=base_url,
                        adapter_name=a.name,
                    ),
                    a,
                )

        # 4) Errors - check ambiguous first, then missing provider
        if len(matches) > 1:
            names = ", ".join(sorted(a.name for a in matches))
            raise ValueError(
                f"Ambiguous adapters ({names}) support this input; specify provider explicitly"
            )
        if not effective_provider and not base_url:
            raise ValueError(
                "Provider must be specified (or prefix model as 'provider/model') or supply base_url"
            )
        raise ValueError(
            f"No adapter found for provider='{effective_provider}', model='{model}', base_url='{base_url}'"
        )

    # --------------------- Construction --------------------

    def create_service(
        self,
        *,
        provider: str | None,
        model: str | None,
        base_url: str | None,
        **kwargs: Any,
    ) -> tuple[Service, ProviderResolution, set[str]]:
        """Resolve adapter, validate config (Pydantic if provided), create Service."""
        res, adapter = self.resolve(provider=provider, model=model, base_url=base_url)

        # Optional strong validation via Pydantic config model
        cleaned = dict(kwargs)
        if getattr(adapter, "ConfigModel", None):
            cfg_model = adapter.ConfigModel
            try:
                cfg = cfg_model(**kwargs)
                # v2: .model_dump(); v1: .dict()
                dumped = getattr(cfg, "model_dump", getattr(cfg, "dict"))
                cleaned = dumped(exclude_none=True)
            except ValidationError as e:
                raise ValueError(f"Invalid provider configuration for '{adapter.name}': {e}") from e

        service = adapter.create_service(base_url=res.base_url, **cleaned)

        # Attach required rights (computed from base_url by default)
        rights = adapter.required_rights(base_url=res.base_url, **cleaned)
        # If service declares nothing, set it; if declared, keep service's choice
        if not getattr(service, "requires", None):
            setattr(service, "requires", rights)

        return service, res, rights

    def known_adapters(self) -> set[str]:
        """Return set of registered adapter names."""
        return set(self._adapters.keys())


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
    from ..adapters.generic_adapter import GenericJSONAdapter
    from ..adapters.openai_adapter import OpenAIAdapter

    for a in (OpenAIAdapter(), GenericJSONAdapter()):
        # tolerate re-registration in test runs
        if a.name not in _registry._adapters:
            _registry.register(a)
