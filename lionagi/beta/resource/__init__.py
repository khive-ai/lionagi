# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Resources module: iModel, ResourceBackend, hooks, and registry.

Core exports:
- iModel: Unified resource interface with rate limiting and hooks
- ResourceBackend/Endpoint: Backend abstractions for API calls
- HookRegistry/HookEvent/HookPhase: Lifecycle hook system
- ResourceRegistry: O(1) name-based resource lookup

Uses lazy loading for fast import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Calling": ("lionagi.beta.resource.backend", "Calling"),
    "Normalized": ("lionagi.beta.resource.backend", "Normalized"),
    "ResourceBackend": ("lionagi.beta.resource.backend", "ResourceBackend"),
    "ResourceConfig": ("lionagi.beta.resource.backend", "ResourceConfig"),
    "ResourceRegistry": ("lionagi.beta.resource.registry", "ResourceRegistry"),
    "iModel": ("lionagi.beta.resource.imodel", "iModel"),
    "HookRegistry": ("lionagi.service.hooks", "HookRegistry"),
    "HookEvent": ("lionagi.service.hooks", "HookEvent"),
    "HookEventTypes": ("lionagi.service.hooks", "HookEventTypes"),
    "Service": ("lionagi.beta.resource.service", "Service"),
    "ServiceCalling": ("lionagi.beta.resource.service", "ServiceCalling"),
    "ResourceMeta": ("lionagi.beta.resource.service", "ResourceMeta"),
    "add_service": ("lionagi.beta.resource.service", "add_service"),
    "clear_services": ("lionagi.beta.resource.service", "clear_services"),
    "get_service": ("lionagi.beta.resource.service", "get_service"),
    "has_service": ("lionagi.beta.resource.service", "has_service"),
    "list_services": ("lionagi.beta.resource.service", "list_services"),
    "list_services_sync": ("lionagi.beta.resource.service", "list_services_sync"),
    "remove_service": ("lionagi.beta.resource.service", "remove_service"),
    "resource": ("lionagi.beta.resource.service", "resource"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    if name in _LOADED:
        return _LOADED[name]

    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    raise AttributeError(f"module 'lionagi.beta.resource' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from .backend import (
        Calling,
        Normalized,
        ResourceBackend,
        ResourceConfig,
    )
    from lionagi.service.hooks import HookEvent, HookEventTypes, HookRegistry
    from .imodel import iModel
    from .registry import ResourceRegistry
    from .service import (
        Service,
        ServiceCalling,
        ResourceMeta,
        add_service,
        clear_services,
        get_service,
        has_service,
        list_services,
        list_services_sync,
        remove_service,
        resource,
    )

__all__ = (
    "Calling",
    "HookEvent",
    "HookEventTypes",
    "HookRegistry",
    "Normalized",
    "ResourceBackend",
    "ResourceConfig",
    "ResourceMeta",
    "ResourceRegistry",
    "Service",
    "ServiceCalling",
    "add_service",
    "clear_services",
    "get_service",
    "has_service",
    "iModel",
    "list_services",
    "list_services_sync",
    "remove_service",
    "resource",
)
