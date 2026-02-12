# Eager imports for core functionality
from typing import TYPE_CHECKING

from lionagi.ln._lazy_init import lazy_import

from .hooks import (
    AssosiatedEventInfo,
    HookDict,
    HookedEvent,
    HookEvent,
    HookEventTypes,
    HookRegistry,
    global_hook_logger,
)

if TYPE_CHECKING:
    from .broadcaster import Broadcaster
    from .connections.api_calling import APICalling
    from .connections.endpoint import Endpoint, EndpointConfig
    from .imodel import iModel
    from .manager import iModelManager
    from .rate_limited_processor import RateLimitedAPIExecutor
    from .token_calculator import TokenCalculator

_LAZY_MAP: dict[str, tuple[str, str | None]] = {
    "RateLimitedAPIExecutor": (
        "rate_limited_processor",
        "RateLimitedAPIExecutor",
    ),
    "Endpoint": ("connections.endpoint", "Endpoint"),
    "EndpointConfig": ("connections.endpoint", "EndpointConfig"),
    "iModelManager": ("manager", "iModelManager"),
    "iModel": ("imodel", "iModel"),
    "APICalling": ("connections.api_calling", "APICalling"),
    "TokenCalculator": ("token_calculator", "TokenCalculator"),
    "Broadcaster": ("broadcaster", "Broadcaster"),
}


def __getattr__(name: str):
    return lazy_import(name, _LAZY_MAP, __name__, globals())


__all__ = (
    "APICalling",
    "AssosiatedEventInfo",
    "Broadcaster",
    "Endpoint",
    "EndpointConfig",
    "HookDict",
    "HookEvent",
    "HookEventTypes",
    "HookRegistry",
    "HookedEvent",
    "RateLimitedAPIExecutor",
    "TokenCalculator",
    "global_hook_logger",
    "iModel",
    "iModelManager",
)
