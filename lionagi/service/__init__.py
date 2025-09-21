# Eager imports for core functionality
from typing import TYPE_CHECKING

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


_lazy_imports = {}


def __getattr__(name: str):
    """Lazy loading for heavy service imports."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "RateLimitedAPIExecutor":
        from .rate_limited_processor import RateLimitedAPIExecutor

        _lazy_imports["RateLimitedAPIExecutor"] = RateLimitedAPIExecutor
        return RateLimitedAPIExecutor

    if name in ("Endpoint", "EndpointConfig"):
        from .connections.endpoint import Endpoint, EndpointConfig

        _lazy_imports["Endpoint"] = Endpoint
        _lazy_imports["EndpointConfig"] = EndpointConfig
        return Endpoint if name == "Endpoint" else EndpointConfig

    if name == "iModelManager":
        from .manager import iModelManager

        _lazy_imports["iModelManager"] = iModelManager
        return iModelManager

    if name == "iModel":
        from .imodel import iModel

        _lazy_imports["iModel"] = iModel
        return iModel

    if name == "APICalling":
        from .connections.api_calling import APICalling

        _lazy_imports["APICalling"] = APICalling
        return APICalling

    if name == "TokenCalculator":
        from .token_calculator import TokenCalculator

        _lazy_imports["TokenCalculator"] = TokenCalculator
        return TokenCalculator

    if name == "Broadcaster":
        from .broadcaster import Broadcaster

        _lazy_imports["Broadcaster"] = Broadcaster
        return Broadcaster

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
    "APICalling",
    "Endpoint",
    "EndpointConfig",
    "RateLimitedAPIExecutor",
    "TokenCalculator",
    "iModel",
    "iModelManager",
    "HookEventTypes",
    "HookDict",
    "AssosiatedEventInfo",
    "HookEvent",
    "HookRegistry",
    "Broadcaster",
    "HookEventTypes",
    "HookDict",
    "AssosiatedEventInfo",
    "HookEvent",
    "HookRegistry",
    "global_hook_logger",
    "HookedEvent",
)
