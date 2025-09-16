# Eager imports for core functionality
from .connections.api_calling import APICalling
from .connections.endpoint import Endpoint, EndpointConfig
from .hooks import *
from .imodel import iModel
from .manager import iModelManager
from .rate_limited_processor import RateLimitedAPIExecutor

# Lazy loading cache
_lazy_imports = {}


def __getattr__(name: str):
    """Lazy loading for heavy service imports."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "TokenCalculator":
        from .token_calculator import TokenCalculator

        _lazy_imports["TokenCalculator"] = TokenCalculator
        return TokenCalculator

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
)
