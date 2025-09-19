# Lazy loading for heavy type imports to improve startup performance
_lazy_type_imports = {}


def __getattr__(name: str):
    """Lazy loading for type definitions."""
    if name in _lazy_type_imports:
        return _lazy_type_imports[name]

    # Import from fields
    try:
        from .fields import __all__ as fields_all

        if name in fields_all:
            from . import fields

            attr = getattr(fields, name)
            _lazy_type_imports[name] = attr
            return attr
    except (ImportError, AttributeError):
        pass

    # Import from models
    try:
        from .models import __all__ as models_all

        if name in models_all:
            from . import models

            attr = getattr(models, name)
            _lazy_type_imports[name] = attr
            return attr
    except (ImportError, AttributeError):
        pass

    # Import from protocols.types
    try:
        from .protocols import types as protocol_types

        if hasattr(protocol_types, name):
            attr = getattr(protocol_types, name)
            _lazy_type_imports[name] = attr
            return attr
    except (ImportError, AttributeError):
        pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
