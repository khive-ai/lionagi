"""LSpec v2: Minimal field specification system for multi-backend validation.

This package provides a thin Python interface layer for field specifications
that route to pluggable validation backends (Pydantic, Rust, Cloud).

Key components:
- FieldSpec: Lightweight field specification (data structure, not validator)
- BackendRegistry: Plugin system for validation backends
- PydanticBackend: Free tier basic validation
- RustBackend: Paid local formal verification (stub)
- CloudBackend: Enterprise SaaS with guarantees (stub)

Design principles:
- Python = distribution layer (DSL + routing)
- Backends = validation layer (where complexity lives)
- Composition = reusable field specs that compose into models
"""

from .spec import FieldSpec
from .registry import BackendRegistry, Backend
from .backends import PydanticBackend, RustBackend, CloudBackend

__all__ = [
    "FieldSpec",
    "BackendRegistry",
    "Backend",
    "PydanticBackend",
    "RustBackend",
    "CloudBackend",
]

__version__ = "2.0.0-alpha"
