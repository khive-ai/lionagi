"""Spec adapters for framework-agnostic field and model creation.

This package provides adapters that bridge the Spec/Operable system
to various validation frameworks (Pydantic, msgspec, Rust, etc.).
"""

from ._protocol import SpecAdapter
from .pydantic_field import PydanticSpecAdapter

__all__ = (
    "SpecAdapter",
    "PydanticSpecAdapter",
)
