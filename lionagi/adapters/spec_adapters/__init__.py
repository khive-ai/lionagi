"""Spec adapters for converting Spec objects to framework-specific field definitions."""

from ._protocol import SpecAdapter
from .pydantic_field import PydanticSpecAdapter

__all__ = (
    "SpecAdapter",
    "PydanticSpecAdapter",
)
