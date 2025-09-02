# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""ID bridge utilities for V0/V1 compatibility.

This module provides utilities to convert between V0's IDType and V1's
canonical UUID representation, enabling seamless interoperability during
the gradual evolution process.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from .generic.element import Element, IDType

__all__ = (
    "to_uuid",
    "canonical_id",
)


def to_uuid(value: Any) -> UUID:
    """Convert ID-like values (IDType | UUID | str | Element) to UUID (v4).

    Optimized version that avoids string conversion when possible by directly
    accessing IDType's internal UUID. Falls back to V0's IDType.validate()
    for validation semantics only when necessary.

    Args:
        value: An ID-like value to convert (IDType, UUID, str, or Element)

    Returns:
        UUID: A validated UUIDv4

    Raises:
        IDError: If the value cannot be converted to a valid UUIDv4

    Examples:
        >>> element = Element()
        >>> uuid_val = to_uuid(element)
        >>> isinstance(uuid_val, UUID)
        True
        >>> to_uuid("550e8400-e29b-41d4-a716-446655440000")
        UUID('550e8400-e29b-41d4-a716-446655440000')
    """
    if isinstance(value, Element):
        return value.id._id
    if isinstance(value, UUID):
        return value
    if hasattr(value, "_id") and isinstance(value._id, UUID):
        return value._id
    # Fallback: Validate then access ._id directly (no string conversion)
    validated_id = IDType.validate(value)
    return validated_id._id


def canonical_id(obj: Any) -> UUID:
    """Accept an Observable-like object or raw ID and return canonical UUID.

    Safe to use across V0/V1 without changing class definitions. Prefers
    attribute access (.id) but falls back to treating the object as a raw ID.

    Args:
        obj: An Observable object with .id attribute, or a raw ID value

    Returns:
        UUID: The canonical UUID representation

    Examples:
        >>> element = Element()
        >>> uuid_val = canonical_id(element)
        >>> isinstance(uuid_val, UUID)
        True
        >>> canonical_id("550e8400-e29b-41d4-a716-446655440000")
        UUID('550e8400-e29b-41d4-a716-446655440000')
    """
    # Prefer attribute access; fall back to treating obj as a raw id
    id_like = getattr(obj, "id", obj)
    return to_uuid(id_like)
