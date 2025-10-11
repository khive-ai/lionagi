# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""ID bridge utilities for V0/V1 compatibility.

This module provides utilities to convert between V0's UUID and V1's
canonical UUID representation, enabling seamless interoperability during
the gradual evolution process.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from .generic.element import ID, Element

__all__ = (
    "to_uuid",
    "canonical_id",
)


def to_uuid(value: Any) -> UUID:
    """Convert ID-like values (UUID | str | Element) to UUID (v4)."""
    if isinstance(value, Element):
        return value.id
    if isinstance(value, UUID):
        return value
    if hasattr(value, "_id") and isinstance(value._id, UUID):
        return value._id
    return ID.get_id(value)


def canonical_id(obj: Any) -> UUID:
    """Accept an Observable-like object or raw ID and return canonical UUID."""
    id_like = getattr(obj, "id", obj)
    return to_uuid(id_like)
