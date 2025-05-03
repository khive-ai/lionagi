"""Utility helpers consolidated for the traits package."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from lionagi.traits._integrity import Algo, digest
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Serialization helpers                                                       #
# --------------------------------------------------------------------------- #
def serialize_model_to_dict(obj) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    raise TypeError(
        "Input value should be a `pydantic.BaseModel` object or a `dict`"
    )


# --------------------------------------------------------------------------- #
# Simple, canonical UUID / datetime helpers                                   #
# --------------------------------------------------------------------------- #
def serialize_id(value: UUID) -> str:
    return str(value)


def validate_id(value: str | UUID) -> UUID:
    from uuid import UUID as _UUID

    if isinstance(value, _UUID):
        return value
    try:
        return _UUID(str(value))
    except Exception as exc:
        raise ValueError("Invalid UUID") from exc


def serialize_created_at(value: datetime) -> str:
    return value.isoformat()


def validate_created_at(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    from datetime import datetime as _dt

    try:
        return _dt.fromisoformat(value)
    except Exception as exc:
        raise ValueError("Invalid datetime isoformat") from exc


# --------------------------------------------------------------------------- #
# Fast-det hash helper exposed for non-Identifiable payloads                  #
# --------------------------------------------------------------------------- #
def quick_hash(obj: dict, *, algo: Algo = Algo.SHA256) -> str:
    """One-liner wrapper over :pyfunc:`integrity.digest`."""
    return digest(obj, algo=algo)
