from __future__ import annotations

from typing import Dict

_REG: Dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator to register a Morphism by its .name."""
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError(
            "Cannot register morphism without string 'name' attribute"
        )
    _REG[name] = cls
    return cls


def get(name: str) -> type | None:
    return _REG.get(name)


def all_names() -> list[str]:
    return sorted(_REG.keys())
