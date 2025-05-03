
"""TOML adapters for any :class:`Adaptable` domain class."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, TypeVar

import toml
from pydantic import BaseModel

from .adapter import Adapter

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Helper to normalise 'many' TOML data                                        #
# --------------------------------------------------------------------------- #
def _ensure_list(data: Any) -> List[dict]:
    """Return *list of dicts* no matter the input shape."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # if dict has exactly one key whose value is list[dict], unwrap
        if len(data) == 1:
            (v,) = data.values()
            if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                return v
    # fallback: wrap single record into list
    return [data]


# --------------------------------------------------------------------------- #
# TOML string adapter                                                         #
# --------------------------------------------------------------------------- #
class TomlAdapter(Adapter[T]):
    """Convert to/from **in-memory** TOML strings."""

    obj_key = "toml"

    # --------------- incoming ------------------------------------------------
    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: str,
        /,
        *,
        many: bool = False,
        **kwargs,
    ):
        parsed = toml.loads(obj, **kwargs)
        if many:
            records = _ensure_list(parsed)
            return [subj_cls.model_validate(r) for r in records]
        return subj_cls.model_validate(parsed)

    # --------------- outgoing ------------------------------------------------
    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        many: bool = False,
        **kwargs,
    ) -> str:
        items = subj if isinstance(subj, list) else [subj]
        if many:
            payload: dict[str, list[dict]] = { "items": [i.model_dump() for i in items] }
        else:
            payload = items[0].model_dump()
        return toml.dumps(payload, **kwargs)


# --------------------------------------------------------------------------- #
# TOML file adapter                                                           #
# --------------------------------------------------------------------------- #
class TomlFileAdapter(Adapter[T]):
    """Read/write **.toml files** on disk."""

    obj_key = ".toml"

    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: str | Path,
        /,
        *,
        many: bool = False,
        **kwargs,
    ):
        path = Path(obj)
        with path.open(encoding="utf-8") as f:
            parsed = toml.load(f, **kwargs)
        if many:
            records = _ensure_list(parsed)
            return [subj_cls.model_validate(r) for r in records]
        return subj_cls.model_validate(parsed)

    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        fp: str | Path,
        many: bool = False,
        mode: str = "w",
        **kwargs,
    ) -> None:
        path = Path(fp)
        items = subj if isinstance(subj, list) else [subj]
        if many:
            payload: dict[str, list[dict]] = { "items": [i.model_dump() for i in items] }
        else:
            payload = items[0].model_dump()
        with path.open(mode, encoding="utf-8") as f:
            toml.dump(payload, f, **kwargs)
        logging.info("Wrote TOML to %s", path)
