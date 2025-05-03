"""JSON adapters (string & file) usable by any :class:`Adaptable` domain class."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeVar

from .adaptable import Adaptable
from .adapter import Adapter

T = TypeVar("T", bound=Adaptable)


class JsonAdapter(Adapter[T]):
    obj_key = "json"

    # in‑memory JSON string <-> objects
    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: str | bytes,
        /,
        *,
        many: bool = False,
        **kwargs,
    ):
        data = json.loads(obj)
        if many:
            return [subj_cls.model_validate(i, **kwargs) for i in data]
        return subj_cls.model_validate(data, **kwargs)

    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        many: bool = False,
        **kwargs,
    ) -> str:
        if many:
            payload = [
                s.model_dump(**kwargs)
                for s in (subj if isinstance(subj, list) else [subj])
            ]
        else:
            payload = subj.model_dump(**kwargs)
        return json.dumps(payload, indent=2, sort_keys=True)


class JsonFileAdapter(Adapter[T]):
    """Reads/writes JSON *files* instead of strings."""

    obj_key = ".json"  # file extension style

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
        text = path.read_text(encoding="utf-8")
        logging.info("Loaded JSON from %s", path)
        return JsonAdapter.from_obj(subj_cls, text, many=many, **kwargs)

    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        many: bool = False,
        **kwargs,
    ) -> None:
        path: Path = kwargs.get("path") or Path("output.json")
        text = JsonAdapter.to_obj(subj, many=many, **kwargs)
        path.write_text(text, encoding="utf-8")
        logging.info("Wrote JSON to %s", path)
