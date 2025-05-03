"""
MongoAdapter – converts between MongoDB documents and pydantic models.
Requires:  pymongo>=4
"""

from __future__ import annotations

from typing import List, TypeVar
from collections.abc import Sequence

from pydantic import BaseModel

try:
    from pymongo import MongoClient  # type: ignore[import]
except ImportError:
    raise ImportError(
        "MongoAdapter requires pymongo>=4.0.0. "
        "Install with: pip install pymongo"
    )

from .adapter import Adapter

T = TypeVar("T", bound=BaseModel)


class MongoAdapter(Adapter[T]):
    obj_key = "mongo"

    @classmethod
    def _client(cls, url: str) -> MongoClient:
        return MongoClient(url)  # connection pooling handled internally

    # ---------------- incoming -------------------------------------------
    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: dict,
        /,
        *,
        many: bool = False,
        **kwargs,
    ) -> T | list[T]:
        """
        obj must contain:  { "mongo_url": "...", "db": "...", "collection": "...", "filter": {...}}
        """
        url = obj["mongo_url"]
        db = obj["db"]
        coll = obj["collection"]
        flt = obj.get("filter") or {}

        cur = cls._client(url)[db][coll].find(flt)
        docs = list(cur)

        if many:
            return [subj_cls.model_validate(d) for d in docs]
        if not docs:
            raise ValueError("No document matched filter")
        return subj_cls.model_validate(docs[0])

    # ---------------- outgoing -------------------------------------------
    @classmethod
    def to_obj(
        cls,
        subj: T | Sequence[T],
        /,
        *,
        mongo_url: str,
        db: str,
        collection: str,
        many: bool = False,
        **kwargs,
    ) -> None:
        docs = subj if isinstance(subj, Sequence) else [subj]
        payload = [d.model_dump(by_alias=True) for d in docs]

        cls._client(mongo_url)[db][collection].insert_many(payload)
