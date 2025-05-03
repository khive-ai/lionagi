"""
Neo4jAdapter – maps nodes / node-lists ⇄ pydantic models.

Requirements:
  • your model's field names == Neo4j property keys
  • there is a (:Label) matching the model class name by default
"""

from __future__ import annotations
from typing import Sequence, TypeVar
from pydantic import BaseModel

try:
    from neo4j import GraphDatabase # type: ignore[import]
except ImportError:
    raise ImportError(
        "Neo4jAdapter requires neo4j>=5.0.0. "
        "Install with: pip install neo4j"
    )

from .adapter import Adapter

T = TypeVar("T", bound=BaseModel)

class Neo4jAdapter(Adapter[T]):
    obj_key = "neo4j"

    # ---------------- incoming --------------------------
    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: dict,
        /,
        *,
        many: bool = False,
        **kwargs,
    ):
        url   = obj["url"]          # bolt://user:pass@host:7687
        label = obj.get("label", subj_cls.__name__)
        where = obj.get("where", "")      # Cypher predicate string

        cypher = f"MATCH (n:`{label}`) {f'WHERE {where}' if where else ''} RETURN n"

        driver = GraphDatabase.driver(url)
        with driver.session() as sess:
            result = sess.run(cypher)
            records = [r["n"]._properties for r in result]  # neo4j types -> dict

        if many:
            return [subj_cls.model_validate(r) for r in records]
        if not records:
            raise ValueError("No nodes matched query")
        return subj_cls.model_validate(records[0])

    # ---------------- outgoing --------------------------
    @classmethod
    def to_obj(
        cls,
        subj: T | Sequence[T],
        /,
        *,
        url: str,
        label: str | None = None,
        many: bool = False,
        merge_on: str = "id",
        **kwargs,
    ) -> None:
        items = subj if isinstance(subj, Sequence) else [subj]
        label = label or items[0].__class__.__name__

        driver = GraphDatabase.driver(url)
        with driver.session() as sess:
            for it in items:
                props = it.model_dump()
                merge_val = props[merge_on]

                cypher = (
                    f"MERGE (n:`{label}` {{{merge_on}: $merge_val}}) "
                    f"SET n += $props"
                )
                sess.run(cypher, merge_val=merge_val, props=props)
