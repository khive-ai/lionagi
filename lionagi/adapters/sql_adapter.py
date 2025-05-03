# sql_adapter.py
"""
SQLAdapter - converts between DB rows and pydantic models.
Requires:  sqlalchemy>=2.0
"""

from __future__ import annotations

from typing import TypeVar
from collections.abc import Sequence

from pydantic import BaseModel

try:
    import sqlalchemy as sa  # type: ignore[import]
except ImportError:
    raise ImportError(
        "SQLAdapter requires sqlalchemy>=2.0.0. "
        "Install with: pip install sqlalchemy"
    )


from .adapter import Adapter

T = TypeVar("T", bound=BaseModel)


class SQLAdapter(Adapter[T]):
    """Generic adapter using SQLAlchemy *Core*.

    *obj_key* is `"sql"`.
    `obj` argument is **always** a mapping with at least:

    ```python
    {
        "engine_url": "sqlite:///tmp.db",
        "table": "trades",
        "selectors": {"id": 5}          # optional for .from_obj
    }
    ```
    """

    obj_key = "sql"

    # ------------ helpers -------------------------------------------------
    @staticmethod
    def _table(metadata: sa.MetaData, table_name: str) -> sa.Table:
        return sa.Table(table_name, metadata, autoload_with=metadata.bind)

    @staticmethod
    def _from_row(model_cls: type[T], row: sa.Row | dict) -> T:
        return model_cls.model_validate(dict(row))

    # ------------ incoming ------------------------------------------------
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
        url = obj["engine_url"]
        table_name = obj["table"]
        selectors = obj.get("selectors") or {}

        engine = sa.create_engine(url, future=True)
        md = sa.MetaData(bind=engine)
        tbl = cls._table(md, table_name)

        stmt = sa.select(tbl).filter_by(**selectors)
        with engine.begin() as conn:
            rows = conn.execute(stmt).fetchall()

        if many:
            return [cls._from_row(subj_cls, r) for r in rows]
        if not rows:
            raise ValueError("No row matched selectors")
        return cls._from_row(subj_cls, rows[0])

    # ------------ outgoing ------------------------------------------------
    @classmethod
    def to_obj(
        cls,
        subj: T | Sequence[T],
        /,
        *,
        many: bool = False,
        engine_url: str,
        table: str,
        **kwargs,
    ) -> None:
        items = subj if isinstance(subj, Sequence) else [subj]
        payload = [i.model_dump() for i in items]

        engine = sa.create_engine(engine_url, future=True)
        md = sa.MetaData(bind=engine)
        tbl = cls._table(md, table)

        with engine.begin() as conn:
            conn.execute(sa.insert(tbl), payload)
