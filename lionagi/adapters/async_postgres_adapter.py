"""
Simplified LionAGI async PostgreSQL adapter for pydapter v1.0.4+

This adapter leverages pydapter's improved raw SQL handling.
No workarounds needed - pydapter now properly handles:
- Raw SQL without table parameter
- No table inspection for raw SQL
- ORDER BY operations
- Both SQLite and PostgreSQL connections
"""

from __future__ import annotations

from typing import ClassVar, TypeVar

import sqlalchemy as sa
from pydapter.extras.async_postgres_ import AsyncPostgresAdapter
from sqlalchemy.ext.asyncio import create_async_engine

from ._utils import check_async_postgres_available

_ASYNC_POSTGRES_AVAILABLE = check_async_postgres_available()

if isinstance(_ASYNC_POSTGRES_AVAILABLE, ImportError):
    raise _ASYNC_POSTGRES_AVAILABLE

T = TypeVar("T")


class LionAGIAsyncPostgresAdapter(AsyncPostgresAdapter[T]):
    """
    Streamlined async adapter for lionagi Nodes.

    Features:
    - Auto-creates tables with lionagi schema
    - Inherits all pydapter v1.0.4+ improvements
    - No workarounds needed for SQLite or raw SQL
    """

    obj_key: ClassVar[str] = "lionagi_async_pg"

    @classmethod
    async def to_obj(
        cls,
        subj,
        /,
        *,
        many: bool = True,
        adapt_meth: str = None,
        **kw,
    ):
        """Write lionagi Node(s) to database with auto-table creation."""
        # Auto-create table if needed
        if table := kw.get("table"):
            if engine_url := (kw.get("dsn") or kw.get("engine_url")):
                await cls._ensure_table(engine_url, table)
            elif engine := kw.get("engine"):
                await cls._ensure_table(engine, table)

        return await super().to_obj(
            subj, many=many, adapt_meth=adapt_meth, **kw
        )

    @classmethod
    async def _ensure_table(cls, engine_or_url, table_name: str):
        """Create table with lionagi schema if it doesn't exist."""
        should_dispose = False
        if isinstance(engine_or_url, str):
            engine = create_async_engine(engine_or_url, future=True)
            should_dispose = True
        else:
            engine = engine_or_url

        try:
            async with engine.begin() as conn:
                # Determine JSON type based on database
                engine_url = str(engine.url)
                json_type = (
                    sa.dialects.postgresql.JSONB
                    if "postgresql" in engine_url
                    else sa.JSON
                )

                # Create table with lionagi schema
                await conn.run_sync(
                    lambda sync_conn: sa.Table(
                        table_name,
                        sa.MetaData(),
                        sa.Column("id", sa.String, primary_key=True),
                        sa.Column("content", json_type),
                        sa.Column("node_metadata", json_type),
                        sa.Column("created_at", sa.DateTime),
                        sa.Column("embedding", json_type, nullable=True),
                    ).create(sync_conn, checkfirst=True)
                )
        finally:
            if should_dispose:
                await engine.dispose()
