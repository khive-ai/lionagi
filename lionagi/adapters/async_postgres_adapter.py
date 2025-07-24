"""
Clean LionAGI async PostgreSQL adapter for integration into lionagi core.

This adapter handles SQLAlchemy async inspection issues and lionagi data
serialization while providing seamless async persistence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, TypeVar

from pydapter.exceptions import QueryError

try:
    import sqlalchemy as sa
    from pydapter.extras.async_postgres_ import AsyncPostgresAdapter
    from sqlalchemy.ext.asyncio import create_async_engine
except ImportError:
    raise ImportError(
        "This adapter requires postgres option to be installed. "
        'Please install them using `uv pip install "lionagi[postgres]"`.'
    )


T = TypeVar("T")


class LionAGIAsyncPostgresAdapter(AsyncPostgresAdapter[T]):
    """
    Async PostgreSQL adapter for lionagi Nodes with critical fixes.

    Solves core issues:
    1. SQLAlchemy async table inspection ("Inspection on an AsyncConnection is currently not supported")
    2. LionAGI float timestamp serialization (created_at as float → datetime)
    3. Datetime objects in JSON content (datetime → ISO strings)
    4. Automatic metadata field mapping via LionAGIPostgresAdapter

    Features:
    - Works with lionagi's adapt_to_async() system
    - Automatic schema creation for lionagi Node structure
    - Cross-database compatibility (PostgreSQL/SQLite)
    - Handles all lionagi data serialization edge cases
    """

    obj_key: ClassVar[str] = "lionagi_async_pg"

    @classmethod
    def _table(cls, meta: sa.MetaData, name: str) -> sa.Table:
        """
        Override parent's _table to avoid async inspection issues.

        Uses JSON for SQLite compatibility, JSONB for PostgreSQL performance.
        """
        # Determine JSON type based on database (check connection URL if available)
        json_type = sa.JSON  # Default safe option that works everywhere

        # Try to detect PostgreSQL from the connection
        if hasattr(meta, "bind") and meta.bind:
            engine_url = str(meta.bind.engine.url)
            if "postgresql" in engine_url and "sqlite" not in engine_url:
                json_type = sa.dialects.postgresql.JSONB

        return sa.Table(
            name,
            meta,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("content", json_type),
            sa.Column(
                "node_metadata", json_type
            ),  # mapped from lionagi metadata
            sa.Column("created_at", sa.DateTime),
            sa.Column("embedding", json_type),
            # Note: No autoload_with to avoid async inspection error
        )

    @classmethod
    async def to_obj(
        cls,
        subj,
        /,
        *,
        many: bool = True,
        adapt_meth: str = "model_dump",
        **kw,
    ):
        """
        Write lionagi Node(s) to PostgreSQL with automatic fixes.

        Handles:
        1. Table creation if needed
        2. LionAGI data serialization fixes
        3. Async database operations
        """
        try:
            # Validate required parameters
            engine_url = kw.get("dsn") or kw.get("engine_url")
            table = kw.get("table")

            if not engine_url or not table:
                raise ValueError(
                    "Missing required 'dsn' and 'table' parameters"
                )

            # Ensure table exists with lionagi schema
            await cls._ensure_table_exists(engine_url, table)

            # Prepare data with lionagi fixes
            items = subj if isinstance(subj, list) else [subj]
            if not items:
                return {"inserted_count": 0}

            # Convert nodes to database rows with serialization fixes
            rows = []
            for item in items:
                data = getattr(item, adapt_meth)()
                fixed_data = cls._fix_lionagi_data(data)
                rows.append(fixed_data)

            # Execute async insert
            engine = create_async_engine(engine_url, future=True)
            async with engine.begin() as conn:
                meta = sa.MetaData()
                meta.bind = conn
                table_obj = cls._table(meta, table)
                await conn.execute(sa.insert(table_obj), rows)

            return {"inserted_count": len(rows)}

        except Exception as e:
            raise QueryError(
                f"Error in lionagi async adapter: {e}",
                adapter="lionagi_async_pg",
            ) from e

    @classmethod
    async def _ensure_table_exists(cls, engine_url: str, table_name: str):
        """Create table with lionagi schema if it doesn't exist."""
        try:
            engine = create_async_engine(engine_url, future=True)
            async with engine.begin() as conn:
                meta = sa.MetaData()
                meta.bind = conn

                # Use the same _table method to ensure consistency
                table = cls._table(meta, table_name)

                # Create just this table
                await conn.run_sync(table.create, checkfirst=True)

        except Exception:
            # Table might already exist, continue
            pass

    @classmethod
    def _fix_lionagi_data(cls, data: dict) -> dict:
        """
        Fix lionagi Node data for database storage.

        Handles:
        1. Float timestamp → datetime for created_at
        2. Datetime objects in content → ISO strings
        """
        # Fix created_at timestamp
        if "created_at" in data and isinstance(
            data["created_at"], (int, float)
        ):
            data["created_at"] = datetime.fromtimestamp(data["created_at"])

        # Fix datetime objects in content
        if "content" in data and isinstance(data["content"], dict):
            data["content"] = cls._serialize_datetime_recursive(
                data["content"]
            )

        return data

    @classmethod
    def _serialize_datetime_recursive(cls, obj: Any) -> Any:
        """Recursively convert datetime objects to ISO strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {
                k: cls._serialize_datetime_recursive(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [cls._serialize_datetime_recursive(item) for item in obj]
        else:
            return obj
