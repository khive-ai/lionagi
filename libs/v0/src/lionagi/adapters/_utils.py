def check_async_postgres_available():
    try:
        import sqlalchemy as sa
        from pydapter.extras.async_postgres_ import AsyncPostgresAdapter
        from sqlalchemy.ext.asyncio import create_async_engine

        return True
    except Exception:
        return ImportError(
            "This adapter requires postgres option to be installed. "
            'Please install them using `uv pip install "lionagi[postgres]"`.'
        )


def check_postgres_available():
    try:
        from pydapter.model_adapters.postgres_model import PostgresModelAdapter
        from sqlalchemy import String
        from sqlalchemy.orm import DeclarativeBase

        return True
    except Exception:
        return ImportError(
            "This adapter requires postgres option to be installed. "
            'Please install them using `uv pip install "lionagi[postgres]"`.'
        )
