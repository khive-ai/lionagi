from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .protocol import SpecAdapter

# Supported adapter types
AdapterType = Literal["pydantic", "sql", "dataclass"]

__all__ = (
    "AdapterType",
    "get_adapter",
)


@functools.cache
def get_adapter(adapter_name: str) -> type[SpecAdapter]:
    """Get adapter class by name (cached).

    Factory method for adapter classes. Caches to avoid repeated imports.

    Args:
        adapter_name: Adapter identifier ("pydantic", "sql", "dataclass", future: "rust")

    Returns:
        Adapter class (not instance)

    Raises:
        ValueError: If adapter not supported
        ImportError: If adapter dependencies not installed
    """
    match adapter_name:
        case "pydantic":
            try:
                from ._pydantic import PydanticSpecAdapter

                return PydanticSpecAdapter
            except ImportError as e:
                raise ImportError(
                    "PydanticSpecAdapter requires Pydantic. Install with: pip install pydantic"
                ) from e
        case "sql":
            from lionagi.libs.db.sql_ddl import SQLSpecAdapter

            return SQLSpecAdapter
        case "dataclass":
            from ._dataclass import DataClassSpecAdapter

            return DataClassSpecAdapter
        # case "rust":
        #     from .spec_adapters.rust_field import RustSpecAdapter
        #     return RustSpecAdapter
        case _:
            raise ValueError(f"Unsupported adapter: {adapter_name}")
