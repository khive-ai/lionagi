from .factory import AdapterType, get_adapter
from .protocol import SpecAdapter

__all__ = (
    "AdapterType",
    "DataClassSpecAdapter",
    "PydanticSpecAdapter",
    "SpecAdapter",
    "get_adapter",
)


def __getattr__(name: str):
    if name == "DataClassSpecAdapter":
        from ._dataclass import DataClassSpecAdapter

        return DataClassSpecAdapter
    if name == "PydanticSpecAdapter":
        from ._pydantic import PydanticSpecAdapter

        return PydanticSpecAdapter
    raise AttributeError(name)
