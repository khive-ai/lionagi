from enum import Enum as _Enum
from typing import Any, Sequence

from typing_extensions import TypedDict

__all__ = (
    "Enum",
    "KeysDict",
    "KeysLike",
)


class Enum(_Enum):
    """Enhanced Enum with utility methods."""

    @classmethod
    def allowed(cls) -> tuple[str, ...]:
        """Return tuple of all enum values."""
        return tuple(e.value for e in cls)


class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""

    key: Any  # Represents any key-type pair


KeysLike = Sequence[str] | KeysDict
