from abc import ABC, abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar
from uuid import UUID

T = TypeVar("T")


class Observable(ABC):
    """A marker interface for observable entities."""


E = TypeVar("E", bound=Observable)


class Ordering(Generic[T]):
    """A base class for ordered collections of items."""

    @abstractmethod
    def __list__(self) -> list[T]: ...

    @abstractmethod
    def include(self, items: Any) -> bool: ...

    @abstractmethod
    def exclude(self, items: Any) -> bool: ...


class Collective(Generic[E]):
    """A base class for collections of elements."""

    @abstractmethod
    def keys(self) -> list[UUID]: ...

    @abstractmethod
    def include(self, items: Any) -> bool: ...

    @abstractmethod
    def exclude(self, items: Any) -> bool: ...


IDAble: TypeAlias = UUID | E
IDLike: TypeAlias = UUID | str | E

__all__ = (
    "Observable",
    "Ordering",
    "Collective",
    "IDAble",
    "IDLike",
)
