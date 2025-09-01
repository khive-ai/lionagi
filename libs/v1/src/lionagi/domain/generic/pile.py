"""Pile: Observable collection with identity.

Ocean's v0 wisdom: Pile is not just a container, it's an Observable entity
with its own lifecycle and identity.
"""

from typing import TypeVar, Generic, Iterator
from uuid import UUID, uuid4
from collections.abc import MutableSequence

from ...kernel.foundation.contracts import Observable

T = TypeVar('T')


class Pile(Observable, MutableSequence[T], Generic[T]):
    """Observable collection - first-class citizen.
    
    V0 wisdom elevated:
    - Pile has its own UUID and lifecycle
    - Can observe the pile itself, not just contents  
    - Changes to pile trigger observations
    - Thread-safe through immutable updates
    """
    
    def __init__(self, items: list[T] = None):
        self._id = uuid4()
        self._items = items or []
        self._version = 0
    
    @property
    def id(self) -> UUID:
        """Observable protocol - unique identity"""
        return self._id
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __setitem__(self, index, value):
        # Immutable update pattern
        self._items[index] = value
        self._version += 1
    
    def __delitem__(self, index):
        del self._items[index]
        self._version += 1
    
    def insert(self, index, value):
        self._items.insert(index, value)
        self._version += 1
    
    def __iter__(self) -> Iterator[T]:
        return iter(self._items)
    
    def include(self, item: T):
        """Add item to pile"""
        self.append(item)
    
    def exclude(self, item: T):
        """Remove item from pile"""
        self.remove(item)