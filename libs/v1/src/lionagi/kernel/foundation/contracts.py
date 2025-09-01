"""Core philosophical contracts for LION v1.

Ocean: "ABCs actually change the entire projection of a framework"

These aren't just interfaces - they're compositional contracts that enable trust.
Based on formal proofs Chapter 1: Category Theory foundations.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, TypeVar, Generic, Any
from uuid import UUID
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')

# ============== The Five Pillars ==============

@runtime_checkable
class Observable(Protocol):
    """Everything in LION can be observed.
    
    Minimal atom - just needs identity. Everything adapts to this.
    """
    
    @property
    def id(self) -> UUID:
        """Unique identity - the only required field"""
        ...


class Morphism(ABC, Generic[T]):
    """Transformations between observables.
    
    From Chapter 1: Category Theory. Morphisms are the arrows in our category.
    Invariants ensure properties are preserved across morphisms.
    """
    
    @abstractmethod
    def apply(self, source: T) -> T:
        """Transform source to target, preserving invariants"""
        pass
    
    @abstractmethod
    def compose(self, other: 'Morphism[T]') -> 'Morphism[T]':
        """Compose with another morphism (associative)"""
        pass
    
    @property
    @abstractmethod
    def invariants(self) -> tuple['Invariant', ...]:
        """What must be preserved during this transformation"""
        pass


class Invariant(ABC):
    """A truth that must hold during morphisms.
    
    Ocean: "invariant came after I tried to formalize field_model"
    
    Invariants are NOT tied to observables, they're tied to morphisms!
    They ensure transformations preserve essential properties.
    """
    
    @abstractmethod
    def check(self, before: Any, after: Any) -> bool:
        """Does the transformation preserve this invariant?"""
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of what this preserves"""
        pass


class Observer(ABC):
    """The watcher - LION system itself.
    
    Ocean: "lion system=observer, observation=AI model give output"
    """
    
    @abstractmethod
    async def observe(self, observable: Observable) -> 'Observation':
        """Observe, validate, and decide effects"""
        pass
    
    @abstractmethod
    def can_observe(self, observable: Observable) -> bool:
        """Do I have the capability to observe this?"""
        pass


class Composable(ABC, Generic[T]):
    """Things that can be safely composed.
    
    Ocean: "composability is the sh*t of our framework"
    
    Composition must respect morphisms and preserve invariants.
    """
    
    @abstractmethod
    def compose(self, other: T) -> T:
        """Compose with another, maintaining invariants"""
        pass
    
    @abstractmethod
    def decompose(self) -> tuple[T, ...]:
        """Break into components"""
        pass


# ============== Core Data Structures ==============

@dataclass
class Observation:
    """Result of observing - carries trust and effects"""
    observable_id: UUID
    observer_id: UUID
    timestamp: float
    valid: bool
    trust_level: float
    capabilities_requested: set[str]
    capabilities_granted: set[str]
    effects: list['Effect']
    reason: str | None = None


@dataclass
class Effect:
    """What happens as result of observation"""
    type: str  # "grant_capability", "trigger_event", "update_state"
    target: UUID
    params: dict[str, Any]
    requires_capability: str | None = None