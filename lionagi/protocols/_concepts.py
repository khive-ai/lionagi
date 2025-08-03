# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

E = TypeVar("E")


__all__ = (
    "Observer",
    "Manager",
    "Relational",
    "Sendable",
    "Observable",
    "Communicatable",
    "Condition",
    "Collective",
    "Ordering",
)


class Observer(ABC):
    """Base for all observers."""

    pass


class Manager(Observer):
    """Base for all managers."""

    pass


class Relational(ABC):
    """Base for graph-connectable objects."""

    pass


class Sendable(ABC):
    """Sendable entities must define 'sender' and 'recipient'."""

    pass


class Observable(ABC):
    """Observable entities must define 'id'."""

    pass


class Communicatable(Observable):
    """Communicatable must define 'mailbox' and send/receive methods."""

    @abstractmethod
    def send(self, *args, **kwargs):
        pass


class Condition(ABC):
    """Base for conditions."""

    @abstractmethod
    async def apply(self, *args, **kwargs) -> bool:
        pass


class Collective(ABC, Generic[E]):
    """Base for collections of elements."""

    @abstractmethod
    def include(self, item, /):
        pass

    @abstractmethod
    def exclude(self, item, /):
        pass


class Ordering(ABC, Generic[E]):
    """Base for element orderings."""

    @abstractmethod
    def include(self, item, /):
        pass

    @abstractmethod
    def exclude(self, item, /):
        pass


class Invariant(ABC):
    """
    Base for system invariants - unchanging foundational rules and contracts.

    Invariants are the "physical laws" of intelligence systems. Unlike Observable 
    entities (Session, Branch) which have IDs, state changes, and communication, 
    Invariant entities represent constant mathematical/structural definitions 
    that define how Observable entities can behave.

    Key Properties:
    - Mathematical contracts that remain constant during IPU observation
    - Define measurement standards for Observable behavior validation
    - Enable trustless coordination through shared validation rules
    - Form the theoretical foundation of Computational Cognitive Physics
    
    Design Pattern:
    - Non-constructable: Use factory methods, never direct construction
    - Immutable: All properties frozen after creation
    - Cacheable: Aggressive caching following FieldModel patterns
    - Composable: Can be combined through mathematical operations
    """
    

# File: lionagi/protocols/_concepts.py
