# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

E = TypeVar("E")


__all__ = (
    "Observer",
    "Observable",
    "Observation",
    "Invariant",
    "Relational",
    "Sendable",
    "Communicatable",
    "Collective",
    "Ordering",
    "Manager",
    "Composable",
    "Condition",
)


class Observer(ABC):
    """Lion ecosystem itself as an observer"""


class Observable(ABC):
    """Observable entities must define 'id'."""


class Observation(ABC):
    """Base for observations."""


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


# specialized observables
class Relational(Observable):
    """Base for graph-connectable objects."""


class Sendable(Observable):
    """Sendable entities must define 'sender' and 'recipient'."""


class Communicatable(Observable):
    """Communicatable must define 'mailbox' and send/receive methods."""

    @abstractmethod
    def send(self, *args, **kwargs):
        pass


class Collective(Observable, Generic[E]):
    """Base for collections of elements."""

    @abstractmethod
    def include(self, item, /):
        pass

    @abstractmethod
    def exclude(self, item, /):
        pass


class Ordering(Observable, Generic[E]):
    """Base for element orderings."""

    @abstractmethod
    def include(self, item, /):
        pass

    @abstractmethod
    def exclude(self, item, /):
        pass


# other specialized concepts
class Manager(ABC):
    """Base for all managers."""


class Composable(Invariant):
    """
    Invariants that can be composed and transformed through pure functions.
    
    Composable invariants enable building sophisticated measurement standards
    from simple mathematical components while preserving invariant properties:
    
    - Combined with other composable invariants
    - Transformed through factory methods (with_*, as_*, from_*)
    - Materialized into runtime objects for IPU validation
    - Maintain mathematical consistency under all transformations
    
    The composition operations preserve invariant properties - no matter how
    you transform/combine them, they remain mathematically consistent contracts
    suitable for IPU observation and trustless coordination.
    
    Examples: FieldModel (composable validation rules), Operable (composable capabilities)
    
    Mathematical Foundation:
    For any composition function f: Composable → Composable,
    Invariant(x) → Invariant(f(x)) must hold
    """


class Condition(Invariant):
    """Base for conditions."""

    @abstractmethod
    async def apply(self, *args, **kwargs) -> bool:
        pass


# File: lionagi/protocols/_concepts.py
