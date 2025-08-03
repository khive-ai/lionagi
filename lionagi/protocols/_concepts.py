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
    """system invariants"""


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
    Invariants that CAN BE composed and transformed through pure functions.

    Composable invariants can be combined while preserving their mathematical nature:
    - Combined with other composable invariants
    - Transformed through factory methods (with_*, as_*)
    - Materialized into runtime objects for validation

    The composition operations preserve invariant properties - no matter how
    you transform/combine them, they remain mathematically consistent contracts.
    """


class Condition(Invariant):
    """Base for conditions."""

    @abstractmethod
    async def apply(self, *args, **kwargs) -> bool:
        pass


# File: lionagi/protocols/_concepts.py
