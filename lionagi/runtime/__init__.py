"""Runtime system for lionagi - IPU, morphism execution, and event coordination.

Core concepts from v1:
- IPU: Invariant Processing Unit that observes and enforces system invariants
- Runner: Executes morphisms with IPU observation
- EventBus: Coordinates service requests and responses
- Policy: Capability-based access control

Adapted for current lionagi:
- Works with existing Morphism class
- Uses Branch as context carrier
- Rules are system morphisms
"""

from .ipu import IPU, Invariant
from .eventbus import EventBus
from .runner import Runner
from .capability import Capability
from .policy import policy_check

__all__ = [
    "IPU",
    "Invariant",
    "EventBus",
    "Runner",
    "Capability",
    "policy_check",
]