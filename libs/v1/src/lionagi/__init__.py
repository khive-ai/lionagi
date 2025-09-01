"""LION V1: Capability-Based AI Orchestration

Ocean's vision: "composability is the sh*t of our framework"

Core Insight: Fields are capabilities. When AI outputs request events,
those are capability requests that must be validated through invariants.

Architecture:
- Capability: Our compositional privilege system (was FieldModel)
- Pydantic: Current implementation backend
- msgspec: Wire protocol serialization (pure speed)
- Rust: Future enterprise backend (mathematically proven)
"""

__version__ = "1.0.0a1"
__author__ = "Ocean (HaiyangLi)"

# Core contracts first
from .kernel.foundation.contracts import (
    Observable,
    Observer,
    Invariant,
    Composable,
    Observation,
    Effect,
)

# Capability system (fields are privileges!)
from .kernel.foundation.capability import (
    Capability,      # Was FieldModel - represents access/privilege
    CapabilityMeta,  # Was FieldMeta - metadata about the capability
    CapabilityTemplate,  # Template for composing capabilities
)

# Domain patterns (v0 battle-tested)
from .domain.generic.pile import Pile
from .domain.generic.element import Element  
from .domain.generic.event import Event
from .domain.generic.progression import Progression

# Services (high-level APIs)
from .services.session import Session
from .services.branch import Branch
from .services.orchestrator import Orchestrator

__all__ = [
    # Foundation
    "Observable",
    "Observer",
    "Invariant", 
    "Composable",
    "Observation",
    "Effect",
    # Capability system
    "Capability",
    "CapabilityMeta",
    "CapabilityTemplate",
    # Domain
    "Pile",
    "Element",
    "Event",
    "Progression",
    # Services
    "Session",
    "Branch",
    "Orchestrator",
]