# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""lionagi.beta.core — The substrate for Agent orchestration.

Architecture layers (bottom to top):

  Types & Identity
    Element        UUID-identified Pydantic model (base for all objects)
    Capability     Immutable rights token (msgspec.Struct)
    Principal      Identified process with context + capabilities (Pydantic)
    Observation    Runtime event record (msgspec.Struct)

  Execution Units
    Morphism       Atomic async executable (msgspec.Struct base class)
    OpNode         Morphism + dependencies + params (dataclass)
    OpGraph        DAG of OpNodes (dataclass, topological sort)
    morph/         Morphism combinators: bind, patch, retry, timeout, subgraph

  Enforcement
    policy_check   Capability gate: principal.rights ⊇ morphism.requires?
    Invariant      Protocol for pre/post condition checks
    IPU            Invariant chain runner (LenientIPU | StrictIPU)

  Event System
    EventBus       Topic-keyed pub/sub for infrastructure events

  Execution Engine
    Runner         One-shot DAG executor (OpGraph + Principal → results)

  Base Primitives (from core/base/)
    Pile           UUID-keyed O(1) collection (Pydantic)
    Progression    Ordered UUID sequence (Pydantic)
    Flow           Pile + Progressions (Pydantic)
    Node           Graph node (Pydantic)
    Graph          Directed graph (Pydantic)
    Event          Async invocable with lifecycle tracking (Pydantic)
    EventStatus    Execution state enum

  Service Base
    Service        Named backend with @resource-decorated actions

Execution naming:
    Runner         -- one-shot OpGraph execution
    EventQueue     -- rate-limited Event queue (was: Processor)
    EventTracker   -- Event lifecycle tracker (was: Executor)
    Pipeline       -- declarative workflow definition (was: Worker)
    PipelineRunner -- Pipeline driver (was: WorkerEngine)
"""

from __future__ import annotations

# --- Core substrate ---
from lionagi.beta.core.types import Principal, Capability, Observation, now_utc
from lionagi.beta.core.morphism import Morphism, MorphismAdapter, MorphismLike
from lionagi.beta.core.graph import OpGraph, OpNode
from lionagi.beta.core.policy import policy_check
from lionagi.beta.core.ipu import (
    IPU,
    Invariant,
    LatencyBound,
    LenientIPU,
    PolicyGatePresent,
    ResultShape,
    StrictIPU,
    default_invariants,
)
from lionagi.protocols.generic.eventbus import EventBus, Handler
from lionagi.beta.core.morph import BaseOp
from lionagi.beta.core.runner import Runner
from lionagi.beta.resource.service import Normalized, ResourceMeta, Service, resource

# --- Base primitives (lazy) ---
# These are accessed via __getattr__ to keep core import startup minimal.

_LAZY: dict[str, tuple[str, str]] = {
    "Element": ("lionagi.protocols.generic.element", "Element"),
    "Event": ("lionagi.protocols.generic.event", "Event"),
    "EventStatus": ("lionagi.protocols.generic.event", "EventStatus"),
    "Execution": ("lionagi.protocols.generic.event", "Execution"),
    "Pile": ("lionagi.beta.core.base.pile", "Pile"),
    "Progression": ("lionagi.beta.core.base.progression", "Progression"),
    "Node": ("lionagi.beta.core.base.node", "Node"),
    "Edge": ("lionagi.beta.core.base.graph", "Edge"),
    "EdgeCondition": ("lionagi.beta.core.base.graph", "EdgeCondition"),
    "Graph": ("lionagi.beta.core.base.graph", "Graph"),
    # Execution model aliases.
    "EventQueue": ("lionagi.beta.core.base.processor", "Processor"),
    "EventTracker": ("lionagi.beta.core.base.processor", "Executor"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    if name in _LOADED:
        return _LOADED[name]
    if name in _LAZY:
        from importlib import import_module
        module_path, attr = _LAZY[name]
        mod = import_module(module_path)
        val = getattr(mod, attr)
        _LOADED[name] = val
        return val
    raise AttributeError(f"module 'lionagi.beta.core' has no attribute {name!r}")


__all__ = [
    # Substrate (always available)
    "Principal",
    "Capability",
    "Observation",
    "now_utc",
    "Morphism",
    "MorphismAdapter",
    "MorphismLike",
    "OpGraph",
    "OpNode",
    "policy_check",
    "IPU",
    "Invariant",
    "LatencyBound",
    "LenientIPU",
    "PolicyGatePresent",
    "ResultShape",
    "StrictIPU",
    "default_invariants",
    "EventBus",
    "Handler",
    "BaseOp",
    "Runner",
    "Normalized",
    "ResourceMeta",
    "Service",
    "resource",
    # Lazy
    "Element",
    "Event",
    "EventStatus",
    "Execution",
    "Pile",
    "Progression",
    "Node",
    "Edge",
    "EdgeCondition",
    "Graph",
    "EventQueue",
    "EventTracker",
]
