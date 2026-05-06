# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""lionagi.core — The substrate for Agent orchestration.

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
    binders        Morphism parameter binding (BoundOp)
    wrappers       Morphism combinators: retry, timeout, subgraph, ctx_set

  Enforcement
    policy_check   Capability gate: principal.rights ⊇ morphism.requires?
    Invariant      Protocol for pre/post condition checks
    IPU            Invariant chain runner (LenientIPU | StrictIPU)

  Event System
    EventBus       Topic-keyed pub/sub for infrastructure events

  Execution Engine
    Runner         One-shot DAG executor (OpGraph + Principal → results)

  Base Primitives (from beta.resource/ and protocols/generic/)
    Pile           UUID-keyed O(1) collection (beta.resource.pile)
    Progression    Ordered UUID sequence (protocols.generic.progression)
    Flow           Pile + Progressions (beta.resource.flow)
    Node           Graph node (beta.resource.node)
    Graph          Directed graph (beta.resource.graph)
    Event          Async invocable with lifecycle tracking (protocols.generic.event)
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

from lionagi.core.binders import BoundOp
from lionagi.core.graph import OpGraph, OpNode
from lionagi.core.ipu import (
    IPU,
    Invariant,
    LatencyBound,
    LenientIPU,
    PolicyGatePresent,
    ResultShape,
    StrictIPU,
    default_invariants,
)
from lionagi.core.morphism import Morphism, MorphismAdapter, MorphismLike
from lionagi.core.policy import policy_check
from lionagi.core.runner import Runner

# --- Core substrate ---
from lionagi.core.types import Capability, Observation, Principal, now_utc
from lionagi.core.wrappers import BaseOp
from lionagi.service.resource import Normalized, ResourceMeta, Service, resource
from lionagi.protocols.generic.eventbus import EventBus, Handler

__all__ = [
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
    "BoundOp",
    "Runner",
    "Normalized",
    "ResourceMeta",
    "Service",
    "resource",
]
