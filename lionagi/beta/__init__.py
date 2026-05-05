# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""lionagi.beta — Next-generation substrate for Agent orchestration.

Architecture overview:
    lionagi.beta.core          Substrate (types, morphism, policy, graph, runner, events)
    lionagi.beta.lndl          LNDL parser (stable, not modified)
    lionagi.beta.resource      Resource backends (iModel, hooks, registry)
    lionagi.beta.session       Session/Branch/Exchange
    lionagi.beta.operations    High-level operations (generate, operate, react, etc.)
    lionagi.beta.work          Declarative pipeline definition (Worker/WorkerEngine)
    lionagi.beta.errors        Exception hierarchy
    lionagi.beta.protocols     Structural protocols (@implements, Morphism, etc.)

Quick start (substrate only):
    from lionagi.beta.core import (
        Principal, Capability, Morphism, OpGraph, OpNode,
        Runner, StrictIPU, default_invariants, EventBus,
    )

Note:
    Submodules are loaded lazily to avoid import errors when optional
    dependencies such as rapidfuzz are not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Only import the substrate core eagerly — it has minimal dependencies.
# All other submodules are lazy to avoid pulling in optional deps at import time.
from lionagi.beta import core  # noqa: F401 — always available

__all__ = ["core"]
__version__ = "0.1.0"


_LAZY_SUBMODULES: dict[str, str] = {
    "errors": "lionagi._errors",
    "protocols": "lionagi.beta.protocols",
    "lndl": "lionagi.beta.lndl",
    "resource": "lionagi.beta.resource",
    "session": "lionagi.beta.session",
    "operations": "lionagi.beta.operations",
    "work": "lionagi.beta.work",
}

_LOADED_SUBMODULES: dict[str, object] = {}


def __getattr__(name: str) -> object:
    if name in _LOADED_SUBMODULES:
        return _LOADED_SUBMODULES[name]
    if name in _LAZY_SUBMODULES:
        from importlib import import_module

        mod = import_module(_LAZY_SUBMODULES[name])
        _LOADED_SUBMODULES[name] = mod
        return mod
    raise AttributeError(f"module 'lionagi.beta' has no attribute {name!r}")


if TYPE_CHECKING:
    from lionagi.beta import errors, lndl, protocols  # noqa: F401
