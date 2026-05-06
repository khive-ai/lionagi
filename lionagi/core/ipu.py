# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Invariant Policy Unit: pre/post enforcement hooks around morphism execution."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from .graph import OpNode
from .policy import policy_check
from .types import Observation, Principal

try:
    import msgspec as _msgspec
except ImportError:  # pragma: no cover
    _msgspec = None  # type: ignore

__all__ = (
    "IPU",
    "Invariant",
    "LatencyBound",
    "LenientIPU",
    "PolicyGatePresent",
    "ResultShape",
    "StrictIPU",
    "default_invariants",
)

logger = logging.getLogger(__name__)


class Invariant(Protocol):
    """Protocol for pre/post condition checks run by the IPU."""

    name: str

    def pre(self, br: Principal, node: OpNode) -> bool: ...

    def post(self, br: Principal, node: OpNode, result: dict[str, Any]) -> bool: ...


class IPU(Protocol):
    """Composable execution enforcement with before/after node hooks."""

    name: str
    invariants: list[Invariant]

    async def before_node(self, br: Principal, node: OpNode) -> None: ...
    async def after_node(
        self,
        br: Principal,
        node: OpNode,
        result: dict[str, Any],
        *,
        error: BaseException | None = None,
    ) -> None: ...
    async def on_observation(self, obs: Observation) -> None: ...


class PolicyGatePresent:
    """Sentinel invariant declaring that policy enforcement is required.

    The Runner performs the authoritative check before ipu.before_node(); this
    invariant cannot re-check because it lacks access to accumulated_provides.
    """

    name = "PolicyGatePresent"

    def pre(self, br: Principal, node: OpNode) -> bool:
        return True

    def post(self, br: Principal, node: OpNode, result: dict[str, Any]) -> bool:
        return True


# ---------------------------------------------------------------------------
# Opt-in invariants (activated by morphism attribute declarations)
# ---------------------------------------------------------------------------


class LatencyBound:
    """Enforce per-node latency budget; opt-in via morphism.latency_budget_ms attribute."""

    name = "LatencyBound"

    def __init__(self) -> None:
        self._t0: dict[tuple[Any, Any], float] = {}

    def pre(self, br: Principal, node: OpNode) -> bool:
        self._t0[(br.id, node.id)] = time.perf_counter()
        return True

    def post(self, br: Principal, node: OpNode, result: dict[str, Any]) -> bool:
        budget = getattr(node.m, "latency_budget_ms", None)
        if budget is None:
            self._t0.pop((br.id, node.id), None)
            return True
        t0 = self._t0.pop((br.id, node.id), time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms <= float(budget)


class ResultShape:
    """Validate result shape; opt-in via morphism.result_keys or morphism.result_schema attributes."""

    name = "ResultShape"

    def pre(self, br: Principal, node: OpNode) -> bool:
        return True

    def post(self, br: Principal, node: OpNode, result: dict[str, Any]) -> bool:
        required = getattr(node.m, "result_keys", None)
        if required:
            if not isinstance(result, dict):
                return False
            miss = set(required) - set(result.keys())
            if miss:
                return False

        schema = getattr(node.m, "result_schema", None)
        if schema is not None and _msgspec is not None:
            try:
                payload = _msgspec.json.encode(result)
                _msgspec.json.decode(payload, type=schema)
            except Exception:
                return False

        return True


# ---------------------------------------------------------------------------
# IPU Implementations
# ---------------------------------------------------------------------------


_RESULT_DEPENDENT_NAMES = frozenset({"ResultShape"})


def _is_result_dependent(inv: Invariant) -> bool:
    """True if this invariant's post-check depends on a valid result dict."""
    return getattr(inv, "name", "") in _RESULT_DEPENDENT_NAMES


class LenientIPU:
    """IPU that logs invariant violations without halting execution; for development/staging."""

    name = "LenientIPU"

    def __init__(self, invariants: list[Invariant]) -> None:
        self.invariants = invariants

    async def before_node(self, br: Principal, node: OpNode) -> None:
        for inv in self.invariants:
            if not inv.pre(br, node):
                logger.warning("[IPU][%s] pre-violation at node %s", inv.name, node.id)

    async def after_node(
        self,
        br: Principal,
        node: OpNode,
        result: dict[str, Any],
        *,
        error: BaseException | None = None,
    ) -> None:
        for inv in self.invariants:
            if error is not None and _is_result_dependent(inv):
                continue
            if not inv.post(br, node, result):
                logger.warning("[IPU][%s] post-violation at node %s", inv.name, node.id)

    async def on_observation(self, obs: Observation) -> None:
        logger.debug("[OBS] %s from %s: %s", obs.what, obs.who, obs.payload)


class StrictIPU(LenientIPU):
    """IPU that raises AssertionError on any invariant violation; for production/tests."""

    name = "StrictIPU"

    async def before_node(self, br: Principal, node: OpNode) -> None:
        for inv in self.invariants:
            if not inv.pre(br, node):
                raise AssertionError(
                    f"Invariant failed (pre): {inv.name} at node {node.id} "
                    f"in principal {br.id}"
                )

    async def after_node(
        self,
        br: Principal,
        node: OpNode,
        result: dict[str, Any],
        *,
        error: BaseException | None = None,
    ) -> None:
        for inv in self.invariants:
            if error is not None and _is_result_dependent(inv):
                continue
            if not inv.post(br, node, result):
                raise AssertionError(
                    f"Invariant failed (post): {inv.name} at node {node.id} "
                    f"in principal {br.id}"
                )


def default_invariants() -> list[Invariant]:
    return [
        PolicyGatePresent(),
        LatencyBound(),
        ResultShape(),
    ]
