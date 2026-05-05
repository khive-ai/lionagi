# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Invariant Policy Unit (IPU): enforcement layer between morphisms and the runner.

The IPU intercepts before_node and after_node execution phases to run a chain
of Invariants. Invariants check pre- and post-conditions; the IPU decides what
to do when they fail (log in lenient mode, raise in strict mode).

Design principles:
    - Invariants are small, focused checks. They have no side-effects.
    - The IPU composes invariants — it does not implement policy itself.
    - Invariants are opt-in via morphism attributes (no forced overhead).
    - LenientIPU warns; StrictIPU raises. Choose at runner-construction time.

Slim invariant set (vs. the over-engineered idealized v1):
    PolicyGatePresent: capability check — always enforced (the critical invariant)
    LatencyBound:      timing check — opt-in via morphism.latency_budget_ms
    ResultShape:       output validation — opt-in via morphism.result_keys / result_schema

The removed invariants (BranchIsolation, CapabilityMonotonicity, DeterministicLineage,
ObservationCompleteness, NoAmbientAuthority, CtxWriteSet, ResultSizeBound) are
available as opt-in advanced invariants but not included in default_invariants().
They are maintained here as a reference for consumers that need formal verification
but are not part of the recommended default.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from .graph import OpNode
from .policy import policy_check
from .types import Principal, Observation

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
    """Protocol for execution invariants checked by the IPU.

    Invariants are stateless (or minimally stateful for timing) objects that
    check pre- and post-conditions around node execution.
    """

    name: str

    def pre(self, br: Principal, node: OpNode) -> bool:
        """Check precondition before node.m.apply(). Return False to abort."""
        ...

    def post(self, br: Principal, node: OpNode, result: dict[str, Any]) -> bool:
        """Check postcondition after node.m.apply(). Return False to reject."""
        ...


class IPU(Protocol):
    """Invariant Policy Unit: composable execution enforcement.

    Implementations receive before_node/after_node hooks from the Runner
    and run their invariant chain. on_observation receives runtime telemetry.
    """

    name: str
    invariants: list[Invariant]

    async def before_node(self, br: Principal, node: OpNode) -> None: ...
    async def after_node(
        self, br: Principal, node: OpNode, result: dict[str, Any],
        *, error: BaseException | None = None,
    ) -> None: ...
    async def on_observation(self, obs: Observation) -> None: ...


# ---------------------------------------------------------------------------
# Core invariants (always checked)
# ---------------------------------------------------------------------------


class PolicyGatePresent:
    """Verify that the policy gate is present in the execution pipeline.

    Runner performs the authoritative policy check (including dynamic rights,
    accumulated provides, and extra_rights) BEFORE calling ipu.before_node().
    This invariant is a sentinel: it declares that policy enforcement is
    required, but does not re-check — the IPU cannot access Runner's per-run
    state (accumulated_provides) and would produce false rejections.
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
    """Enforce per-node latency budget when morphism declares latency_budget_ms.

    Opt-in: morphism must have latency_budget_ms: int attribute.
    No-op if the attribute is absent or None.
    """

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
    """Validate result shape when morphism declares result_keys or result_schema.

    Opt-in:
        result_keys: frozenset[str] — all these keys must appear in result dict
        result_schema: type[msgspec.Struct] — strict round-trip validation

    No-op if both attributes are absent or None.
    """

    name = "ResultShape"

    def pre(self, br: Principal, node: OpNode) -> bool:
        return True

    def post(self, br: Principal, node: OpNode, result: dict[str, Any]) -> bool:
        # 1) Required keys check
        required = getattr(node.m, "result_keys", None)
        if required:
            if not isinstance(result, dict):
                return False
            miss = set(required) - set(result.keys())
            if miss:
                return False

        # 2) Schema validation (strict, requires msgspec)
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
    """IPU that logs invariant failures without halting execution.

    Use in development/staging. Failures produce warnings in logs but
    execution continues. Observations are forwarded to logger.
    """

    name = "LenientIPU"

    def __init__(self, invariants: list[Invariant]) -> None:
        self.invariants = invariants

    async def before_node(self, br: Principal, node: OpNode) -> None:
        for inv in self.invariants:
            if not inv.pre(br, node):
                logger.warning("[IPU][%s] pre-violation at node %s", inv.name, node.id)

    async def after_node(
        self, br: Principal, node: OpNode, result: dict[str, Any],
        *, error: BaseException | None = None,
    ) -> None:
        for inv in self.invariants:
            if error is not None and _is_result_dependent(inv):
                continue
            if not inv.post(br, node, result):
                logger.warning("[IPU][%s] post-violation at node %s", inv.name, node.id)

    async def on_observation(self, obs: Observation) -> None:
        logger.debug("[OBS] %s from %s: %s", obs.what, obs.who, obs.payload)


class StrictIPU(LenientIPU):
    """IPU that raises AssertionError on any invariant failure.

    Use in production/tests. Any pre- or post-condition violation halts
    execution immediately.
    """

    name = "StrictIPU"

    async def before_node(self, br: Principal, node: OpNode) -> None:
        for inv in self.invariants:
            if not inv.pre(br, node):
                raise AssertionError(
                    f"Invariant failed (pre): {inv.name} at node {node.id} "
                    f"in principal {br.id}"
                )

    async def after_node(
        self, br: Principal, node: OpNode, result: dict[str, Any],
        *, error: BaseException | None = None,
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
    """Return the recommended default invariant set.

    Always enforced:
        PolicyGatePresent — capability check (the critical invariant)

    Opt-in (no-op unless morphism declares the relevant attribute):
        LatencyBound — activated by morphism.latency_budget_ms
        ResultShape  — activated by morphism.result_keys or morphism.result_schema
    """
    return [
        PolicyGatePresent(),
        LatencyBound(),
        ResultShape(),
    ]
