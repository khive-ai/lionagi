from __future__ import annotations

import copy
import time
from typing import Any, Protocol

import msgspec

from .graph import OpNode
from .policy import policy_check
from .types import Branch, Observation


class Invariant(Protocol):
    name: str

    def pre(self, br: Branch, node: OpNode) -> bool: ...
    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool: ...


class IPU(Protocol):
    name: str
    invariants: list[Invariant]

    async def before_node(self, br: Branch, node: OpNode) -> None: ...
    async def after_node(self, br: Branch, node: OpNode, result: dict[str, Any]) -> None: ...
    async def on_observation(self, obs: Observation) -> None: ...


# ---- Baseline invariants ----


class BranchIsolation:
    """Runner executes within a single Branch; cross-branch ops must be explicit morphisms."""

    name = "BranchIsolation"

    def pre(self, br: Branch, node: OpNode) -> bool:
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        return True


class CapabilityMonotonicity:
    """Capabilities shouldn't expand unless done via explicit grant morphisms."""

    name = "CapabilityMonotonicity"

    def __init__(self):
        self._pre_caps: set[str] = set()

    def pre(self, br: Branch, node: OpNode) -> bool:
        self._pre_caps = {r for c in br.caps for r in c.rights}
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        post_caps = {r for c in br.caps for r in c.rights}
        return post_caps.issubset(self._pre_caps) or post_caps == self._pre_caps


class DeterministicLineage:
    name = "DeterministicLineage"

    def pre(self, br: Branch, node: OpNode) -> bool:
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        return True


class ObservationCompleteness:
    name = "ObservationCompleteness"

    def __init__(self):
        self._started: set[tuple] = set()

    def pre(self, br: Branch, node: OpNode) -> bool:
        self._started.add((br.id, node.id))
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        return (br.id, node.id) in self._started


class PolicyGatePresent:
    name = "PolicyGatePresent"

    def pre(self, br: Branch, node: OpNode) -> bool:
        return policy_check(br, node.m)

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        return True


# ---- Advanced invariants (opt-in via morphism attributes) ----


class LatencyBound:
    """
    Enforce per-node latency if morphism defines: `latency_budget_ms: int`.
    """

    name = "LatencyBound"

    def __init__(self):
        self._t0: dict[tuple, float] = {}

    def pre(self, br: Branch, node: OpNode) -> bool:
        self._t0[(br.id, node.id)] = time.perf_counter()
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        budget = getattr(node.m, "latency_budget_ms", None)
        if budget is None:
            self._t0.pop((br.id, node.id), None)
            return True
        t0 = self._t0.pop((br.id, node.id), time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms <= float(budget)


class ResultShape:
    """
    Enforce result keys or schema if morphism defines either:
      - `result_keys: set[str]`  (all keys must be present in result dict)
      - `result_schema: Type[msgspec.Struct]` (strictly validate via msgspec decode)
    """

    name = "ResultShape"

    def pre(self, br: Branch, node: OpNode) -> bool:
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        # 1) Required keys
        required = getattr(node.m, "result_keys", None)
        if required:
            if not isinstance(result, dict):
                return False
            miss = set(required) - set(result.keys())
            if miss:
                return False

        # 2) Schema (strict)
        schema = getattr(node.m, "result_schema", None)
        if schema is not None and msgspec is not None:
            try:
                # strict validation by round-tripping through msgspec decoder
                payload = msgspec.json.encode(result)
                msgspec.json.decode(payload, type=schema)
            except Exception:
                return False

        return True


class CtxWriteSet:
    """
    Constrain writes to Branch.ctx if morphism defines:
      - `ctx_writes: set[str]` (only these keys may change/add)
    """

    name = "CtxWriteSet"

    def __init__(self):
        self._snap: dict[tuple, dict[str, Any]] = {}

    def pre(self, br: Branch, node: OpNode) -> bool:
        # Deepcopy since ctx values can be nested
        self._snap[(br.id, node.id)] = copy.deepcopy(br.ctx)
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        allowed = getattr(node.m, "ctx_writes", None)
        if not allowed:
            # If not declared, don't enforce (opt-in)
            self._snap.pop((br.id, node.id), None)
            return True
        before = self._snap.pop((br.id, node.id), {})
        after = br.ctx
        added = set(after.keys()) - set(before.keys())
        modified = {k for k in (set(after.keys()) & set(before.keys())) if after[k] != before[k]}
        changed = added | modified
        return changed.issubset(set(allowed))


class NoAmbientAuthority:
    """
    If a morphism declares `io=True`, it must also declare non-empty `requires`.
    """

    name = "NoAmbientAuthority"

    def pre(self, br: Branch, node: OpNode) -> bool:
        if getattr(node.m, "io", False):
            req = getattr(node.m, "requires", set())
            return bool(req)
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        return True


class ResultSizeBound:
    """
    Enforce serialized size bound if morphism defines:
      - `result_bytes_limit: int` (bytes, using msgspec.json.encode length)
    """

    name = "ResultSizeBound"

    def pre(self, br: Branch, node: OpNode) -> bool:
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        limit = getattr(node.m, "result_bytes_limit", None)
        if limit is None or msgspec is None:
            return True
        try:
            size = len(msgspec.json.encode(result))
        except Exception:
            # If not serializable, consider it a violation when a limit is requested
            return False
        return size <= int(limit)


# ---- Implementations ----


class LenientIPU:
    name = "LenientIPU"

    def __init__(self, invariants: list[Invariant]):
        self.invariants = invariants

    async def before_node(self, br: Branch, node: OpNode) -> None:
        for inv in self.invariants:
            ok = inv.pre(br, node)
            if not ok:
                # Log only in lenient mode
                print(f"[WARN][{inv.name}] pre-phase violation at node {node.id}")

    async def after_node(self, br: Branch, node: OpNode, result: dict[str, Any]) -> None:
        for inv in self.invariants:
            ok = inv.post(br, node, result)
            if not ok:
                print(f"[WARN][{inv.name}] post-phase violation at node {node.id}")

    async def on_observation(self, obs: Observation) -> None:
        pass


class StrictIPU(LenientIPU):
    name = "StrictIPU"

    async def before_node(self, br: Branch, node: OpNode) -> None:
        for inv in self.invariants:
            if not inv.pre(br, node):
                raise AssertionError(f"Invariant failed (pre): {inv.name} at node {node.id}")

    async def after_node(self, br: Branch, node: OpNode, result: dict[str, Any]) -> None:
        for inv in self.invariants:
            if not inv.post(br, node, result):
                raise AssertionError(f"Invariant failed (post): {inv.name} at node {node.id}")


def default_invariants() -> list[Invariant]:
    # Baselines + advanced (advanced enforce only when morphisms opt-in with attributes)
    return [
        BranchIsolation(),
        CapabilityMonotonicity(),
        DeterministicLineage(),
        ObservationCompleteness(),
        PolicyGatePresent(),
        # Advanced:
        LatencyBound(),
        ResultShape(),
        CtxWriteSet(),
        NoAmbientAuthority(),
        ResultSizeBound(),
    ]
