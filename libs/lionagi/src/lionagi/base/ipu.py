from __future__ import annotations

import copy
import logging
import time
from typing import Any, Protocol

import msgspec

from lionagi.ln.concurrency import is_coro_func

from .graph import OpNode
from .policy import policy_check
from .types import Branch, Observation

# Set up logger for IPU
logger = logging.getLogger(__name__)


class Invariant(Protocol):
    """Polyglot invariant protocol - supports both sync and async methods."""

    name: str

    def pre(self, br: Branch, node: OpNode) -> bool: ...
    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool: ...

    # Optional async versions - invariants can implement either sync or async
    async def async_pre(self, br: Branch, node: OpNode) -> bool: ...
    async def async_post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool: ...


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
    Monitor per-node latency if morphism defines: `latency_budget_ms: int`.

    Note: The actual proactive enforcement is done by Runner using fail_after.
    This invariant provides monitoring and post-execution validation.
    """

    name = "LatencyBound"

    def __init__(self):
        self._t0: dict[tuple, float] = {}

    def pre(self, br: Branch, node: OpNode) -> bool:
        self._t0[(br.id, node.id)] = time.perf_counter()
        budget = getattr(node.m, "latency_budget_ms", None)
        if budget and budget <= 0:
            # Invalid budget configuration
            return False
        return True

    def post(self, br: Branch, node: OpNode, result: dict[str, Any]) -> bool:
        budget = getattr(node.m, "latency_budget_ms", None)
        if budget is None:
            self._t0.pop((br.id, node.id), None)
            return True
        t0 = self._t0.pop((br.id, node.id), time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Log if we're close to budget (within 90%)
        if elapsed_ms > budget * 0.9:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Node {node.id} took {elapsed_ms:.1f}ms "
                f"(budget: {budget}ms, {elapsed_ms / budget * 100:.0f}% used)"
            )

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


# ---- Exception Classes ----


class InvariantViolationError(Exception):
    """Raised when an invariant check fails in strict mode."""

    pass


# ---- Adapter Classes for TDD Interface ----


class CtxWriteSetInvariant:
    """Adapter class that wraps CtxWriteSet with the expected test interface."""

    def __init__(self):
        self._impl = CtxWriteSet()
        self._snapshots = {}

    def pre_check(self, branch, operation_context):
        """Take snapshot and return (success, message) tuple."""
        try:
            # Create a mock node for the existing interface
            mock_node = type(
                "MockNode",
                (),
                {
                    "id": operation_context.get("node_id", "test_node"),
                    "m": type(
                        "MockMorphism",
                        (),
                        {"ctx_writes": operation_context.get("ctx_writes")},
                    )(),
                },
            )()

            # Take snapshot using existing implementation
            success = self._impl.pre(branch, mock_node)
            return success, ("Snapshot taken successfully" if success else "Snapshot failed")
        except Exception as e:
            return False, str(e)

    def post_check(self, branch, operation_context, result):
        """Check constraints and return (success, message) tuple."""
        try:
            # Create a mock node for the existing interface
            mock_node = type(
                "MockNode",
                (),
                {
                    "id": operation_context.get("node_id", "test_node"),
                    "m": type(
                        "MockMorphism",
                        (),
                        {"ctx_writes": operation_context.get("ctx_writes")},
                    )(),
                },
            )()

            # Get the snapshot key for this branch/node combination
            snapshot_key = (branch.id, mock_node.id)
            before_ctx = self._impl._snap.get(snapshot_key, {})

            # Check using existing implementation
            success = self._impl.post(branch, mock_node, result)
            if success:
                return True, "Context writes within allowed set"
            else:
                # Provide more detailed error message about which keys were written
                allowed = set(operation_context.get("ctx_writes", []))
                current_keys = set(branch.ctx.keys())
                before_keys = set(before_ctx.keys())

                # Find new keys and modified keys
                added_keys = current_keys - before_keys
                modified_keys = {
                    k
                    for k in (current_keys & before_keys)
                    if branch.ctx.get(k) != before_ctx.get(k)
                }
                changed_keys = added_keys | modified_keys
                undeclared_keys = changed_keys - allowed

                if undeclared_keys:
                    key_list = ", ".join(sorted(undeclared_keys))
                    return (
                        False,
                        f"Detected writes to undeclared context keys: {key_list}",
                    )
                else:
                    return False, "Detected writes to undeclared context keys"
        except Exception as e:
            return False, str(e)


class CapabilityMonotonicityInvariant:
    """Adapter class that wraps CapabilityMonotonicity with the expected test interface."""

    def __init__(self):
        self._snapshots = {}

    def pre_check(self, branch, operation_context):
        """Take capability snapshot and return (success, message) tuple."""
        try:
            # Take snapshot of current capabilities
            snapshot_key = (branch.id, operation_context.get("node_id", "test_node"))
            self._snapshots[snapshot_key] = set(branch.capabilities)
            return True, "Capability snapshot taken"
        except Exception as e:
            return False, str(e)

    def post_check(self, branch, operation_context, result):
        """Check for capability escalation and return (success, message) tuple."""
        try:
            # Get the snapshot
            snapshot_key = (branch.id, operation_context.get("node_id", "test_node"))
            pre_caps = self._snapshots.get(snapshot_key, set())
            post_caps = set(branch.capabilities)

            # Check if capabilities were reduced or stayed the same (monotonicity)
            # Escalation means post_caps has capabilities not in pre_caps
            escalated = post_caps - pre_caps

            if escalated:
                escalated_list = ", ".join(sorted(escalated))
                return (
                    False,
                    f"Capability escalation detected - privilege escalation forbidden: {escalated_list}",
                )
            else:
                return True, "No capability escalation detected"
        except Exception as e:
            return False, str(e)


class ResultShapeInvariant:
    """Adapter class that wraps ResultShape with the expected test interface."""

    def __init__(self, expected_schema=None):
        self._impl = ResultShape()
        self.expected_schema = expected_schema

    def pre_check(self, branch, operation_context):
        """ResultShape pre-check always passes."""
        return True, "ResultShape pre-check passed"

    def post_check(self, branch, operation_context, result):
        """Check result shape and return (success, message) tuple."""
        try:
            # Create a mock node with the expected schema
            mock_node = type(
                "MockNode",
                (),
                {
                    "id": operation_context.get("node_id", "test_node"),
                    "m": type("MockMorphism", (), {"result_schema": self.expected_schema})(),
                },
            )()

            success = self._impl.post(branch, mock_node, result)
            if success:
                return True, "Result shape validation passed"
            else:
                return (
                    False,
                    "Result shape validation failed - type mismatch or missing fields",
                )
        except Exception as e:
            return False, f"Result validation error: {str(e)}"


# ---- Implementations ----


class LenientIPU:
    name = "LenientIPU"

    def __init__(self, invariants):
        # Accept both Invariant protocol objects and test-style invariants
        self.invariants = invariants

    async def _call_invariant_pre(self, inv, br: Branch, node: OpNode) -> bool:
        """Normalize sync/async calls for invariant pre checks."""
        # Check if invariant has async_pre method and it's coroutine
        if hasattr(inv, "async_pre") and is_coro_func(inv.async_pre):
            return await inv.async_pre(br, node)
        # Check if pre method is async (backwards compatibility)
        elif hasattr(inv, "pre") and is_coro_func(inv.pre):
            return await inv.pre(br, node)
        # Default to sync pre method
        elif hasattr(inv, "pre"):
            return inv.pre(br, node)
        else:
            logger.warning(f"Invariant {getattr(inv, 'name', 'Unknown')} has no pre method")
            return True

    async def _call_invariant_post(
        self, inv, br: Branch, node: OpNode, result: dict[str, Any]
    ) -> bool:
        """Normalize sync/async calls for invariant post checks."""
        # Check if invariant has async_post method and it's coroutine
        if hasattr(inv, "async_post") and is_coro_func(inv.async_post):
            return await inv.async_post(br, node, result)
        # Check if post method is async (backwards compatibility)
        elif hasattr(inv, "post") and is_coro_func(inv.post):
            return await inv.post(br, node, result)
        # Default to sync post method
        elif hasattr(inv, "post"):
            return inv.post(br, node, result)
        else:
            logger.warning(f"Invariant {getattr(inv, 'name', 'Unknown')} has no post method")
            return True

    async def before_node(self, br: Branch, node: OpNode) -> None:
        for inv in self.invariants:
            ok = await self._call_invariant_pre(inv, br, node)
            if not ok:
                # Log only in lenient mode
                logger.warning(f"[WARN][{inv.name}] pre-phase violation at node {node.id}")

    async def after_node(self, br: Branch, node: OpNode, result: dict[str, Any]) -> None:
        for inv in self.invariants:
            ok = await self._call_invariant_post(inv, br, node, result)
            if not ok:
                logger.warning(f"[WARN][{inv.name}] post-phase violation at node {node.id}")

    def enter_node(self, branch: Branch, operation_context: dict) -> None:
        """TDD interface method for pre-node checks."""
        for inv in self.invariants:
            try:
                # Check if invariant has pre_check method (test interface) or pre method (protocol interface)
                if hasattr(inv, "pre_check"):
                    valid, message = inv.pre_check(branch, operation_context)
                    if not valid:
                        logger.warning(
                            f"[{getattr(inv, '__class__', {}).get('__name__', 'Unknown')}] pre-check violation: {message}"
                        )
                elif hasattr(inv, "pre"):
                    # Create mock node for protocol interface
                    mock_node = type(
                        "MockNode",
                        (),
                        {"id": operation_context.get("node_id", "test_node")},
                    )()
                    ok = inv.pre(branch, mock_node)
                    if not ok:
                        logger.warning(
                            f"[{getattr(inv, 'name', 'Unknown')}] pre-phase violation at node {mock_node.id}"
                        )
            except Exception as e:
                logger.warning(f"Invariant check error: {e}")

    def exit_node(self, branch: Branch, operation_context: dict, result: dict) -> None:
        """TDD interface method for post-node checks."""
        for inv in self.invariants:
            try:
                # Check if invariant has post_check method (test interface) or post method (protocol interface)
                if hasattr(inv, "post_check"):
                    valid, message = inv.post_check(branch, operation_context, result)
                    if not valid:
                        logger.warning(
                            f"[{getattr(inv, '__class__', {}).get('__name__', 'Unknown')}] post-check violation: {message}"
                        )
                elif hasattr(inv, "post"):
                    # Create mock node for protocol interface
                    mock_node = type(
                        "MockNode",
                        (),
                        {"id": operation_context.get("node_id", "test_node")},
                    )()
                    ok = inv.post(branch, mock_node, result)
                    if not ok:
                        logger.warning(
                            f"[{getattr(inv, 'name', 'Unknown')}] post-phase violation at node {mock_node.id}"
                        )
            except Exception as e:
                logger.warning(f"Invariant check error: {e}")

    async def on_observation(self, obs: Observation) -> None:
        pass


class StrictIPU(LenientIPU):
    name = "StrictIPU"

    async def before_node(self, br: Branch, node: OpNode) -> None:
        for inv in self.invariants:
            ok = await self._call_invariant_pre(inv, br, node)
            if not ok:
                raise AssertionError(f"Invariant failed (pre): {inv.name} at node {node.id}")

    async def after_node(self, br: Branch, node: OpNode, result: dict[str, Any]) -> None:
        for inv in self.invariants:
            ok = await self._call_invariant_post(inv, br, node, result)
            if not ok:
                raise AssertionError(f"Invariant failed (post): {inv.name} at node {node.id}")

    def enter_node(self, branch: Branch, operation_context: dict) -> None:
        """TDD interface method for pre-node checks - raises InvariantViolationError on failure."""
        for inv in self.invariants:
            try:
                # Check if invariant has pre_check method (test interface) or pre method (protocol interface)
                if hasattr(inv, "pre_check"):
                    valid, message = inv.pre_check(branch, operation_context)
                    if not valid:
                        raise InvariantViolationError(message)
                elif hasattr(inv, "pre"):
                    # Create mock node for protocol interface
                    mock_node = type(
                        "MockNode",
                        (),
                        {"id": operation_context.get("node_id", "test_node")},
                    )()
                    ok = inv.pre(branch, mock_node)
                    if not ok:
                        raise InvariantViolationError(
                            f"Invariant failed (pre): {getattr(inv, 'name', 'Unknown')} at node {mock_node.id}"
                        )
            except InvariantViolationError:
                raise
            except Exception as e:
                raise InvariantViolationError(f"Invariant check error: {e}")

    def exit_node(self, branch: Branch, operation_context: dict, result: dict) -> None:
        """TDD interface method for post-node checks - raises InvariantViolationError on failure."""
        for inv in self.invariants:
            try:
                # Check if invariant has post_check method (test interface) or post method (protocol interface)
                if hasattr(inv, "post_check"):
                    valid, message = inv.post_check(branch, operation_context, result)
                    if not valid:
                        raise InvariantViolationError(message)
                elif hasattr(inv, "post"):
                    # Create mock node for protocol interface
                    mock_node = type(
                        "MockNode",
                        (),
                        {"id": operation_context.get("node_id", "test_node")},
                    )()
                    ok = inv.post(branch, mock_node, result)
                    if not ok:
                        raise InvariantViolationError(
                            f"Invariant failed (post): {getattr(inv, 'name', 'Unknown')} at node {mock_node.id}"
                        )
            except InvariantViolationError:
                raise
            except Exception as e:
                raise InvariantViolationError(f"Invariant check error: {e}")


def default_invariants() -> list[Invariant]:
    return [
        BranchIsolation(),
        CapabilityMonotonicity(),
        DeterministicLineage(),
        ObservationCompleteness(),
        PolicyGatePresent(),
        LatencyBound(),
        ResultShape(),
        CtxWriteSet(),
        NoAmbientAuthority(),
        ResultSizeBound(),
    ]
