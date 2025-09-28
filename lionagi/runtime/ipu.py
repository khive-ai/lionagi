"""IPU - Invariant Processing Unit for morphism execution.

Adapted from v1 IPU with async-only design.
Observes morphism execution and enforces system invariants.
"""

from typing import Protocol, List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import time

if TYPE_CHECKING:
    from lionagi.session.branch import Branch
    from lionagi.morphism.base import Morphism


class Invariant(Protocol):
    """Protocol for invariants that must hold during morphism execution."""

    name: str

    async def pre(self, branch: "Branch", morphism: "Morphism") -> bool:
        """Check invariant before morphism execution."""
        ...

    async def post(
        self,
        branch: "Branch",
        morphism: "Morphism",
        result: Dict[str, Any]
    ) -> bool:
        """Check invariant after morphism execution."""
        ...


@dataclass
class BranchIsolation:
    """Ensure morphisms don't cross branch boundaries."""

    name: str = "BranchIsolation"

    async def pre(self, branch: "Branch", morphism: "Morphism") -> bool:
        # Store branch ID for verification
        if not hasattr(branch, '_ipu_branch_id'):
            branch._ipu_branch_id = branch.id
        return True

    async def post(
        self,
        branch: "Branch",
        morphism: "Morphism",
        result: Dict[str, Any]
    ) -> bool:
        # Verify same branch
        return getattr(branch, '_ipu_branch_id', None) == branch.id


@dataclass
class CapabilityMonotonicity:
    """Ensure capabilities don't expand without explicit grants."""

    name: str = "CapabilityMonotonicity"
    _pre_caps: Dict[str, set] = field(default_factory=dict)

    async def pre(self, branch: "Branch", morphism: "Morphism") -> bool:
        # Store current capabilities
        if hasattr(branch, 'capabilities'):
            self._pre_caps[branch.id] = set(branch.capabilities)
        return True

    async def post(
        self,
        branch: "Branch",
        morphism: "Morphism",
        result: Dict[str, Any]
    ) -> bool:
        if not hasattr(branch, 'capabilities'):
            return True

        pre_caps = self._pre_caps.get(branch.id, set())
        post_caps = set(branch.capabilities)

        # Allow capability reduction, but not expansion
        # (unless morphism is explicitly a grant operation)
        if morphism.name == "grant_capability":
            return True

        return post_caps.issubset(pre_caps) or post_caps == pre_caps


@dataclass
class LatencyBound:
    """Enforce latency bounds on morphism execution."""

    name: str = "LatencyBound"
    _start_times: Dict[tuple, float] = field(default_factory=dict)

    async def pre(self, branch: "Branch", morphism: "Morphism") -> bool:
        key = (branch.id, id(morphism))
        self._start_times[key] = time.perf_counter()
        return True

    async def post(
        self,
        branch: "Branch",
        morphism: "Morphism",
        result: Dict[str, Any]
    ) -> bool:
        # Check if morphism has latency budget
        budget_ms = getattr(morphism, 'latency_budget_ms', None)
        if budget_ms is None:
            return True

        key = (branch.id, id(morphism))
        start_time = self._start_times.pop(key, time.perf_counter())
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return elapsed_ms <= float(budget_ms)


@dataclass
class ResultShape:
    """Enforce result shape constraints."""

    name: str = "ResultShape"

    async def pre(self, branch: "Branch", morphism: "Morphism") -> bool:
        return True

    async def post(
        self,
        branch: "Branch",
        morphism: "Morphism",
        result: Dict[str, Any]
    ) -> bool:
        # Check required keys if specified
        required_keys = getattr(morphism, 'result_keys', None)
        if required_keys:
            if not isinstance(result, dict):
                return False
            missing = set(required_keys) - set(result.keys())
            if missing:
                return False

        return True


class IPU:
    """Invariant Processing Unit - observes and enforces invariants.

    The IPU doesn't execute morphisms - it observes their execution
    and ensures system invariants are maintained.
    """

    def __init__(
        self,
        invariants: Optional[List[Invariant]] = None,
        strict: bool = False,
        event_bus: Optional["EventBus"] = None
    ):
        """Initialize IPU.

        Args:
            invariants: List of invariants to enforce
            strict: If True, raise exceptions on violations
            event_bus: Optional event bus for service communication
        """
        self.invariants = invariants or self._default_invariants()
        self.strict = strict
        self.event_bus = event_bus

    def _default_invariants(self) -> List[Invariant]:
        """Get default system invariants."""
        return [
            BranchIsolation(),
            CapabilityMonotonicity(),
            LatencyBound(),
            ResultShape(),
        ]

    async def before_morphism(
        self,
        branch: "Branch",
        morphism: "Morphism"
    ) -> None:
        """Process pre-execution invariants.

        This is called before morphism execution.
        IPU can emit events to request services.
        """
        for invariant in self.invariants:
            ok = await invariant.pre(branch, morphism)
            if not ok:
                if self.strict:
                    raise AssertionError(
                        f"Pre-execution invariant {invariant.name} failed "
                        f"for morphism {morphism.name}"
                    )
                else:
                    # Emit warning event
                    if self.event_bus:
                        await self.event_bus.emit(
                            "invariant.violation",
                            {
                                "phase": "pre",
                                "invariant": invariant.name,
                                "morphism": morphism.name,
                                "branch": str(branch.id),
                            }
                        )

    async def after_morphism(
        self,
        branch: "Branch",
        morphism: "Morphism",
        result: Dict[str, Any]
    ) -> None:
        """Process post-execution invariants.

        This is called after morphism execution.
        IPU can validate results and emit events.
        """
        for invariant in self.invariants:
            ok = await invariant.post(branch, morphism, result)
            if not ok:
                if self.strict:
                    raise AssertionError(
                        f"Post-execution invariant {invariant.name} failed "
                        f"for morphism {morphism.name}"
                    )
                else:
                    # Emit warning event
                    if self.event_bus:
                        await self.event_bus.emit(
                            "invariant.violation",
                            {
                                "phase": "post",
                                "invariant": invariant.name,
                                "morphism": morphism.name,
                                "branch": str(branch.id),
                                "result_keys": list(result.keys()),
                            }
                        )

    async def request_service(
        self,
        service: str,
        payload: Dict[str, Any]
    ) -> Optional[Any]:
        """Request a service through the event bus.

        This is how IPU communicates with capability providers.

        Args:
            service: Service identifier (e.g., "validation.execute")
            payload: Request payload

        Returns:
            Service response if available
        """
        if not self.event_bus:
            return None

        return await self.event_bus.request(service, payload)


# Specialized IPUs
class LenientIPU(IPU):
    """IPU that logs violations but doesn't stop execution."""

    def __init__(self, invariants: Optional[List[Invariant]] = None, **kwargs):
        super().__init__(invariants=invariants, strict=False, **kwargs)


class StrictIPU(IPU):
    """IPU that raises exceptions on invariant violations."""

    def __init__(self, invariants: Optional[List[Invariant]] = None, **kwargs):
        super().__init__(invariants=invariants, strict=True, **kwargs)