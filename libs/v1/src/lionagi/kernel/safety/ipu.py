"""IPU: Invariant Protection Unit - The trust mechanism.

Ocean: "mechanism we can TRUST"

IPU validates that morphisms preserve invariants, ensuring compositional trust.
This is what allows us to safely compose complex systems from simple parts.
"""

from typing import Any, Optional, Tuple
from uuid import UUID, uuid4
import time as _time

from ..foundation.contracts import Observer, Observable, Observation, Invariant, Morphism

Tri = tuple[bool, Optional[str]]  # (valid, reason) - ChatGPT's 3-valued logic


class IPU(Observer):
    """Invariant Protection Unit - ensures composition maintains trust.
    
    The IPU is THE trust mechanism. When IPU validates something,
    the system can trust it completely.
    
    Enhanced with ChatGPT's pipeline: deny-safe, cacheable, 3-valued.
    """
    
    def __init__(self, system_invariants: list[Invariant] = None):
        self.id = uuid4()
        self.system_invariants = system_invariants or []
        # Cache key should include invariant set/version in production
        self._cache: dict[Tuple[str, str], Tri] = {}
    
    async def observe(self, observable: Observable) -> Observation:
        """Validate observable under system invariants (morphism-agnostic fast path).

        NOTE: Full morphism validation happens in `validate_morphism` at composition points.
        Here we provide a deny-safe baseline with caching:
          - If any system invariant refers only to the object, it can validate (o,o).
          - Otherwise, result remains valid=True by default and specific morphisms
            must be validated when applied.
        """
        key = (type(observable).__name__, str(observable.id))
        if key in self._cache:
            valid, reason = self._cache[key]
        else:
            valid, reason = True, None
            for inv in self.system_invariants:
                try:
                    if not inv.check(observable, observable):
                        valid, reason = False, inv.describe()
                        break
                except Exception as e:
                    valid, reason = False, f"invariant-error:{e}"
                    break
            self._cache[key] = (valid, reason)

        return Observation(
            observable_id=observable.id,
            observer_id=self.id,
            timestamp=_time.time(),
            valid=valid,
            trust_level=1.0 if valid else 0.0,
            capabilities_requested=set(),
            capabilities_granted=set(),
            effects=[],
            reason=reason,
        )
    
    def can_observe(self, observable: Observable) -> bool:
        """Check if we have capability to observe"""
        # TODO: Implement capability checking
        return True
    
    def validate_morphism(self, morphism: Morphism, before: Any, after: Any) -> bool:
        """Validate that a morphism preserves all invariants"""
        for invariant in morphism.invariants:
            if not invariant.check(before, after):
                return False
        
        for invariant in self.system_invariants:
            if not invariant.check(before, after):
                return False
        
        return True
    
    def compute_trust_level(self, observable: Observable) -> float:
        """Compute trust level for an observable"""
        # TODO: Implement trust computation
        return 1.0
