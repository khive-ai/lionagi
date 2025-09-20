# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
IPU: Invariant Preservation Unit.

The IPU orchestrates validation while preserving mathematical invariants.
It checks invariants before and after operations, delegating actual
validation to backends.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Set

if TYPE_CHECKING:
    from .backend import ValidationResult
    from .form import Form
    from .invariant import Invariant


class IPUMode(Enum):
    """IPU enforcement modes."""

    LENIENT = "lenient"  # Log violations but continue
    STRICT = "strict"  # Fail on violations


@dataclass
class FrameAnalysis:
    """
    Frame analysis for invariant optimization.

    Following opinion 2: Track which fields affect which invariants
    to avoid checking all invariants on every change.
    """

    affected_fields: Set[str]
    relevant_invariants: Set[str]

    @classmethod
    def analyze(cls, form: Form, operation: str) -> "FrameAnalysis":
        """Analyze which fields and invariants are affected."""
        # For now, return all - will optimize later
        return cls(
            affected_fields=set(form.fields.keys()),
            relevant_invariants={
                inv.name
                for inv in form.field_specs.values()
                if hasattr(inv, "invariants")
                for inv in inv.invariants
            },
        )


class IPU:
    """
    Invariant Preservation Unit.

    Orchestrates validation while ensuring invariants hold.
    The IPU is the guardian of system consistency.
    """

    def __init__(
        self,
        backend: ValidationBackend,
        invariants: List[Invariant] = None,
        mode: IPUMode = IPUMode.STRICT,
    ):
        """
        Initialize IPU.

        Args:
            backend: Validation backend to delegate to
            invariants: List of invariants to preserve
            mode: Enforcement mode (strict or lenient)
        """
        self.backend = backend
        self.invariants = invariants or []
        self.mode = mode

    async def validate(self, form: Form) -> Form:
        """
        Validate form while preserving invariants.

        Following opinion 2: IPU is thick (orchestration, capabilities,
        invariants) while backend is thin (just execution).

        Args:
            form: Form to validate

        Returns:
            Validated form

        Raises:
            InvariantViolation: If invariant check fails (strict mode)
        """
        # Chain of Custody: Hash initial state
        initial_hash = self._hash_form(form)
        form.record_event(
            "ipu_entry", {"mode": self.mode.value, "state_hash": initial_hash}
        )

        # Frame analysis for optimization (opinion 2)
        frame = FrameAnalysis.analyze(form, "validate")

        # PRE: Check only relevant invariants
        for inv in self.invariants:
            if (
                inv.name in frame.relevant_invariants
                or self._is_global_invariant(inv)
            ):
                if not await inv.check_pre(form):
                    self._handle_violation(
                        f"Pre-invariant {inv.name} failed", form
                    )
                    if self.mode == IPUMode.STRICT:
                        raise InvariantViolation(f"Pre: {inv.name}")

        # Extract data for backend (FFI-optimized)
        data = form.fields.copy()
        rules = self._extract_rules(form)

        # VALIDATE: Delegate to backend with serialized data
        try:
            from .backend import ValidationBackend

            if isinstance(self.backend, ValidationBackend):
                result = await self.backend.execute_batch(data, rules)
                form.fields.update(result.fields)

                # Record validation events
                for field_name, errors in result.errors.items():
                    form.record_event(
                        "field_validation_failed",
                        {"field": field_name, "errors": errors},
                    )
                for field_name, warnings in result.warnings.items():
                    form.record_event(
                        "field_warning",
                        {"field": field_name, "warnings": warnings},
                    )
            else:
                # Legacy backend support
                form = await self.backend.validate(form)
        except Exception as e:
            form.record_event("validation_error", {"error": str(e)})
            raise

        # POST: Check only relevant invariants
        for inv in self.invariants:
            if (
                inv.name in frame.relevant_invariants
                or self._is_global_invariant(inv)
            ):
                if not await inv.check_post(form):
                    self._handle_violation(
                        f"Post-invariant {inv.name} failed", form
                    )
                    if self.mode == IPUMode.STRICT:
                        raise InvariantViolation(f"Post: {inv.name}")

        # Chain of Custody: Hash final state
        final_hash = self._hash_form(form)
        form.record_event(
            "ipu_exit", {"success": True, "state_hash": final_hash}
        )

        return form

    def _handle_violation(self, message: str, form: Form):
        """Handle invariant violation based on mode."""
        form.record_event("invariant_violation", {"message": message})
        if self.mode == IPUMode.LENIENT:
            print(f"[WARN] {message}")  # In production, use proper logging

    def _hash_form(self, form: Form) -> str:
        """Create hash of form state for Chain of Custody."""
        # Simple hash of fields for audit trail
        state = json.dumps(form.fields, sort_keys=True, default=str)
        return hashlib.sha256(state.encode()).hexdigest()[:16]

    def _extract_rules(self, form: Form) -> Dict[str, dict]:
        """Extract validation rules for backend."""
        rules = {}
        for field_name, spec in form.field_specs.items():
            rules[field_name] = {"type": spec.type, "default": spec.default}
        return rules

    def _is_global_invariant(self, inv) -> bool:
        """Check if invariant is global (affects all operations)."""
        # Global invariants that always need checking
        global_names = {"CapabilityMonotonicity", "FieldIntegrity"}
        return inv.name in global_names


class InvariantViolation(Exception):
    """Raised when an invariant check fails."""

    pass


class LenientIPU(IPU):
    """IPU that logs violations but doesn't fail."""

    def __init__(
        self, backend: ValidationBackend, invariants: List[Invariant] = None
    ):
        super().__init__(backend, invariants, IPUMode.LENIENT)


class StrictIPU(IPU):
    """IPU that fails on any violation."""

    def __init__(
        self, backend: ValidationBackend, invariants: List[Invariant] = None
    ):
        super().__init__(backend, invariants, IPUMode.STRICT)
