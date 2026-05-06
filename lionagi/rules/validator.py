# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validator: orchestrates rule lookup, defaulting, and custom validators per Spec.

The validator owns three concerns: pick a Rule for each Spec, drive the rule
to coerce / validate the value, then run any user-supplied validators in
declaration order. A bounded deque records recent failures for diagnostics.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from lionagi.ln.concurrency import is_coro_func
from lionagi.ln.types import Unset, is_sentinel, not_sentinel

from .registry import RuleRegistry, get_default_registry
from .rule import Rule, ValidationError

if TYPE_CHECKING:
    from pydantic import BaseModel

    from lionagi.ln.types import Operable, Spec

__all__ = ("Validator",)


class Validator:
    DEFAULT_MAX_LOG_ENTRIES: ClassVar[int] = 1000

    def __init__(
        self,
        registry: RuleRegistry | None = None,
        *,
        max_log_entries: int | None = None,
    ) -> None:
        self.registry = registry or get_default_registry()
        max_entries = (
            max_log_entries
            if max_log_entries is not None
            else self.DEFAULT_MAX_LOG_ENTRIES
        )
        self.validation_log: deque[dict[str, Any]] = deque(
            maxlen=max_entries if max_entries > 0 else None
        )

    # ------------------------------------------------------------------
    # Diagnostic log
    # ------------------------------------------------------------------

    def log_validation_error(self, field: str, value: Any, error: str) -> None:
        self.validation_log.append(
            {
                "field": field,
                "value": value,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_validation_summary(self) -> dict[str, Any]:
        fields = sorted({e["field"] for e in self.validation_log if "field" in e})
        return {
            "total_errors": len(self.validation_log),
            "fields_with_errors": fields,
            "error_entries": list(self.validation_log),
        }

    def clear_log(self) -> None:
        self.validation_log.clear()

    # ------------------------------------------------------------------
    # Rule lookup
    # ------------------------------------------------------------------

    def get_rule_for_spec(self, spec: Spec) -> Rule | None:
        override = spec.get("rule")
        if isinstance(override, Rule):
            return override
        return self.registry.get_rule(
            base_type=spec.base_type,
            field_name=spec.name if spec.name else None,
        )

    # ------------------------------------------------------------------
    # Per-spec validation
    # ------------------------------------------------------------------

    async def validate_spec(
        self,
        spec: Spec,
        value: Any,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> Any:
        field_name = spec.name or "<unnamed>"

        if value is None or value is Unset:
            value, short_circuit = await self._resolve_missing(
                spec, field_name, value, strict
            )
            if short_circuit:
                return value

        rule = self.get_rule_for_spec(spec)

        if spec.is_listable:
            value = await self._validate_list(
                rule, field_name, value, spec.base_type, auto_fix, strict
            )
        else:
            value = await self._validate_scalar(
                rule, field_name, value, spec.base_type, auto_fix, strict
            )

        return await self._apply_custom_validators(spec, field_name, value)

    async def _resolve_missing(
        self, spec: Spec, field_name: str, value: Any, strict: bool
    ) -> tuple[Any, bool]:
        """Handle None/Unset values — returns (value, short_circuit).

        short_circuit=True means skip rule and custom validators entirely.
        Three exits:
          - nullable + None: (None, True) — accept null as-is
          - default succeeds: (default, False) — keep validating
          - no default, strict: raise ValidationError
          - no default, not strict: (value, True) — pass-through
        """
        if value is None and spec.is_nullable:
            return None, True
        try:
            return await spec.acreate_default_value(), False
        except ValueError as e:
            if not strict:
                return value, True
            msg = (
                f"Field '{field_name}' is missing and has no default "
                f"(value={value!r}, nullable={spec.is_nullable})"
            )
            self.log_validation_error(field_name, value, msg)
            raise ValidationError(msg) from e

    async def _validate_list(
        self,
        rule: Rule | None,
        field_name: str,
        value: Any,
        base_type: type,
        auto_fix: bool,
        strict: bool,
    ) -> list[Any]:
        if not isinstance(value, list):
            if not auto_fix:
                msg = f"Field '{field_name}' expected list, got {type(value).__name__}"
                self.log_validation_error(field_name, value, msg)
                raise ValidationError(msg)
            value = [value]

        if rule is None:
            return list(value)

        validated: list[Any] = []
        for i, item in enumerate(value):
            item_name = f"{field_name}[{i}]"
            try:
                validated.append(
                    await rule.invoke(item_name, item, base_type, auto_fix=auto_fix)
                )
            except Exception as e:
                self.log_validation_error(item_name, item, str(e))
                raise
        return validated

    async def _validate_scalar(
        self,
        rule: Rule | None,
        field_name: str,
        value: Any,
        base_type: type,
        auto_fix: bool,
        strict: bool,
    ) -> Any:
        if rule is None:
            if not strict:
                return value
            msg = (
                f"No rule found for field '{field_name}' with type {base_type}. "
                f"Register a rule or set strict=False."
            )
            self.log_validation_error(field_name, value, msg)
            raise ValidationError(msg)

        try:
            return await rule.invoke(field_name, value, base_type, auto_fix=auto_fix)
        except Exception as e:
            self.log_validation_error(field_name, value, str(e))
            raise

    async def _apply_custom_validators(
        self, spec: Spec, field_name: str, value: Any
    ) -> Any:
        """Run user-supplied validators in declaration order; each may transform value."""
        validators = _coerce_validator_list(spec.get("validator"))
        for fn in validators:
            try:
                result = fn(value)
                value = await result if is_coro_func(fn) else result
            except Exception as e:
                msg = f"Custom validator failed for '{field_name}': {e}"
                self.log_validation_error(field_name, value, msg)
                raise ValidationError(msg) from e
        return value

    # ------------------------------------------------------------------
    # Whole-operable validation
    # ------------------------------------------------------------------

    async def validate(
        self,
        data: dict[str, Any],
        operable: Operable,
        capabilities: set[str] | None = None,
        auto_fix: bool = True,
        strict: bool = True,
        structure: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        if not_sentinel(capabilities, additions={"none"}) and not capabilities.issubset(
            operable.allowed()
        ):
            raise ValidationError("Capabilities exceed operable's allowed set")

        capabilities = capabilities or operable.allowed()
        validated: dict[str, Any] = {}

        for spec in operable.get_specs():
            field_name = spec.name
            if is_sentinel(field_name) or not isinstance(field_name, str):
                continue
            if field_name not in capabilities:
                continue
            validated[field_name] = await self.validate_spec(
                spec, data.get(field_name), auto_fix=auto_fix, strict=strict
            )

        if structure is not None:
            validated = operable.validate_instance(structure, validated)
        return validated


def _coerce_validator_list(raw: Any) -> list[Callable[..., Any]]:
    """Normalize the spec.validator field into a list of callables."""
    if is_sentinel(raw) or raw is None:
        return []
    if callable(raw):
        return [raw]
    if isinstance(raw, list):
        return [fn for fn in raw if callable(fn)]
    return []
