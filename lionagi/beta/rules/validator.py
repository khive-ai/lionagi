# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from lionagi.ln.concurrency import is_coro_func
from lionagi.ln.types import is_sentinel, not_sentinel

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
        rulebook: RuleRegistry | None = None,
        max_log_entries: int | None = None,
    ):
        if registry is not None and rulebook is not None and registry is not rulebook:
            raise ValueError("Pass either registry or rulebook, not both")
        self.registry = registry or rulebook or get_default_registry()
        max_entries = (
            max_log_entries if max_log_entries is not None else self.DEFAULT_MAX_LOG_ENTRIES
        )
        self.validation_log: deque[dict[str, Any]] = deque(
            maxlen=max_entries if max_entries > 0 else None
        )

    @property
    def rulebook(self) -> RuleRegistry:
        return self.registry

    def log_validation_error(self, field: str, value: Any, error: str) -> None:
        """Log a validation error with timestamp.

        Args:
            field: Field name that failed validation
            value: Value that failed validation
            error: Error message
        """
        log_entry = {
            "field": field,
            "value": value,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        self.validation_log.append(log_entry)

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of validation log.

        Returns:
            Dict with total_errors, fields_with_errors, and error_entries
        """
        fields_with_errors = set()
        for entry in self.validation_log:
            if "field" in entry:
                fields_with_errors.add(entry["field"])

        return {
            "total_errors": len(self.validation_log),
            "fields_with_errors": sorted(list(fields_with_errors)),
            "error_entries": list(self.validation_log),
        }

    def clear_log(self) -> None:
        """Clear the validation log."""
        self.validation_log.clear()

    def get_rule_for_spec(self, spec: Spec) -> Rule | None:
        override = spec.get("rule")
        if override is not None and isinstance(override, Rule):
            return override

        return self.registry.get_rule(
            base_type=spec.base_type,
            field_name=spec.name if spec.name else None,
        )

    async def validate_spec(
        self,
        spec: Spec,
        value: Any,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> Any:
        field_name = spec.name or "<unnamed>"

        # Treat Unset (LNDL sentinel for unprovided fields) the same as None —
        # use the field's default before invoking any type-specific rule.
        from lionagi.ln.types import Unset

        if value is None or value is Unset:
            if spec.is_nullable and value is None:
                return None
            try:
                value = await spec.acreate_default_value()
            except ValueError as e:
                if strict:
                    error_msg = (
                        f"Field '{field_name}' is missing and has no default "
                        f"(value={value!r}, nullable={spec.is_nullable})"
                    )
                    self.log_validation_error(field_name, value, error_msg)
                    raise ValidationError(error_msg) from e
                return value

        rule = self.get_rule_for_spec(spec)

        if spec.is_listable:
            if not isinstance(value, list):
                if auto_fix:
                    value = [value]
                else:
                    error_msg = f"Field '{field_name}' expected list, got {type(value).__name__}"
                    self.log_validation_error(field_name, value, error_msg)
                    raise ValidationError(error_msg)

            validated_items = []
            for i, item in enumerate(value):
                item_name = f"{field_name}[{i}]"
                if rule is not None:
                    try:
                        validated_item = await rule.invoke(
                            item_name, item, spec.base_type, auto_fix=auto_fix
                        )
                    except Exception as e:
                        self.log_validation_error(item_name, item, str(e))
                        raise
                else:
                    validated_item = item
                validated_items.append(validated_item)

            value = validated_items
        else:
            if rule is None:
                if strict:
                    error_msg = (
                        f"No rule found for field '{field_name}' with type {spec.base_type}. "
                        f"Register a rule or set strict=False."
                    )
                    self.log_validation_error(field_name, value, error_msg)
                    raise ValidationError(error_msg)
            else:
                try:
                    value = await rule.invoke(field_name, value, spec.base_type, auto_fix=auto_fix)
                except Exception as e:
                    self.log_validation_error(field_name, value, str(e))
                    raise

        custom_validators = spec.get("validator")
        if is_sentinel(custom_validators) or custom_validators is None:
            validators = []
        elif callable(custom_validators):
            validators = [custom_validators]
        elif isinstance(custom_validators, list):
            validators = custom_validators
        else:
            validators = []

        for validator_fn in validators:
            if not callable(validator_fn):
                continue
            try:
                if is_coro_func(validator_fn):
                    value = await validator_fn(value)
                else:
                    value = validator_fn(value)
            except Exception as e:
                error_msg = f"Custom validator failed for '{field_name}': {e}"
                self.log_validation_error(field_name, value, error_msg)
                raise ValidationError(error_msg) from e

        return value

    async def validate(
        self,
        data: dict[str, Any],
        operable: Operable,
        capabilities: set[str] | None = None,
        auto_fix: bool = True,
        strict: bool = True,
        structure: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        if not_sentinel(capabilities, additions={"none"}) and not capabilities.issubset(operable.allowed()):
            raise ValidationError("Capabilities exceed operable's allowed set")

        capabilities = capabilities or operable.allowed()
        validated: dict[str, Any] = {}

        for spec in operable.get_specs():
            field_name = spec.name
            if is_sentinel(field_name) or not isinstance(field_name, str):
                continue

            if field_name not in capabilities:
                continue

            value = data.get(field_name)
            validated[field_name] = await self.validate_spec(
                spec, value, auto_fix=auto_fix, strict=strict
            )

        if structure is not None:
            validated = operable.validate_instance(structure, validated)

        return validated
