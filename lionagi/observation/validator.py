# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Validator: Core validation engine with simple hooks integration.

Clean validation with optional hooks through lionagi's service hooks system.
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Dict, List, Optional

from lionagi._errors import ValidationError
from lionagi.ln.concurrency import CompletionStream, bounded_map
from lionagi.service.hooks import HookRegistry

from .form import Form
from .rule.base import Rule, get_rule
from .rulebook import RuleBook


@dataclass
class ValidationResult:
    """Result of validating a single field."""

    field: str
    value: Any
    original_value: Any
    errors: List[str] = dataclass_field(default_factory=list)
    applied_rules: List[str] = dataclass_field(default_factory=list)
    success: bool = True


class Validator:
    """Core validation engine with optional service hooks integration."""

    def __init__(
        self,
        rulebook: Optional[RuleBook] = None,
        max_concurrent: int = 10,
        default_fix: bool = False,
        hooks: Optional[HookRegistry] = None,
    ):
        """Initialize validator.

        Args:
            rulebook: RuleBook with validation rules
            max_concurrent: Maximum concurrent validations
            default_fix: Whether to fix values by default
            hooks: Optional service hooks registry for validation events
        """
        self.rulebook = rulebook or self._default_rulebook()
        self.default_fix = default_fix
        self.max_concurrent = max_concurrent
        self.hooks = hooks

    def _default_rulebook(self) -> RuleBook:
        """Create default rulebook with registered rules."""
        from .rule._default import DEFAULT_RULES

        rules = {}
        for rule_enum in DEFAULT_RULES:
            rule_name = rule_enum.name.lower()
            rules[rule_name] = rule_enum.value

        return RuleBook(rules=rules)

    async def validate(
        self,
        field: str,
        value: Any,
        annotation: Optional[str] = None,
        fix: Optional[bool] = None,
        **kwargs,
    ) -> ValidationResult:
        """Validate a single field value.

        Args:
            field: Field name
            value: Value to validate
            annotation: Type annotation hint
            fix: Whether to fix invalid values
            **kwargs: Additional validation options

        Returns:
            ValidationResult with validated/fixed value
        """
        fix = fix if fix is not None else self.default_fix
        result = ValidationResult(
            field=field, value=value, original_value=value
        )

        # Simple hook: before validation
        if self.hooks:
            await self._call_hook("validation.pre", field, value, annotation)

        # Find applicable rule
        rule = self._find_rule(field, value, annotation, **kwargs)
        if not rule:
            # Simple hook: no rule found
            if self.hooks:
                await self._call_hook(
                    "validation.no_rule", field, value, annotation
                )
            return result

        try:
            # Apply validation
            if asyncio.iscoroutinefunction(rule.validate):
                validated_value = await rule.validate(value, **kwargs)
            else:
                validated_value = rule.validate(value, **kwargs)

            result.value = validated_value
            result.applied_rules.append(rule.__class__.__name__)

            # Simple hook: validation succeeded
            if self.hooks:
                await self._call_hook(
                    "validation.success",
                    field,
                    validated_value,
                    rule.__class__.__name__,
                )

        except ValidationError as e:
            result.errors.append(str(e))
            result.success = False

            # Simple hook: validation failed
            if self.hooks:
                await self._call_hook("validation.error", field, value, str(e))

            if fix and hasattr(rule, "fix"):
                try:
                    # Try to fix the value
                    if asyncio.iscoroutinefunction(rule.fix):
                        fixed_value = await rule.fix(value, **kwargs)
                    else:
                        fixed_value = rule.fix(value, **kwargs)

                    result.value = fixed_value
                    result.success = True
                    result.errors.append(f"Fixed: {str(e)}")

                    # Simple hook: value fixed
                    if self.hooks:
                        await self._call_hook(
                            "validation.fixed", field, fixed_value, str(e)
                        )

                except Exception as fix_error:
                    result.errors.append(f"Fix failed: {str(fix_error)}")

        except Exception as e:
            result.errors.append(f"Unexpected error: {str(e)}")
            result.success = False

            # Simple hook: unexpected error
            if self.hooks:
                await self._call_hook(
                    "validation.unexpected_error", field, value, str(e)
                )

        # Simple hook: after validation
        if self.hooks:
            await self._call_hook(
                "validation.post", field, result.value, result.success
            )

        return result

    async def _call_hook(self, hook_type: str, *args) -> None:
        """Call a validation hook if registered.

        Args:
            hook_type: Type of hook to call
            *args: Arguments to pass to hook
        """
        if not self.hooks:
            return

        try:
            # Use stream handler approach
            if (
                hasattr(self.hooks, "_stream_handlers")
                and hook_type in self.hooks._stream_handlers
            ):
                handler = self.hooks._stream_handlers[hook_type]
                await handler(*args)
        except Exception as e:
            # Don't let hook errors break validation
            print(f"Hook {hook_type} failed: {e}")

    def _find_rule(
        self,
        field: str,
        value: Any,
        annotation: Optional[str] = None,
        **kwargs,
    ) -> Optional[Rule]:
        """Find applicable rule for a field."""
        # First try annotation-based lookup
        if annotation:
            rule_class = get_rule(annotation)
            if rule_class:
                return rule_class()

            if annotation in self.rulebook:
                return self.rulebook.get_rule(annotation)

        # Fall back to rulebook search
        for rule_name in self.rulebook.rule_names:
            rule = self.rulebook.get_rule(rule_name)
            if rule:
                if not asyncio.iscoroutinefunction(rule.applies):
                    if rule.applies(field, value, None, annotation, **kwargs):
                        return rule

        return None

    async def validate_form(
        self, form: Form, fix: Optional[bool] = None, **kwargs
    ) -> Dict[str, ValidationResult]:
        """Validate all fields in a form."""
        fix = fix if fix is not None else self.default_fix
        results = {}

        # Simple hook: form validation starting
        if self.hooks:
            await self._call_hook("validation.form.start", form)

        # Create validation coroutines
        validation_coros = []
        field_names = []

        for field_name, (field_model, field_value) in form.fields.items():
            field_kwargs = {**kwargs, "field_model": field_model, "form": form}
            annotation = None
            if hasattr(field_model, "base_type"):
                annotation = field_model.base_type.__name__.lower()

            coro = self.validate(
                field_name,
                field_value,
                annotation=annotation,
                fix=fix,
                **field_kwargs,
            )
            validation_coros.append(coro)
            field_names.append(field_name)

        # Execute with rate limiting
        if self.max_concurrent and len(validation_coros) > 1:
            async with CompletionStream(
                validation_coros, limit=self.max_concurrent
            ) as stream:
                async for idx, result in stream:
                    results[field_names[idx]] = result
        else:
            for i, coro in enumerate(validation_coros):
                results[field_names[i]] = await coro

        # Update form with validated values
        for field_name, result in results.items():
            if result.success and result.value != result.original_value:
                field_model, _ = form.fields[field_name]
                form.fields[field_name] = (field_model, result.value)

        # Simple hook: form validation completed
        if self.hooks:
            success = all(r.success for r in results.values())
            await self._call_hook(
                "validation.form.complete", form, results, success
            )

        return results

    async def validate_dict(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, str]] = None,
        fix: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, ValidationResult]:
        """Validate a dictionary of values."""
        fix = fix if fix is not None else self.default_fix

        async def validate_field(
            item: tuple[str, Any],
        ) -> tuple[str, ValidationResult]:
            field_name, value = item
            annotation = schema.get(field_name) if schema else None
            result = await self.validate(
                field_name, value, annotation=annotation, fix=fix, **kwargs
            )
            return field_name, result

        if self.max_concurrent and len(data) > 1:
            items = list(data.items())
            validated_items = await bounded_map(
                validate_field, items, limit=self.max_concurrent
            )
            results = dict(validated_items)
        else:
            results = {}
            for field_name, value in data.items():
                annotation = schema.get(field_name) if schema else None
                results[field_name] = await self.validate(
                    field_name, value, annotation=annotation, fix=fix, **kwargs
                )

        return results

    def get_validated_dict(
        self, results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """Extract validated values from results."""
        return {
            name: result.value
            for name, result in results.items()
            if result.success
        }

    def get_errors(
        self, results: Dict[str, ValidationResult]
    ) -> Dict[str, List[str]]:
        """Extract errors from results."""
        return {
            name: result.errors
            for name, result in results.items()
            if result.errors
        }

    def __repr__(self) -> str:
        """String representation."""
        hooks_info = f"hooks={'yes' if self.hooks else 'no'}"
        return f"Validator(rulebook={self.rulebook}, {hooks_info})"
