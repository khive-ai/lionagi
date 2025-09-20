# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence

from lionagi._errors import ValidationError
from lionagi.ln.fuzzy import fuzzy_match_keys

from .base import Rule, register_rule


@register_rule("choice")
class ChoiceRule(Rule):
    """Rule for validating values against a set of choices."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to fields with choice annotation or choices constraint."""
        if annotation == "choice":
            return True

        # Check if form field has choices defined
        if hasattr(form, "fields") and field in form.fields:
            field_model = form.fields[field]
            if hasattr(field_model, "choices") and field_model.choices:
                return True

        return False

    async def validate(
        self, value: Any, choices: Sequence[Any] = None, **kwargs
    ) -> Any:
        """Validate value is in allowed choices.

        Args:
            value: Value to validate
            choices: Allowed choices (can be passed or from field metadata)
            **kwargs: Validation options (case_sensitive, fuzzy_match, threshold)

        Returns:
            Validated value (possibly normalized)

        Raises:
            ValidationError: If value not in choices
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            raise ValidationError("Choice value cannot be None")

        if not choices:
            choices = kwargs.get("choices", [])

        if not choices:
            # No choices defined - pass through
            return value

        # Exact match first
        if value in choices:
            return value

        # Case-insensitive match for strings
        if isinstance(value, str) and not kwargs.get("case_sensitive", True):
            value_lower = value.lower()
            for choice in choices:
                if isinstance(choice, str) and choice.lower() == value_lower:
                    return choice  # Return the canonical choice

        # Fuzzy matching if enabled
        if kwargs.get("fuzzy_match", False) and isinstance(value, str):
            str_choices = [str(c) for c in choices]
            threshold = kwargs.get("threshold", 0.8)

            matches = fuzzy_match_keys(value, str_choices, threshold=threshold)
            if matches:
                # Return the original choice (not stringified)
                best_match = matches[0]
                idx = str_choices.index(best_match)
                return choices[idx]

        # Value not in choices
        raise ValidationError(
            f"Invalid choice '{value}'. Must be one of: {list(choices)}"
        )

    async def fix(
        self, value: Any, choices: Sequence[Any] = None, **kwargs
    ) -> Any:
        """Attempt to fix choice value.

        Args:
            value: Value to fix
            choices: Allowed choices
            **kwargs: Fix options

        Returns:
            Fixed choice value
        """
        if value is None:
            # Use default if provided
            default = kwargs.get("default")
            if default is not None:
                return default
            # Return first choice if available
            if choices:
                return choices[0]
            raise ValidationError("Cannot fix: no choices available")

        if not choices:
            choices = kwargs.get("choices", [])

        if not choices:
            # No choices - can't fix
            return value

        # Try fuzzy matching with low threshold
        if isinstance(value, str):
            str_choices = [str(c) for c in choices]
            matches = fuzzy_match_keys(value, str_choices, threshold=0.5)
            if matches:
                idx = str_choices.index(matches[0])
                return choices[idx]

        # Return default or first choice
        default = kwargs.get("default")
        if default is not None and default in choices:
            return default

        return choices[0] if choices else value
