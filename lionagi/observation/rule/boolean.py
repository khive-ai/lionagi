# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lionagi._errors import ValidationError

from .base import Rule, register_rule


@register_rule("bool")
@register_rule("boolean")
class BooleanRule(Rule):
    """Rule for validating and converting boolean values."""

    # Common truthy/falsy string values
    TRUTHY_VALUES = {
        "true",
        "yes",
        "y",
        "on",
        "1",
        "enabled",
        "active",
        "ok",
        "correct",
    }
    FALSY_VALUES = {
        "false",
        "no",
        "n",
        "off",
        "0",
        "disabled",
        "inactive",
        "none",
        "incorrect",
    }

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to fields with bool annotation."""
        return annotation == "bool" if annotation else False

    async def validate(self, value: Any, **kwargs) -> bool:
        """Validate and convert value to boolean.

        Args:
            value: Value to validate
            **kwargs: Validation options

        Returns:
            Boolean value

        Raises:
            ValidationError: If cannot determine boolean value
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            if kwargs.get("default") is not None:
                return kwargs["default"]
            raise ValidationError("Boolean value cannot be None")

        # Already a boolean
        if isinstance(value, bool):
            return value

        # Numeric values
        if isinstance(value, (int, float)):
            return bool(value)

        # String values
        if isinstance(value, str):
            value_lower = value.strip().lower()

            if value_lower in self.TRUTHY_VALUES:
                return True
            if value_lower in self.FALSY_VALUES:
                return False

            # Check custom mappings
            custom_true = kwargs.get("custom_true", set())
            custom_false = kwargs.get("custom_false", set())

            if value_lower in custom_true:
                return True
            if value_lower in custom_false:
                return False

            # Strict mode - don't guess
            if kwargs.get("strict", True):
                raise ValidationError(
                    f"Cannot determine boolean value from '{value}'. "
                    f"Expected one of: {self.TRUTHY_VALUES | self.FALSY_VALUES}"
                )

            # Non-strict: any non-empty string is True
            return bool(value)

        # Other types - use Python truthiness
        return bool(value)

    async def fix(self, value: Any, **kwargs) -> bool:
        """Attempt to fix boolean value.

        Args:
            value: Value to fix
            **kwargs: Fix options

        Returns:
            Fixed boolean value
        """
        if value is None:
            return kwargs.get("default", False)

        # For any value, just use Python truthiness
        return bool(value)
