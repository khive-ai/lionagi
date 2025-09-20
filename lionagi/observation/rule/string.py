# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lionagi._errors import ValidationError

from .base import Rule, register_rule


@register_rule("str")
@register_rule("string")
@register_rule("text")
class StringRule(Rule):
    """Rule for validating and converting values to strings."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to fields with str annotation or any value."""
        return annotation == "str" if annotation else True

    async def validate(self, value: Any, **kwargs) -> str:
        """Validate and convert value to string.

        Args:
            value: Value to validate
            **kwargs: Additional options (strip, lower, upper)

        Returns:
            String value

        Raises:
            ValidationError: If conversion fails
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            raise ValidationError("Value cannot be None")

        try:
            # Convert to string
            if isinstance(value, str):
                result = value
            elif isinstance(value, bytes):
                result = value.decode("utf-8")
            else:
                result = str(value)

            # Apply string transformations
            if kwargs.get("strip", False):
                result = result.strip()
            if kwargs.get("lower", False):
                result = result.lower()
            if kwargs.get("upper", False):
                result = result.upper()

            # Check length constraints
            min_length = kwargs.get("min_length")
            max_length = kwargs.get("max_length")

            if min_length is not None and len(result) < min_length:
                raise ValidationError(
                    f"String must be at least {min_length} characters"
                )
            if max_length is not None and len(result) > max_length:
                raise ValidationError(
                    f"String must be at most {max_length} characters"
                )

            return result

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Cannot convert to string: {e}")

    async def fix(self, value: Any, **kwargs) -> str:
        """Attempt to fix string value.

        Args:
            value: Value to fix
            **kwargs: Fix options

        Returns:
            Fixed string value
        """
        if value is None:
            # Use default value if provided
            default = kwargs.get("default")
            if default is not None:
                return default
            return ""

        # Force convert to string
        result = str(value)

        # Truncate if too long
        max_length = kwargs.get("max_length")
        if max_length and len(result) > max_length:
            result = result[:max_length]

        # Pad if too short
        min_length = kwargs.get("min_length")
        if min_length and len(result) < min_length:
            pad_char = kwargs.get("pad_char", " ")
            result = result.ljust(min_length, pad_char)

        return result
