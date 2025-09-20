# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

from lionagi._errors import ValidationError
from lionagi.libs.validate.to_num import to_num

from .base import Rule, register_rule


@register_rule("number")
@register_rule("int")
@register_rule("float")
@register_rule("numeric")
class NumberRule(Rule):
    """Rule for validating and converting numeric values."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to fields with numeric annotations."""
        numeric_types = {"int", "float", "number", "numeric"}
        return annotation in numeric_types if annotation else False

    async def validate(self, value: Any, **kwargs) -> Union[int, float]:
        """Validate and convert value to number.

        Args:
            value: Value to validate
            **kwargs: Validation options (min_value, max_value, precision)

        Returns:
            Numeric value

        Raises:
            ValidationError: If not a valid number or constraints violated
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            raise ValidationError("Value cannot be None")

        try:
            # Convert to number
            result = to_num(value, **kwargs)

            # Check constraints
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")

            if min_value is not None and result < min_value:
                raise ValidationError(f"Value must be at least {min_value}")
            if max_value is not None and result > max_value:
                raise ValidationError(f"Value must be at most {max_value}")

            # Apply precision if specified
            precision = kwargs.get("precision")
            if precision is not None and isinstance(result, float):
                result = round(result, precision)

            # Force to int if requested
            if kwargs.get("force_int", False):
                result = int(result)

            return result

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid number: {e}")

    async def fix(self, value: Any, **kwargs) -> Union[int, float]:
        """Attempt to fix numeric value.

        Args:
            value: Value to fix
            **kwargs: Fix options

        Returns:
            Fixed numeric value
        """
        if value is None:
            # Use default if provided
            default = kwargs.get("default", 0)
            return default

        try:
            # Try aggressive conversion
            if isinstance(value, str):
                # Remove common non-numeric characters
                value = value.replace(",", "").replace("$", "").strip()

            result = float(value)

            # Clamp to range if specified
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")

            if min_value is not None and result < min_value:
                result = min_value
            if max_value is not None and result > max_value:
                result = max_value

            # Apply precision
            precision = kwargs.get("precision", 2)
            result = round(result, precision)

            # Force to int if needed
            if kwargs.get("force_int", False):
                result = int(result)

            return result

        except:
            # Last resort - return default
            return kwargs.get("default", 0)
