# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..rule import Rule, RuleParams, RuleQualifier

__all__ = ("BooleanRule",)


def _get_boolean_params() -> RuleParams:
    """Default params: applies to bool via ANNOTATION qualifier, auto_fix enabled."""
    return RuleParams(
        apply_types={bool},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class BooleanRule(Rule):
    """Rule for validating and converting boolean values.

    Features:
    - Type checking (must be bool)
    - Auto-conversion from strings ("true", "false", "yes", "no", "1", "0")
    - Auto-conversion from numbers (0 = False, non-zero = True)

    Usage:
        rule = BooleanRule()
        result = await rule.invoke("active", "true", bool)  # → True
    """

    def __init__(self, params: RuleParams | None = None, **kw):
        """Initialize boolean rule.

        Args:
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_boolean_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a boolean.

        Raises:
            ValueError: If not a boolean
        """
        if not isinstance(v, bool):
            raise ValueError(f"Invalid boolean value: expected bool, got {type(v).__name__}")

    async def perform_fix(self, v: Any, _t: type) -> Any:
        """Attempt to convert value to boolean.

        Conversion rules:
        - Strings: "true", "yes", "1", "on" → True (case-insensitive)
        - Strings: "false", "no", "0", "off" → False (case-insensitive)
        - Numbers: 0 → False, non-zero → True
        - Other: bool(v)

        Returns:
            Boolean value

        Raises:
            ValueError: If conversion fails
        """
        try:
            if isinstance(v, str):
                v_lower = v.strip().lower()
                if v_lower in ("true", "yes", "1", "on"):
                    return True
                elif v_lower in ("false", "no", "0", "off"):
                    return False
                else:
                    raise ValueError(f"Cannot parse '{v}' as boolean")

            if isinstance(v, (int, float)):
                return bool(v)

            return bool(v)
        except Exception as e:
            raise ValueError(f"Failed to convert {v} to boolean") from e
