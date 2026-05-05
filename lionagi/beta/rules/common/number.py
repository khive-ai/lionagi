# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..rule import Rule, RuleParams, RuleQualifier

__all__ = ("NumberRule",)


def _get_number_params() -> RuleParams:
    """Default params: applies to int/float via ANNOTATION qualifier, auto_fix enabled."""
    return RuleParams(
        apply_types={int, float},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class NumberRule(Rule):
    """Rule for validating and converting numeric values.

    Features:
    - Type checking (int or float)
    - Range constraints (ge, gt, le, lt)
    - Auto-conversion from string/other types to number

    Usage:
        rule = NumberRule(ge=0.0, le=1.0)  # Confidence score
        result = await rule.invoke("confidence", 0.95, float)
    """

    def __init__(
        self,
        ge: int | float | None = None,
        gt: int | float | None = None,
        le: int | float | None = None,
        lt: int | float | None = None,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize number rule.

        Args:
            ge: Greater than or equal to (>=)
            gt: Greater than (>)
            le: Less than or equal to (<=)
            lt: Less than (<)
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_number_params()
        super().__init__(params, **kw)
        self.ge = ge
        self.gt = gt
        self.le = le
        self.lt = lt

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a number within constraints.

        Raises:
            ValueError: If not a number or constraints violated
        """
        if not isinstance(v, (int, float)):
            raise ValueError(f"Invalid number value: expected int or float, got {type(v).__name__}")

        if self.ge is not None and v < self.ge:
            raise ValueError(f"Number too small: {v} < {self.ge}")
        if self.gt is not None and v <= self.gt:
            raise ValueError(f"Number too small: {v} <= {self.gt}")
        if self.le is not None and v > self.le:
            raise ValueError(f"Number too large: {v} > {self.le}")
        if self.lt is not None and v >= self.lt:
            raise ValueError(f"Number too large: {v} >= {self.lt}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert value to number and re-validate.

        Returns:
            Numeric value (int or float), validated against constraints

        Raises:
            ValueError: If conversion or re-validation fails
        """
        fixed: int | float
        try:
            if isinstance(v, str):
                v = v.strip()
            fixed = int(v) if t is int else float(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert {v} to number") from e

        await self.validate(fixed, t)
        return fixed
