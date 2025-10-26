from typing import Any

from .base import Rule, RuleParams, RuleQualifier


def _get_number_params():
    return RuleParams(
        apply_types={int, float},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class NumberRule(Rule):
    """
    Rule for validating that a value is a number within specified bounds.

    Uses type annotation targeting to apply to int and float fields.
    Supports automatic conversion from various types to numbers.
    Supports bounds checking and precision control through validation_kwargs.
    """

    def __init__(self, params=None, **kw):
        if params is None:
            params = _get_number_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that the value is a number."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Invalid number value: {v}")

        # Check bounds if specified
        upper_bound = self.validation_kwargs.get("upper_bound")
        lower_bound = self.validation_kwargs.get("lower_bound")

        if upper_bound is not None and v > upper_bound:
            raise ValueError(f"Value {v} exceeds upper bound {upper_bound}")
        if lower_bound is not None and v < lower_bound:
            raise ValueError(f"Value {v} below lower bound {lower_bound}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to fix the value by converting it to a number."""
        if isinstance(v, (int, float)):
            # Apply bounds and precision if needed
            fixed_v = v
        else:
            from lionagi.libs.validate.to_num import to_num

            fixed_v = to_num(v, **self.validation_kwargs)

        # Re-validate the fixed value
        await self.validate(fixed_v, t)
        return fixed_v
