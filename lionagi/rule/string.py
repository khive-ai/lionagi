from typing import Any
from .base import Rule, RuleParams, RuleQualifier


def _get_string_params():
    return RuleParams(
        apply_types={str},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={}
    )


class StringRule(Rule):
    """
    Rule for validating and converting string values.

    Uses type annotation targeting to apply to string fields.
    Supports automatic conversion from various types to string.
    """

    def __init__(self, params=None, **kw):
        if params is None:
            params = _get_string_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that the value is a string."""
        if not isinstance(v, str):
            raise ValueError(f"Invalid string value: {v}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert a value to a string."""
        try:
            return str(v)
        except Exception as e:
            raise ValueError(
                f"Failed to convert {v} into a string value"
            ) from e
