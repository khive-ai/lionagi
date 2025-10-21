from typing import Any

from .base import Rule, RuleParams, RuleQualifier


def _get_boolean_params():
    return RuleParams(
        apply_types={bool},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class BooleanRule(Rule):

    def __init__(self, params=None, **kw):
        if params is None:
            params = _get_boolean_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that the value is a boolean."""
        if not isinstance(v, bool):
            raise ValueError(f"Invalid boolean value: {v}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to fix the value by converting it to a boolean."""
        if not isinstance(v, str):
            v = str(v)

        v = v.strip().lower()
        if v in {"true", "1", "correct", "yes"}:
            return True
        elif v in {"false", "0", "incorrect", "no", "none", "n/a"}:
            return False
        raise ValueError(f"Failed to convert {v} into a boolean value")
