from enum import Enum
from typing import Any, Literal, get_args, get_origin

from .base import Rule, RuleParams, RuleQualifier


def _is_literal(tp):
    return get_origin(tp) is Literal


def _get_choice_params():
    return RuleParams(
        apply_types={Enum},
        apply_fields=set(),
        default_qualifier=RuleQualifier.CONDITION,
        auto_fix=True,
        kw={},
    )


class ChoiceRule(Rule):

    def __init__(self, params=None, **kw):
        if params is None:
            params = _get_choice_params()
        super().__init__(params, **kw)
        self.keys = None
        if "keys" in self.validation_kwargs:
            self.keys = self.validation_kwargs["keys"]
        elif "choices" in self.validation_kwargs:
            self.keys = self.validation_kwargs["choices"]

    def _get_keys(self, t):
        keys = self.keys
        if _is_literal(t):
            keys = list(get_args(t))
        if isinstance(t, type) and issubclass(t, Enum):
            keys = [e.value for e in t]
        return keys

    async def rule_condition(self, k, v, t, **kw) -> bool:
        if _is_literal(t):
            return True
        if isinstance(t, type) and issubclass(t, Enum):
            return True
        if self.keys is not None:
            return True
        return False

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that the value is within the set of predefined choices."""
        keys = self._get_keys(t)
        if keys is None:
            raise ValueError("No choices available for validation")
        if v not in keys:
            raise ValueError(f"{v} is not in choices {keys}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Suggest a fix for a value that is not within the set of predefined choices."""
        keys = self._get_keys(t)
        if keys is None:
            raise ValueError("No choices available for fixing")

        from lionagi.ln import string_similarity

        fixed_v = string_similarity(
            v,
            keys,
            threshold=0.85,
            case_sensitive=True,
            return_most_similar=True,
        )
        if not fixed_v:
            raise ValueError(
                f"Failed to suggest a fix for {v} from choices {keys}"
            )
        if isinstance(t, type) and issubclass(t, Enum):
            return t(fixed_v)
        return fixed_v
