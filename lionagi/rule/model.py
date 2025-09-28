from typing import Any
from .base import Rule, RuleParams, RuleQualifier
from lionagi.ln.fuzzy import fuzzy_validate_pydantic

from pydantic import BaseModel


def _get_model_params():
    return RuleParams(
        apply_types=set(),
        apply_fields=set(),
        default_qualifier=RuleQualifier.CONDITION,
        auto_fix=True,
        kw={}
    )


class ModelRule(Rule):
    """
    Rule for validating Pydantic models with fuzzy parsing and key matching.

    Uses condition-based targeting to apply to fields containing Pydantic models.
    Supports automatic JSON extraction, fuzzy key matching, and model validation.
    Uses lionagi.ln.fuzzy.fuzzy_validate_pydantic for intelligent fixing.
    """

    def __init__(self, params=None, **kw):
        if params is None:
            params = _get_model_params()
        super().__init__(params, **kw)

    async def rule_condition(self, k: str, v: Any, t: type, **kw) -> bool:
        """Apply this rule to fields that are Pydantic model types."""
        try:
            return (
                isinstance(t, type)
                and issubclass(t, BaseModel)
                and hasattr(t, 'model_fields')
            )
        except TypeError:
            return False

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that the value is an instance of the expected Pydantic model."""
        if not isinstance(v, t):
            raise ValueError(f"Invalid model value: expected {t.__name__}, got {type(v).__name__}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to fix the value by parsing and validating as a Pydantic model."""
        # Set defaults and pass all validation kwargs, let it fail if invalid
        fuzzy_params = {
            "fuzzy_parse": True,
            "fuzzy_match": True,
            "fuzzy_match_params": None,
            **self.validation_kwargs
        }

        try:
            return fuzzy_validate_pydantic(
                v,
                model_type=t,
                **fuzzy_params
            )
        except Exception as e:
            raise ValueError(f"Failed to validate as {t.__name__}: {e}") from e