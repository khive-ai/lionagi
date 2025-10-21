from collections.abc import Mapping
from typing import Any

from lionagi.ln.fuzzy import fuzzy_validate_mapping

from .base import Rule, RuleParams, RuleQualifier


def _get_mapping_params():
    return RuleParams(
        apply_types={dict},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class MappingRule(Rule):
    """
    Rule for validating that a value is a mapping (dictionary) with specific keys.

    Uses type annotation targeting to apply to dict fields.
    Supports fuzzy key matching and automatic dictionary conversion.
    Uses lionagi.ln.fuzzy.fuzzy_validate_mapping for intelligent fixing.
    """

    def __init__(self, params=None, **kw):
        if params is None:
            params = _get_mapping_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that the value is a mapping with expected keys."""
        if not isinstance(v, Mapping):
            raise ValueError(f"Invalid mapping value: {v}")

        # Check specific keys if provided
        expected_keys = self.validation_kwargs.get("keys")
        if expected_keys is not None:
            current_keys = set(v.keys())
            expected_keys_set = set(expected_keys)
            if current_keys != expected_keys_set:
                raise ValueError(
                    f"Invalid mapping keys. Current keys {list(current_keys)} != {list(expected_keys_set)}"
                )

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to fix the value using fuzzy validation."""
        expected_keys = self.validation_kwargs.get("keys", [])

        # Set defaults for fuzzy matching if not provided
        fuzzy_params = {
            "fuzzy_match": True,
            "handle_unmatched": "remove",
            "similarity_threshold": 0.85,
            "suppress_conversion_errors": False,
            **self.validation_kwargs,  # Pass all validation kwargs, let it fail if invalid
        }

        try:
            return fuzzy_validate_mapping(v, expected_keys, **fuzzy_params)
        except Exception as e:
            raise ValueError(f"Failed to fix mapping value: {e}") from e
