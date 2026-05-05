# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..rule import Rule, RuleParams, RuleQualifier

__all__ = ("ChoiceRule",)


class ChoiceRule(Rule):
    """Rule for validating values against allowed choices.

    Features:
    - Validates value is in allowed set
    - Optional case-insensitive matching for strings
    - Auto-correction to closest match (fuzzy matching)

    Usage:
        rule = ChoiceRule(
            choices=["low", "medium", "high"],
            case_sensitive=False
        )
        result = await rule.invoke("priority", "HIGH", str)  # → "high"
    """

    def __init__(
        self,
        choices: set[Any] | list[Any],
        case_sensitive: bool = True,
        apply_fields: set[str] | None = None,
        apply_types: set[type] | None = None,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize choice rule.

        Args:
            choices: Allowed values
            case_sensitive: Whether string matching is case-sensitive
            apply_fields: Field names to apply to
            apply_types: Types to apply to
            params: Custom RuleParams (overrides other settings)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = RuleParams(
                apply_types=set(apply_types) if apply_types else set(),
                apply_fields=set(apply_fields) if apply_fields else set(),
                default_qualifier=(
                    RuleQualifier.FIELD if apply_fields else RuleQualifier.ANNOTATION
                ),
                auto_fix=True,
                kw={},
            )
        super().__init__(params, **kw)
        self.choices = set(choices) if not isinstance(choices, set) else choices
        self.case_sensitive = case_sensitive

        if not case_sensitive:
            self._lower_map = {str(c).lower(): c for c in self.choices if isinstance(c, str)}

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is in allowed choices (exact match only).

        For case-insensitive matching, validation will fail for non-canonical
        values, triggering perform_fix() to normalize.

        Raises:
            ValueError: If value not in choices (exact match)
        """
        if v in self.choices:
            return

        raise ValueError(f"Invalid choice: {v} not in {sorted(str(c) for c in self.choices)}")

    async def perform_fix(self, v: Any, _t: type) -> Any:
        """Attempt to fix value to closest choice.

        For strings with case_sensitive=False, returns canonical case.
        Otherwise, raises error.

        Returns:
            Canonical choice value

        Raises:
            ValueError: If cannot fix
        """
        if v in self.choices:
            return v

        if not self.case_sensitive and isinstance(v, str):
            v_lower = v.lower()
            if v_lower in self._lower_map:
                return self._lower_map[v_lower]

        raise ValueError(f"Cannot fix choice: {v} not in {sorted(str(c) for c in self.choices)}")
