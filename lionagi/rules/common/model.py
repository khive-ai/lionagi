# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..rule import Rule, RuleParams, RuleQualifier

__all__ = ("BaseModelRule",)


def _get_basemodel_params() -> RuleParams:
    """Default params: applies to BaseModel via ANNOTATION qualifier, auto_fix enabled."""
    return RuleParams(
        apply_types={BaseModel},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class BaseModelRule(Rule):
    """Rule for validating Pydantic BaseModel subclasses.

    Validates values against expected Pydantic model types with auto-conversion from dict.
    Uses model_validate() for dict-to-model conversion when auto_fix is enabled.

    Usage:
        rule = BaseModelRule()
        result = await rule.invoke("config", {"name": "test"}, MyModel)  # -> MyModel instance
    """

    def __init__(
        self,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize BaseModel rule.

        Args:
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_basemodel_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate value as a Pydantic model.

        Args:
            v: Value to validate
            t: Expected BaseModel subclass

        Raises:
            ValueError: If value cannot be validated as the model type
        """
        if not isinstance(t, type) or not issubclass(t, BaseModel):
            raise ValueError(f"expected_type must be a BaseModel subclass, got {t}")

        if isinstance(v, t):
            return

        from lionagi.lndl.types import ActionCall, has_action_calls

        if isinstance(v, ActionCall):
            return
        if isinstance(v, BaseModel) and has_action_calls(v):
            return

        if isinstance(v, dict):
            try:
                t.model_validate(v)
                return
            except Exception as e:
                raise ValueError(f"Dict validation failed: {e}") from e

        raise ValueError(f"Cannot validate {type(v).__name__} as {t.__name__}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert value to model using standard validation.

        Args:
            v: Value to fix
            t: Expected BaseModel subclass

        Returns:
            Validated model instance

        Raises:
            ValueError: If conversion fails
        """
        if not isinstance(t, type) or not issubclass(t, BaseModel):
            raise ValueError(f"expected_type must be a BaseModel subclass, got {t}")

        if isinstance(v, t):
            return v

        from lionagi.lndl.types import ActionCall, has_action_calls

        if isinstance(v, ActionCall):
            return v
        if isinstance(v, BaseModel) and has_action_calls(v):
            return v

        if isinstance(v, dict):
            try:
                return t.model_validate(v)
            except Exception as e:
                raise ValueError(f"Cannot convert dict to {t.__name__}: {e}") from e

        raise ValueError(f"Cannot convert {type(v).__name__} to {t.__name__}")
