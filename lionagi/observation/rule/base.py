# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Dict, Optional, Set

from lionagi._errors import ValidationError
from lionagi.ln.types import Params, not_sentinel
from lionagi.protocols._concepts import Condition


class RuleCondition(Condition):

    async def apply(self, **kw) -> bool:
        return True


@dataclass(frozen=True, slots=True, init=False)
class RuleConfig(Params):

    include_types: set[type] = dataclass_field(default_factory=set)
    """Types/annotations to include."""

    exclude_types: set[type] = dataclass_field(default_factory=set)
    """Types/annotations to exclude."""

    validation_kwargs: dict[str, Any] = dataclass_field(default_factory=dict)
    """Additional keyword arguments for validation."""

    fix_kwargs: dict[str, Any] = dataclass_field(default_factory=dict)
    """Additional keyword arguments for fixing."""

    auto_fix: bool = False
    """Whether this rule can attempt to fix invalid values."""

    none_as_valid: bool = False
    """If True, None is considered a valid value."""

    apply_fields: set[str] = dataclass_field(default_factory=set)
    """Specific field names to which this rule applies. Empty means all fields of the applicable types."""

    use_annotation: bool = True
    """Whether to consider type annotations when determining applicability."""

    condition: RuleCondition | None = None
    """An optional condition that must be met for the rule to apply."""

    def update(self, **kw) -> RuleConfig:
        params = self.to_dict()
        params.update(kw)
        return RuleConfig(**params)


class Rule:

    def __init__(self, config: RuleConfig | None = None, **kw):
        """Initialize rule with configuration."""
        self.config = config or RuleConfig()
        if kw:
            self.config = self.config.update(**kw)

    def is_valid_value(self, value: Any) -> bool:
        if value is None and self.config.none_as_valid:
            return True
        return not_sentinel(value)

    async def apply(
        self,
    ): ...

    async def apply(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: Optional[str] = None,
        **kwargs,
    ) -> bool:
        # Check field inclusion/exclusion
        if (
            self.config.include_fields
            and field not in self.config.include_fields
        ):
            return False
        if field in self.config.exclude_fields:
            return False

        # Check type inclusion/exclusion if annotation provided
        if annotation:
            if (
                self.config.include_types
                and annotation not in self.config.include_types
            ):
                return False
            if annotation in self.config.exclude_types:
                return False

        # Subclasses can add custom logic
        return await self._custom_applies(
            field, value, form, annotation, **kwargs
        )

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Custom application logic for subclasses.

        Override this to add custom logic for when a rule applies.

        Returns:
            True if rule should be applied (default)
        """
        return True

    @abstractmethod
    async def validate(self, value: Any, **kwargs) -> Any:
        """Validate a value.

        Args:
            value: Value to validate
            **kwargs: Additional validation context

        Returns:
            Validated value (possibly transformed)

        Raises:
            ValidationError: If validation fails
        """
        pass

    async def fix(self, value: Any, **kwargs) -> Any:
        """Attempt to fix an invalid value.

        Default implementation just returns the value unchanged.
        Override this to provide fixing logic.

        Args:
            value: Value to fix
            **kwargs: Additional context

        Returns:
            Fixed value

        Raises:
            ValidationError: If fixing fails
        """
        return value

    async def invoke(self, field: str, value: Any, form: Any, **kwargs) -> Any:
        """Invoke the rule on a field value.

        This is the main entry point for rule execution.
        It attempts validation and optionally fixing.

        Args:
            field: Field name
            value: Field value
            form: Form context
            **kwargs: Additional context

        Returns:
            Validated/fixed value

        Raises:
            ValidationError: If validation and fixing both fail
        """
        try:
            # Try to validate
            return await self.validate(value, **self.config.kwargs)
        except ValidationError as e:
            # If validation fails and we can fix, try fixing
            if self.config.can_fix:
                try:
                    return await self.fix(value, **self.config.kwargs)
                except Exception as fix_error:
                    # If fixing also fails, re-raise original validation error
                    raise e from fix_error
            # Can't fix, re-raise validation error
            raise


# Registry for rule classes
_rule_registry: Dict[str, type[Rule]] = {}


def register_rule(name: str = None):
    """Decorator to register a rule class.

    Args:
        name: Optional name for the rule (defaults to class name)
    """

    def decorator(cls: type[Rule]):
        rule_name = name or cls.__name__
        _rule_registry[rule_name] = cls
        return cls

    return decorator


def get_rule(name: str) -> type[Rule]:
    """Get a registered rule class by name.

    Args:
        name: Rule name

    Returns:
        Rule class

    Raises:
        KeyError: If rule not found
    """
    if name not in _rule_registry:
        raise KeyError(f"Rule '{name}' not registered")
    return _rule_registry[name]
