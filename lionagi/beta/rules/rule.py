# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from lionagi._errors import ValidationError
from lionagi.ln.types import Params

__all__ = ("Rule", "RuleParams", "RuleQualifier", "ValidationError")


class RuleQualifier(IntEnum):
    """Qualifier types for rules - determines WHEN a rule applies.

    - FIELD: Match by field name (e.g., "confidence", "output")
    - ANNOTATION: Match by type annotation (e.g., str, int, float)
    - CONDITION: Match by custom condition (e.g., is BaseModel subclass)

    Default precedence order: FIELD > ANNOTATION > CONDITION
    """

    FIELD = auto()
    ANNOTATION = auto()
    CONDITION = auto()

    @classmethod
    def from_str(cls, s: str) -> RuleQualifier:
        """Convert string to RuleQualifier."""
        s = s.strip().upper()
        if s == "FIELD":
            return cls.FIELD
        elif s == "ANNOTATION":
            return cls.ANNOTATION
        elif s == "CONDITION":
            return cls.CONDITION
        else:
            raise ValueError(f"Unknown RuleQualifier: {s}")


def _decide_qualifier_order(
    qualifier: str | RuleQualifier | None = None,
) -> list[RuleQualifier]:
    """Determine qualifier precedence order (default: FIELD > ANNOTATION > CONDITION).

    Args:
        qualifier: Preferred qualifier moved to front of order. None uses default.

    Returns:
        List of RuleQualifier in precedence order.
    """
    default_order = [
        RuleQualifier.FIELD,
        RuleQualifier.ANNOTATION,
        RuleQualifier.CONDITION,
    ]

    if qualifier is None:
        return default_order

    if isinstance(qualifier, str):
        qualifier = RuleQualifier.from_str(qualifier)

    default_order.remove(qualifier)
    return [qualifier, *default_order]


@dataclass(slots=True, frozen=True)
class RuleParams(Params):
    """Immutable configuration for rules.

    Defines:
    - WHAT the rule applies to (types or fields)
    - HOW to determine applicability (default_qualifier)
    - WHETHER to auto-fix (auto_fix)
    - ADDITIONAL validation parameters (kw)

    Uses ln.types.Params (leaner than Pydantic BaseModel).
    """

    apply_types: set[type] = field(default_factory=set)
    """Types this rule applies to (e.g., {str, int})"""

    apply_fields: set[str] = field(default_factory=set)
    """Field names this rule applies to (e.g., {"confidence", "output"})"""

    default_qualifier: RuleQualifier = RuleQualifier.FIELD
    """Preferred qualifier type"""

    auto_fix: bool = False
    """Enable automatic fixing on validation failure"""

    kw: dict = field(default_factory=dict)
    """Additional validation parameters"""

    def __post_init__(self) -> None:
        """Validate after dataclass initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate params consistency (extensible hook for subclasses).

        Rules use multiple qualifier mechanisms (OR matching):
        - apply_types: ANNOTATION qualifier (type-based)
        - apply_fields: FIELD qualifier (name-based)
        - rule_condition(): CONDITION qualifier (custom logic)

        Empty sets are valid for explicit/manual rule invocation.
        """
        pass


class Rule:
    """Base validation rule with auto-correction support.

    Pattern from lionagi v0.2.2 + lionherd-old:
    1. apply() - Does this rule apply? (uses qualifiers)
    2. validate() - Is the value valid? (abstract, subclass implements)
    3. perform_fix() - Can we auto-correct? (optional, if auto_fix=True)

    Usage:
        # Create rule
        rule = StringRule(min_length=1, max_length=100)

        # Check if applies
        if await rule.apply("name", "Ocean", str):
            # Validate + fix
            result = await rule.invoke("name", "Ocean", str)
    """

    def __init__(self, params: RuleParams, **kw):
        """Initialize rule with parameters.

        Args:
            params: Rule configuration
            **kw: Additional validation kwargs (merged with params.kw)
        """
        if kw:
            # Merge additional kwargs using with_updates
            params = params.with_updates(kw={**params.kw, **kw})
        self.params = params

    @property
    def apply_types(self) -> set[type]:
        """Types this rule applies to (ANNOTATION qualifier)."""
        return self.params.apply_types

    @property
    def apply_fields(self) -> set[str]:
        """Field names this rule applies to (FIELD qualifier)."""
        return self.params.apply_fields

    @property
    def default_qualifier(self) -> RuleQualifier:
        """Preferred qualifier type for apply() precedence."""
        return self.params.default_qualifier

    @property
    def auto_fix(self) -> bool:
        """Whether perform_fix() is called on validation failure."""
        return self.params.auto_fix

    @property
    def validation_kwargs(self) -> dict:
        """Additional parameters passed to validate()."""
        return self.params.kw

    async def rule_condition(self, k: str, v: Any, t: type, **kw) -> bool:
        """Custom condition for CONDITION qualifier.

        Override in subclass to use CONDITION qualifier.

        Args:
            k: Field name
            v: Field value
            t: Field type
            **kw: Additional kwargs

        Returns:
            True if rule should apply
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement rule_condition() to use CONDITION qualifier"
        )

    async def _apply(self, k: str, v: Any, t: type, q: RuleQualifier, **kw) -> bool:
        """Determine if rule applies based on qualifier.

        Args:
            k: Field name
            v: Field value
            t: Field type
            q: Qualifier type
            **kw: Additional kwargs

        Returns:
            True if rule applies
        """
        match q:
            case RuleQualifier.FIELD:
                return k in self.apply_fields

            case RuleQualifier.ANNOTATION:
                return t in self.apply_types

            case RuleQualifier.CONDITION:
                return await self.rule_condition(k, v, t, **kw)

    async def apply(
        self,
        k: str,
        v: Any,
        t: type | None = None,
        qualifier: str | RuleQualifier | None = None,
        **kw,
    ) -> bool:
        """Check if rule applies using qualifier precedence.

        Args:
            k: Field name
            v: Field value
            t: Field type (optional)
            qualifier: Override default qualifier order
            **kw: Additional kwargs for condition checking

        Returns:
            True if rule applies (any qualifier matches)
        """
        _order = _decide_qualifier_order(qualifier)

        for q in _order:
            try:
                if await self._apply(k, v, t or type(v), q, **kw):
                    return True
            except NotImplementedError:
                continue

        return False

    @abstractmethod
    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate value (abstract, implement in subclass).

        Args:
            v: Value to validate
            t: Expected type
            **kw: Additional validation parameters

        Raises:
            Exception: If validation fails
        """
        pass

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to fix invalid value (optional, override in subclass).

        Args:
            v: Value to fix
            t: Expected type

        Returns:
            Fixed value

        Raises:
            NotImplementedError: If auto_fix=True but perform_fix not implemented
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement perform_fix() to use auto_fix=True"
        )

    async def invoke(
        self, k: str, v: Any, t: type | None = None, *, auto_fix: bool | None = None
    ) -> Any:
        """Execute validation with optional auto-fixing.

        Args:
            k: Field name (for error messages)
            v: Value to validate
            t: Field type (optional)
            auto_fix: Override self.auto_fix for this invocation (thread-safe)

        Returns:
            Validated (and possibly fixed) value

        Raises:
            ValidationError: If validation fails and auto_fix disabled
        """
        effective_type = t or type(v)
        should_auto_fix = auto_fix if auto_fix is not None else self.auto_fix
        try:
            await self.validate(v, effective_type, **self.validation_kwargs)
            return v
        except Exception as e:
            if should_auto_fix:
                try:
                    return await self.perform_fix(v, effective_type)
                except Exception as e1:
                    raise ValidationError(f"Failed to fix field '{k}': {e1}") from e
            raise ValidationError(f"Failed to validate field '{k}': {e}") from e

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"types={self.apply_types}, "
            f"fields={self.apply_fields}, "
            f"auto_fix={self.auto_fix})"
        )
