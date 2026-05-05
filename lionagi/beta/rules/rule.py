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
    """Determines when a rule applies: FIELD > ANNOTATION > CONDITION by default."""

    FIELD = auto()
    ANNOTATION = auto()
    CONDITION = auto()

    @classmethod
    def from_str(cls, s: str) -> RuleQualifier:
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
    """Immutable rule configuration: target types/fields, qualifier precedence, and auto-fix flag."""

    apply_types: set[type] = field(default_factory=set)
    apply_fields: set[str] = field(default_factory=set)
    default_qualifier: RuleQualifier = RuleQualifier.FIELD
    auto_fix: bool = False
    kw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        # Empty sets are valid — explicit/manual rule invocation bypasses qualifier matching.
        pass


class Rule:
    """Base validation rule: apply() checks qualification, validate() is abstract, perform_fix() is optional."""

    def __init__(self, params: RuleParams, **kw):
        if kw:
            params = params.with_updates(kw={**params.kw, **kw})
        self.params = params

    @property
    def apply_types(self) -> set[type]:
        return self.params.apply_types

    @property
    def apply_fields(self) -> set[str]:
        return self.params.apply_fields

    @property
    def default_qualifier(self) -> RuleQualifier:
        return self.params.default_qualifier

    @property
    def auto_fix(self) -> bool:
        return self.params.auto_fix

    @property
    def validation_kwargs(self) -> dict:
        return self.params.kw

    async def rule_condition(self, k: str, v: Any, t: type, **kw) -> bool:
        """Override to enable the CONDITION qualifier; raises NotImplementedError by default."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement rule_condition() to use CONDITION qualifier"
        )

    async def _apply(self, k: str, v: Any, t: type, q: RuleQualifier, **kw) -> bool:
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
        _order = _decide_qualifier_order(qualifier)

        for q in _order:
            try:
                if await self._apply(k, v, t or type(v), q, **kw):
                    return True
            except NotImplementedError:
                continue

        return False

    @abstractmethod
    async def validate(self, v: Any, t: type, **kw) -> None: ...

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Override to enable auto-correction; raises NotImplementedError by default."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement perform_fix() to use auto_fix=True"
        )

    async def invoke(
        self, k: str, v: Any, t: type | None = None, *, auto_fix: bool | None = None
    ) -> Any:
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
        return (
            f"{self.__class__.__name__}("
            f"types={self.apply_types}, "
            f"fields={self.apply_fields}, "
            f"auto_fix={self.auto_fix})"
        )
