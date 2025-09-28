from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from lionagi.ln.types import Params

from .._errors import ValidationError
from ..protocols._concepts import Condition


class RuleQualifier(IntEnum):
    """Qualifier types for rules
    - FIELD: use field name as qualifier
    - ANNOTATION: use field annotation as qualifier
    - CONDITION: use custom condition as qualifier

    default order is FIELD > ANNOTATION > CONDITION
    """

    FIELD = auto()
    ANNOTATION = auto()
    CONDITION = auto()

    @classmethod
    def map_literal(cls, s: str) -> "RuleQualifier":
        s = s.strip().upper()
        if s == "FIELD":
            return cls.FIELD
        elif s == "ANNOTATION":
            return cls.ANNOTATION
        elif s == "CONDITION":
            return cls.CONDITION
        else:
            raise ValueError(f"Unknown RuleQualifier: {s}")


def _decide_qualifier_order(d=None) -> list[RuleQualifier]:
    default_order = [
        RuleQualifier.FIELD,
        RuleQualifier.ANNOTATION,
        RuleQualifier.CONDITION,
    ]
    if d is None:
        return default_order
    if isinstance(d, str):
        d = RuleQualifier.map_literal(d)
    default_order.remove(d)
    return [d] + default_order


@dataclass(slots=True, frozen=True, init=False)
class RuleParams(Params):
    _none_as_sentinel: ClassVar[bool] = True
    apply_types: set[type]
    apply_fields: set[str]
    default_qualifier: RuleQualifier = RuleQualifier.FIELD
    auto_fix: bool = False
    kw: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_type_or_fields(self):
        if sum(bool(x) for x in (self.apply_types, self.apply_fields)) != 1:
            raise ValueError(
                "Either apply_types or apply_fields must be set, but not both."
            )
        return self


class Rule(Condition):

    def __init__(self, params: RuleParams, **kw):
        super().__init__()
        if kw:
            object.__setattr__(params, "kw", {**params.kw, **kw})
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

    async def rule_condition(self, k, v, t, **kw) -> bool:
        raise NotImplementedError(
            "rule_condition must be implemented in subclass to use CONDITION qualifier"
        )

    async def _apply(
        self, k: str, v: Any, t: type, q: RuleQualifier, /, **kw
    ) -> bool:
        """Determine if the rule applies based on the qualifier.

        Args:
            k (str): field name
            v (Any): field value
            t (type): field type
            q (RuleQualifier): qualifier type
            **kw: additional keyword arguments

        Returns:
            bool: True if the rule applies, False otherwise.
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
        t: type = None,
        qualifier: Literal["field", "annotation", "condition"] = None,
        **kw,
    ) -> bool:
        """Apply the rule based on the specified qualifier or default order.

        Args:
            field (str): field name
            value (Any): field value
            annotation (type): field type
            qualifier (Literal["field", "annotation", "condition"], optional): qualifier type. Defaults to None.
            **kw: additional keyword arguments for condition checking.

        Returns:
            bool: True if the rule applies, False otherwise.

        Note:
            If qualifier is None, the default order is used: FIELD > ANNOTATION > CONDITION
            If any qualifier matches, the rule is applied and returns True.
        """
        _order = _decide_qualifier_order(qualifier)

        for q in _order:
            if await self._apply(k, v, t, q, **kw):
                return True
        return False

    @abstractmethod
    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate the value. Should be implemented in subclass.
        Args:
            value (Any): The value to validate.
            **kw: Additional keyword arguments.

        Note:
            Should raise an exception if validation fails.
        """

    async def invoke(self, k: str, v: Any, t: type = None) -> Any:
        """
        Invokes the rule's validation logic on a field and value.

        Args:
            field (str): The field being validated.
            value (Any): The value of the field.

        Returns:
            Any: The validated or fixed value.

        Raises:
            ValidationError: If validation or fixing fails.
        """
        try:
            await self.validate(v, **self.validation_kwargs)
        except Exception as e:
            if self.auto_fix:
                try:
                    return await self.perform_fix(v, t)
                except Exception as e1:
                    raise ValidationError(
                        f"failed to fix field: {k} with fix error: {e1}"
                    ) from e
            raise ValidationError(f"failed to validate field: {k}") from e

    async def perform_fix(self, v: Any, t: type) -> Any:
        """
        Attempts to fix a value if validation fails.

        Args:
            value (Any): The value to fix.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The fixed value.
        """
        raise NotImplementedError(
            "perform_fix must be implemented in subclass to use auto_fix"
        )
