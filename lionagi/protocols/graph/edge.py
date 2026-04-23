# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from lionagi.utils import is_same_dtype

from .._concepts import Condition, Relational
from ..generic.element import ID, Element


class EdgeCondition(BaseModel, Condition):
    """Concrete Condition with Pydantic model support for edge traversal.

    ``source`` is a general-purpose slot for conditions that need to
    carry state alongside their ``apply()`` logic.  ``extra='allow'``
    lets subclasses assign attributes in ``__init__`` without declaring
    them as model fields.
    """

    source: Any = Field(default=None)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    async def apply(self, *args, **kwargs) -> bool:
        return True


__all__ = (
    "Edge",
    "EdgeCondition",
)


class Edge(Element):
    """A directed edge connecting head → tail, with optional Condition.

    Condition controls traversal: ``check_condition()`` delegates to
    ``condition.apply()``.  Any ``Condition`` subclass works — just
    implement ``async def apply(self, *args, **kwargs) -> bool``.
    """

    head: UUID
    tail: UUID
    properties: dict[str, Any] = Field(
        default_factory=dict,
        title="Properties",
        description="Custom properties associated with this edge.",
    )

    def __init__(
        self,
        head: ID[Relational].Ref,
        tail: ID[Relational].Ref,
        condition: Condition | None = None,
        label: list[str] | None = None,
        **kwargs,
    ):
        head = ID.get_id(head)
        tail = ID.get_id(tail)
        if condition is not None:
            if not isinstance(condition, Condition):
                raise ValueError(
                    "Condition must be an instance of EdgeCondition."
                )
            kwargs["condition"] = condition
        if label:
            if isinstance(label, str):
                kwargs["label"] = [label]
            elif isinstance(label, list) and is_same_dtype(label, str):
                kwargs["label"] = label
            else:
                raise ValueError("Label must be a string or a list of strings.")

        super().__init__(head=head, tail=tail, properties=kwargs)

    @field_serializer("head", "tail")
    def _serialize_id(self, value: UUID) -> str:
        return str(value)

    @field_validator("head", "tail", mode="before")
    def _validate_id(cls, value: str) -> UUID:
        return ID.get_id(value)

    @property
    def label(self) -> list[str] | None:
        return self.properties.get("label", None)

    @property
    def condition(self) -> Condition | None:
        return self.properties.get("condition", None)

    @condition.setter
    def condition(self, value: Condition | None) -> None:
        if value is not None and not isinstance(value, Condition):
            raise ValueError("Condition must be an instance of EdgeCondition.")
        self.properties["condition"] = value

    @label.setter
    def label(self, value: list[str] | None) -> None:
        if not value:
            self.properties["label"] = []
            return
        if isinstance(value, str):
            self.properties["label"] = [value]
            return
        if isinstance(value, list) and is_same_dtype(value, str):
            self.properties["label"] = value
            return
        raise ValueError("Label must be a string or a list of strings.")

    async def check_condition(self, *args, **kwargs) -> bool:
        if self.condition is not None:
            return await self.condition.apply(*args, **kwargs)
        return True

    def update_property(self, key: str, value: Any) -> None:
        self.properties[key] = value


# File: lionagi/protocols/graph/edge.py
