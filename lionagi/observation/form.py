# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field, model_validator

from lionagi.fields.flow import FlowDefinition
from lionagi.ln import not_sentinel
from lionagi.models.field_model import FieldModel
from lionagi.protocols.generic.element import Element


class Form(Element):

    assignment: str | None = None
    """A small DSL describing transformation, e.g. 'a,b -> c'."""

    output_fields: list[str] = Field(default_factory=list)
    """Which fields are considered mandatory outputs."""

    has_processed: bool = False
    """Marks if the form is considered completed or 'processed'."""

    none_as_sentinel: bool = False
    """If True, None is treated as a sentinel value."""

    fields: dict[str, tuple[FieldModel, Any]] = Field(default_factory=dict)
    """The {name: (field_model, value)} in this form, keyed by field name."""

    flow_definition: FlowDefinition | None = None
    guidance: str | None = None
    task: str | None = None

    @model_validator(mode="before")
    def parse_assignment_into_flow(cls, values):
        """
        If the 'assignment' has semicolons, assume multiple steps, parse into FlowDefinition.
        If it's a single step or no semicolons, we remain in 'simple' mode.
        """
        assignment_str = values.get("assignment")
        if assignment_str and ";" in assignment_str:
            flow = FlowDefinition()
            flow.parse_flow_string(assignment_str)
            values["flow_definition"] = flow
        return values

    @model_validator(mode="after")
    def compute_output_fields(self):
        """
        If in simple mode, we parse something like 'a,b->c' and set output_fields=[c].
        If in multi-step mode, we set output_fields to the final produced fields of the flow.
        """
        if self.flow_definition:
            # multi-step
            produced = self.flow_definition.get_produced_fields()
            if not self.output_fields:
                self.output_fields = list(produced)
        else:
            # single-step
            if self.assignment and "->" in self.assignment:
                # parse the single arrow
                ins_outs = self.assignment.split("->", 1)
                outs_str = ins_outs[1]
                outs = [x.strip() for x in outs_str.split(",") if x.strip()]
                if not self.output_fields:
                    self.output_fields = outs
        return self

    def fill_fields(self, update_: bool = False, **kwargs) -> None:
        """
        A small helper: fill fields in this form by direct assignment.
        Usually you'd do 'myform(field=val, field2=val2)', but sometimes you want partial updates.
        """
        for k, v in kwargs.items():
            if k not in self.fields:
                raise ValueError(f"Field '{k}' not defined in form.")
            if self._not_sentinel(v):
                field_model, current_value = self.fields[k]
                if not update_ and self._not_sentinel(current_value):
                    raise ValueError(
                        f"Field '{k}' already has a value, use update_=True to overwrite."
                    )
                # Update the tuple by replacing it (tuples are immutable)
                self.fields[k] = (field_model, v)

    def _not_sentinel(self, value: Any) -> bool:
        """Check if a value is not a sentinel (Undefined or Unset, optionally also None)."""
        if value is None and self.none_as_sentinel:
            return False
        return not_sentinel(value)
