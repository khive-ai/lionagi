# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Form: input/output contract for a single work unit using the assignment DSL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from lionagi.beta.core.base.element import Element

__all__ = ("Form", "ParsedAssignment", "parse_assignment", "parse_full_assignment")

_BRANCH_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass
class ParsedAssignment:
    """Result of parsing the full assignment DSL."""

    branch: str | None
    inputs: list[str]
    outputs: list[str]
    resource: str | None
    raw: str


def parse_assignment(assignment: str) -> tuple[list[str], list[str]]:
    parsed = parse_full_assignment(assignment)
    return parsed.inputs, parsed.outputs


def parse_full_assignment(assignment: str) -> ParsedAssignment:
    """Parse the full 'branch: inputs -> outputs | resource' assignment DSL."""
    raw = assignment.strip()
    branch = None
    resource = None

    if "|" in raw:
        main_part, resource_part = raw.rsplit("|", 1)
        resource = resource_part.strip()
        raw = main_part.strip()

    if ":" in raw:
        colon_idx = raw.find(":")
        arrow_idx = raw.find("->")
        if (
            (arrow_idx == -1 or colon_idx < arrow_idx)
            and raw[colon_idx + 1 : colon_idx + 2].isspace()
            and _BRANCH_NAME_RE.fullmatch(raw[:colon_idx].strip())
        ):
            branch_part, raw = raw.split(":", 1)
            branch = branch_part.strip()
            raw = raw.strip()

    if "->" not in raw:
        raise ValueError(f"Invalid assignment syntax (missing '->'): {assignment}")

    parts = raw.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid assignment syntax: {assignment}")

    inputs = [f.strip() for f in parts[0].split(",") if f.strip()]
    outputs = [f.strip() for f in parts[1].split(",") if f.strip()]

    return ParsedAssignment(
        branch=branch,
        inputs=inputs,
        outputs=outputs,
        resource=resource,
        raw=assignment,
    )


class Form(Element):
    """Input/output contract and data store for one work unit."""

    assignment: str = Field(
        default="",
        description="Assignment DSL: 'branch: inputs -> outputs | resource'",
    )
    branch: str | None = Field(
        default=None,
        description="Worker/branch name for routing",
    )
    resource: str | None = Field(
        default=None,
        description="Resource hint (e.g., 'api:fast')",
    )
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    available_data: dict[str, Any] = Field(default_factory=dict)
    output: Any = Field(default=None)
    filled: bool = Field(default=False)

    def model_post_init(self, _: Any) -> None:
        if self.assignment and not self.input_fields and not self.output_fields:
            parsed = parse_full_assignment(self.assignment)
            self.input_fields = parsed.inputs
            self.output_fields = parsed.outputs
            if parsed.branch and self.branch is None:
                self.branch = parsed.branch
            if parsed.resource and self.resource is None:
                self.resource = parsed.resource

    def is_workable(self) -> bool:
        """Return True when all declared inputs are present and non-None."""
        if self.filled:
            return False
        for field in self.input_fields:
            if field not in self.available_data:
                return False
            if self.available_data[field] is None:
                return False
        return True

    def get_inputs(self) -> dict[str, Any]:
        return {
            field: self.available_data[field]
            for field in self.input_fields
            if field in self.available_data
        }

    def fill(self, **data: Any) -> None:
        self.available_data.update(data)

    def extract_output_data(self, output: Any) -> dict[str, Any]:
        data = {}
        if output is None:
            return data

        dumped = None
        for field in self.output_fields:
            if isinstance(output, dict) and field in output:
                data[field] = output[field]
            elif hasattr(output, field):
                data[field] = getattr(output, field)
            elif hasattr(output, "model_dump"):
                if dumped is None:
                    dumped = output.model_dump()
                if field in dumped:
                    data[field] = dumped[field]
        return data

    def missing_outputs(self, output: Any | None = None) -> list[str]:
        data = self.extract_output_data(output) if output is not None else {}
        available = {**self.available_data, **data}
        return [
            field
            for field in self.output_fields
            if field not in available or available[field] is None
        ]

    def set_output(self, output: Any, *, partial: bool = False) -> None:
        output_data = self.extract_output_data(output)
        available = {**self.available_data, **output_data}
        missing = [
            field
            for field in self.output_fields
            if field not in available or available[field] is None
        ]
        if missing and not partial:
            raise ValueError(
                f"Output for form '{self.assignment}' missing fields: {missing}"
            )

        self.output = output
        self.available_data.update(output_data)
        self.filled = not missing

    def get_output_data(self) -> dict[str, Any]:
        result = {}
        for field in self.output_fields:
            if field in self.available_data:
                result[field] = self.available_data[field]
        return result

    def __repr__(self) -> str:
        status = (
            "filled" if self.filled else ("ready" if self.is_workable() else "pending")
        )
        return f"Form('{self.assignment}', {status})"
