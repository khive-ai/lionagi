from __future__ import annotations

from typing import Any

import msgspec


def parse_assignment(
    expr: str,
) -> tuple[list[str], list[str], list[tuple[list[str], list[str]]]]:
    """
    Returns: (initial_inputs, final_outputs, steps)
    steps: list of (inputs, outputs) for each 'a,b->c' segment.
    final outputs = produced fields that never appear as inputs in ANY step.
    initial inputs = inputs that are not produced before needed.
    """
    steps: list[tuple[list[str], list[str]]] = []
    produced_all: list[str] = []
    inputs_all: list[str] = []

    for seg in [s.strip() for s in expr.split(";") if s.strip()]:
        if "->" not in seg:
            raise ValueError(f"Invalid segment: {seg!r}")
        left, right = [x.strip() for x in seg.split("->", 1)]
        ins = [t.strip() for t in left.split(",") if t.strip()]
        outs = [t.strip() for t in right.split(",") if t.strip()]
        steps.append((ins, outs))
        produced_all.extend(outs)
        inputs_all.extend(ins)

    # initial inputs = inputs that are not produced before they are needed
    initial_inputs: list[str] = []
    produced_so_far: set[str] = set()
    for ins, outs in steps:
        for i in ins:
            if i not in produced_so_far and i not in initial_inputs:
                initial_inputs.append(i)
        produced_so_far.update(outs)

    # final outputs = outputs that are never used as inputs anywhere
    final_outputs = [o for o in dict.fromkeys(produced_all) if o not in set(inputs_all)]
    return initial_inputs, final_outputs, steps


class BaseForm(msgspec.Struct, kw_only=True):
    # Keep an observable identity for forms as well
    id: str = msgspec.field(default_factory=lambda: __import__("uuid").uuid4().hex)
    has_processed: bool = False


class Form(BaseForm, kw_only=True):
    assignment: str = ""  # "a,b->c; c->d"
    input_fields: list[str] = msgspec.field(default_factory=list)
    output_fields: list[str] = msgspec.field(default_factory=list)
    steps: list[tuple[list[str], list[str]]] = msgspec.field(default_factory=list)
    values: dict[str, Any] = msgspec.field(default_factory=dict)
    guidance: str = ""  # optional: prompt hint
    task: str = ""  # optional: intent description

    def parse(self) -> None:
        self.input_fields, self.output_fields, self.steps = parse_assignment(self.assignment)

    def check_inputs(self) -> None:
        missing = [k for k in self.input_fields if k not in self.values]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")

    def check_outputs(self) -> None:
        missing = [k for k in self.output_fields if k not in self.values]
        if missing:
            raise ValueError(f"Missing outputs: {missing}")

    def to_instructions(self) -> dict:
        return {
            "assignment": self.assignment,
            "guidance": self.guidance,
            "task": self.task,
        }

    def get_results(self) -> dict:
        return {k: self.values.get(k) for k in self.output_fields}
