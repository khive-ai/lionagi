from __future__ import annotations

import msgspec


class StepSpec(msgspec.Struct, frozen=True):
    """
    Declarative step:
      - op: name to resolve in registry
      - inputs: ctx keys this step needs
      - outputs: result keys produced by the step operation; they are copied to Branch.ctx via out_map (result_key -> ctx_key); default is identity mapping
      - bind: map inner-op param -> ctx key (rename when needed)
      - out_map: map result key -> ctx key (if result names differ from outputs)
    """

    name: str
    op: str
    inputs: list[str]
    outputs: list[str]
    bind: dict[str, str] = msgspec.field(default_factory=dict)
    out_map: dict[str, str] = msgspec.field(default_factory=dict)
    doc: str | None = None


class FlowSpec(msgspec.Struct, frozen=True):
    steps: list[StepSpec]

    def validate(self) -> None:
        """Validate flow specification constraints."""
        for step in self.steps:
            validate_step(step)


class FormSpec(msgspec.Struct, frozen=False):
    """
    High-level form. All orchestration happens via FlowSpec.
    'output_fields' optional; if absent, computed as finals(flow).
    """

    name: str
    flow: FlowSpec
    output_fields: list[str] = msgspec.field(default_factory=list)
    none_as_valid: bool = False


# ---------- Validation ----------


def validate_step(step: StepSpec) -> None:
    """Validate individual step specification constraints."""
    # Validate out_map keys are subset of outputs
    if step.out_map:
        unknown_keys = set(step.out_map.keys()) - set(step.outputs)
        if unknown_keys:
            raise ValueError(
                f"Step '{step.name}': out_map keys not in outputs: {sorted(unknown_keys)}"
            )

    # Validate no duplicate outputs
    if len(step.outputs) != len(set(step.outputs)):
        duplicates = [x for x in step.outputs if step.outputs.count(x) > 1]
        raise ValueError(f"Step '{step.name}': duplicate outputs: {sorted(set(duplicates))}")

    # Validate no duplicate inputs
    if len(step.inputs) != len(set(step.inputs)):
        duplicates = [x for x in step.inputs if step.inputs.count(x) > 1]
        raise ValueError(f"Step '{step.name}': duplicate inputs: {sorted(set(duplicates))}")


# ---------- Flow analysis ----------


def required_inputs(flow: FlowSpec) -> set[str]:
    produced: set[str] = set()
    req: set[str] = set()
    for st in flow.steps:
        for i in st.inputs:
            if i not in produced:
                req.add(i)
        # outputs populate ctx by their ctx target names (out_map may remap)
        for o in st.outputs:
            produced.add(st.out_map.get(o, o))
    return req


def final_outputs(flow: FlowSpec) -> set[str]:
    # outputs that are not used as inputs of later steps
    consumed_later: set[str] = set()
    # compute outputs as ctx keys after out_map
    ctx_outputs_per_step: list[set[str]] = []
    for st in flow.steps:
        ctx_outs = {st.out_map.get(o, o) for o in st.outputs}
        ctx_outputs_per_step.append(ctx_outs)

    for idx, st in enumerate(flow.steps):
        for later in flow.steps[idx + 1 :]:
            consumed_later.update(later.inputs)

    all_ctx_outputs = set().union(*ctx_outputs_per_step) if ctx_outputs_per_step else set()
    return all_ctx_outputs - consumed_later
