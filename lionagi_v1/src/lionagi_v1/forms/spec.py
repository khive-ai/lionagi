from __future__ import annotations
import msgspec
from typing import Dict, List, Set, Tuple

class StepSpec(msgspec.Struct, frozen=True):
    """
    Declarative step:
      - op: name to resolve in registry
      - inputs: ctx keys this step needs
      - outputs: ctx keys this step will populate (after patch)
      - bind: map inner-op param -> ctx key (rename when needed)
      - out_map: map result key -> ctx key (if result names differ)
    """
    name: str
    op: str
    inputs: List[str]
    outputs: List[str]
    bind: Dict[str, str] = msgspec.field(default_factory=dict)
    out_map: Dict[str, str] = msgspec.field(default_factory=dict)
    doc: str | None = None

class FlowSpec(msgspec.Struct, frozen=True):
    steps: List[StepSpec]

class FormSpec(msgspec.Struct, frozen=False):
    """
    High-level form. All orchestration happens via FlowSpec.
    'output_fields' optional; if absent, computed as finals(flow).
    """
    name: str
    flow: FlowSpec
    output_fields: List[str] = msgspec.field(default_factory=list)
    none_as_valid: bool = False

# ---------- Flow analysis ----------

def required_inputs(flow: FlowSpec) -> Set[str]:
    produced: Set[str] = set()
    req: Set[str] = set()
    for st in flow.steps:
        for i in st.inputs:
            if i not in produced:
                req.add(i)
        # outputs populate ctx by their ctx target names (out_map may remap)
        for o in st.outputs:
            produced.add(st.out_map.get(o, o))
    return req

def final_outputs(flow: FlowSpec) -> Set[str]:
    # outputs that are not used as inputs of later steps
    consumed_later: Set[str] = set()
    # compute outputs as ctx keys after out_map
    ctx_outputs_per_step: List[Set[str]] = []
    for st in flow.steps:
        ctx_outs = {st.out_map.get(o, o) for o in st.outputs}
        ctx_outputs_per_step.append(ctx_outs)

    for idx, st in enumerate(flow.steps):
        for later in flow.steps[idx+1:]:
            consumed_later.update(later.inputs)

    all_ctx_outputs = set().union(*ctx_outputs_per_step) if ctx_outputs_per_step else set()
    return all_ctx_outputs - consumed_later
