from __future__ import annotations
from typing import Dict, Callable
from lionagi_v1.base.types import Branch
from lionagi_v1.base.runner import Runner
from lionagi_v1.forms.spec import FormSpec, final_outputs, required_inputs
from lionagi_v1.forms.compile import compile_flow_to_graph, MorphismFactory

async def run_form(
    br: Branch,
    form: FormSpec,
    registry: Dict[str, MorphismFactory],
    runner: Runner | None = None,
) -> dict:
    """
    Validate inputs -> compile -> run -> return results and projected finals.

    Returns:
      {
        "results": { node_id: op_result, ... },
        "finals": { key: value, ... }
      }
    """
    g, req, finals = compile_flow_to_graph(form.flow, registry)

    # early input check
    missing = sorted([k for k in req if k not in br.ctx])
    if missing:
        raise ValueError(f"Form '{form.name}': missing required inputs in Branch.ctx: {missing}")

    if runner is None:
        from lionagi_v1.base.ipu import StrictIPU, default_invariants  # keep your existing names
        runner = Runner(ipu=StrictIPU(default_invariants()))

    results = await runner.run(br, g)

    # compute finals (from branch.ctx), update form.output_fields if empty
    finals_map = {k: br.ctx[k] for k in finals if k in br.ctx}
    if not form.output_fields:
        form.output_fields = list(finals)

    return {"results": results, "finals": finals_map}
