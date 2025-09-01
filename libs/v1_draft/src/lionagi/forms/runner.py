from __future__ import annotations

from ..base import Branch, Runner
from .compile import MorphismFactory, compile_flow_to_graph
from .spec import FormSpec


async def run_form(
    br: Branch,
    form: FormSpec,
    registry: dict[str, MorphismFactory],
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
        from lionagi.base.ipu import StrictIPU, default_invariants

        runner = Runner(ipu=StrictIPU(default_invariants()))

    results = await runner.run(br, g)

    # compute finals (from branch.ctx), update form.output_fields if empty
    finals_map = {k: br.ctx[k] for k in finals if k in br.ctx}
    if not form.output_fields:
        form.output_fields = list(finals)

    return {"results": results, "finals": finals_map}
