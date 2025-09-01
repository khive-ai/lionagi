from __future__ import annotations

from collections.abc import Callable

from ..base.graph import OpGraph, OpNode
from ..morph.binders import BoundOp
from ..morph.wrappers import OpThenPatch
from .spec import FlowSpec, StepSpec, final_outputs, required_inputs

# MorphismFactory: given a StepSpec -> Morphism instance
MorphismFactory = Callable[[StepSpec], object]


def compile_flow_to_graph(
    flow: FlowSpec,
    registry: dict[str, MorphismFactory],
) -> tuple[OpGraph, set[str], set[str]]:
    """
    Build an OpGraph:
      - Each step becomes: BoundOp(inner(op_from_registry), bind) -> OpThenPatch(...out_map/outputs...)
      - Dependencies inferred from ctx dataflow: if a step consumes a key
        that an earlier step produces (after out_map), add an edge.
    """
    nodes_by_id: dict = {}
    produced_ctx_key_to_node = {}  # ctx_key -> node_id

    for st in flow.steps:
        if st.op not in registry:
            raise KeyError(f"Registry missing factory for op '{st.op}' (step '{st.name}')")

        # Validate out_map keys are subset of outputs
        if st.out_map:
            unknown = set(st.out_map.keys()) - set(st.outputs)
            if unknown:
                raise ValueError(
                    f"Step '{st.name}': out_map keys not in outputs: {sorted(unknown)}"
                )

        inner = registry[st.op](st)

        # Type check inner - ensure it has morphism-like methods
        for attr in ("pre", "apply", "post"):
            if not callable(getattr(inner, attr, None)):
                raise TypeError(f"Registry op '{st.op}' did not return a Morphism-like object")

        bound = BoundOp(inner, bind=st.bind)
        # Patch mapping: result_key -> ctx_key
        if st.out_map:
            patch = st.out_map
        else:
            patch = {o: o for o in st.outputs}
        wrapped = OpThenPatch(bound, patch=patch)
        node = OpNode(m=wrapped)

        # Store step name for debugging
        node.params["step_name"] = st.name

        # deps: any input produced earlier => depend on that producer
        node.deps = set()
        for i in st.inputs:
            prod_node = produced_ctx_key_to_node.get(i)
            if prod_node is not None:
                node.deps.add(prod_node)

        # Check for ambiguous producers before recording
        for ctx_key in patch.values():
            if ctx_key in produced_ctx_key_to_node:
                existing_node = nodes_by_id[produced_ctx_key_to_node[ctx_key]]
                existing_step_name = existing_node.params.get("step_name", "?")
                raise ValueError(
                    f"Ambiguous producer for ctx '{ctx_key}': steps "
                    f"'{existing_step_name}' and '{st.name}'"
                )
            produced_ctx_key_to_node[ctx_key] = node.id

        nodes_by_id[node.id] = node

    roots = {nid for nid, n in nodes_by_id.items() if not n.deps}
    graph = OpGraph(nodes=nodes_by_id, roots=roots)
    req = required_inputs(flow)
    finals = final_outputs(flow)
    return graph, req, finals
