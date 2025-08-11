from __future__ import annotations
from typing import Callable, Dict, Tuple, Set
from lionagi_v1.base.graph import OpNode, OpGraph
from lionagi_v1.morph.binders import BoundOp
from lionagi_v1.morph.wrappers import OpThenPatch
from .spec import FlowSpec, StepSpec, required_inputs, final_outputs

# MorphismFactory: given a StepSpec -> Morphism instance
MorphismFactory = Callable[[StepSpec], object]

def compile_flow_to_graph(
    flow: FlowSpec,
    registry: Dict[str, MorphismFactory],
) -> Tuple[OpGraph, Set[str], Set[str]]:
    """
    Build an OpGraph:
      - Each step becomes: BoundOp(inner(op_from_registry), bind) -> OpThenPatch(...out_map/outputs...)
      - Dependencies inferred from ctx dataflow: if a step consumes a key
        that an earlier step produces (after out_map), add an edge.
    """
    nodes_by_id: Dict = {}
    produced_ctx_key_to_node = {}  # ctx_key -> node_id

    for st in flow.steps:
        if st.op not in registry:
            raise KeyError(f"Registry missing factory for op '{st.op}' (step '{st.name}')")
        inner = registry[st.op](st)
        bound = BoundOp(inner, bind=st.bind)
        # Patch mapping: result_key -> ctx_key
        if st.out_map:
            patch = st.out_map
        else:
            patch = {o: o for o in st.outputs}
        wrapped = OpThenPatch(bound, patch=patch)
        node = OpNode(m=wrapped)
        # deps: any input produced earlier => depend on that producer
        node.deps = set()
        for i in st.inputs:
            prod_node = produced_ctx_key_to_node.get(i)
            if prod_node is not None:
                node.deps.add(prod_node)
        # record producers for this step's ctx outputs
        for ctx_key in patch.values():
            produced_ctx_key_to_node[ctx_key] = node.id
        nodes_by_id[node.id] = node

    roots = {nid for nid, n in nodes_by_id.items() if not n.deps}
    graph = OpGraph(nodes=nodes_by_id, roots=roots)
    req = required_inputs(flow)
    finals = final_outputs(flow)
    return graph, req, finals
