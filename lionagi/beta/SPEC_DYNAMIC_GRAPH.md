# Spec: Dynamic Graph Expansion for Runner

## Problem

Runner executes a static OpGraph — all nodes are known before `run()` starts.
Iterative patterns (ReAct, self-refine, tree-of-thought) need graphs that grow
at runtime based on intermediate results.

## Current State

```
run_stream(br, g):
    validate_dag()
    ready = roots
    while ready:
        batch = [n for n in ready if deps ⊆ done]
        execute wave (regular nodes || control nodes seq)
        for each control result:
            _apply_control_action(action, g, ready, done, results)
            actions: halt, skip, retry, route
```

Control nodes can halt, skip, retry, or route — but cannot create new nodes.

## Design: `spawn` Control Action

### Control node returns

```python
{
    "action": "spawn",
    "nodes": [
        {
            "name": "round_2_operate",      # human-readable, used for _resolve_node_id
            "morphism": <Morphism>,          # required — the executable unit
            "deps": [],                      # UUIDs or names of existing nodes
            "params": {...},                 # passed to morphism.apply as kwargs
            "control": False,               # is this a control node?
        },
        {
            "name": "round_2_control",
            "morphism": <ControlMorphism>,
            "deps": ["round_2_operate"],
            "params": {...},
            "control": True,
        },
    ],
    "reason": "ReAct round 2: actions detected, continuing",
}
```

### Runner handles `spawn` in `_apply_control_action`

```python
if action_name == "spawn":
    if self._total_spawned + len(new_nodes) > self.max_dynamic_nodes:
        raise ExecutionError(
            f"Dynamic node limit ({self.max_dynamic_nodes}) exceeded",
            retryable=False,
        )

    for spec in new_nodes:
        node = OpNode(
            id=uuid4(),
            m=spec["morphism"],
            deps=resolve_deps(spec.get("deps", []), g, name_map),
            params=spec.get("params", {}),
            control=spec.get("control", False),
        )
        # Register name for future dep resolution
        if "name" in spec:
            node.params["_lionagi_operation_name"] = spec["name"]

        g.nodes[node.id] = node
        name_map[spec.get("name", "")] = node.id

        # Add to ready if deps already satisfied
        if node.deps.issubset(done):
            ready.add(node.id)

    self._total_spawned += len(new_nodes)
```

### Invariants preserved

1. **Policy check on spawned nodes**: `_exec_node` already runs `policy_check`
   before any execution. Spawned nodes go through the same path — no special
   case needed.

2. **Accumulated provides**: Spawned nodes see all provides from prior waves.
   The `accumulated_provides` set is already maintained in `run_stream` and
   passed to `_exec_node`.

3. **IPU hooks**: `before_node`/`after_node` fire on every `_exec_node` call.
   Spawned nodes are just regular nodes — no special case.

4. **Telemetry**: Add `graph.spawn` event to EventBus:
   ```python
   await self.bus.emit("graph.spawn", br, control_node, {
       "spawned_count": len(new_nodes),
       "total_dynamic": self._total_spawned,
       "names": [s.get("name") for s in new_nodes],
   })
   ```

5. **Termination guarantee**: `max_dynamic_nodes` (default 100) hard ceiling.
   Configurable in `Runner.__init__`. Runner raises `ExecutionError` if a
   control node tries to exceed it.

6. **DAG validity**: Spawned nodes can only depend on nodes that exist (done
   or ready). Forward references to not-yet-spawned nodes are rejected at
   spawn time. This prevents cycles by construction — new nodes always point
   backward.

7. **Ctx isolation**: Spawned nodes get the same per-node write dict isolation
   as static nodes. No special case.

## Runner.__init__ Change

```python
def __init__(
    self,
    ipu=None,
    event_bus=None,
    max_concurrent=None,
    max_dynamic_nodes=100,     # NEW
):
    ...
    self.max_dynamic_nodes = max_dynamic_nodes
    self._total_spawned = 0    # reset per run() call
```

## OpGraph Changes

None. OpGraph.nodes is a mutable dict — adding entries at runtime is already
supported. `validate_dag()` is called once at start; spawned subgraphs don't
need re-validation because backward-only deps prevent cycles.

Add one helper:

```python
def add_node(self, node: OpNode) -> None:
    """Add a node to a live graph. Deps must reference existing nodes."""
    for d in node.deps:
        if d not in self.nodes:
            raise ValueError(f"Spawn: dependency {d} not in graph")
    self.nodes[node.id] = node
```

## ReAct as Dynamic Graph

### Compilation

```python
def compile_react_graph(operate_morphism, control_morphism, initial_params):
    """Create a one-round ReAct graph. Control node spawns next round if needed."""
    g = OpGraph()

    op_node = OpNode(m=operate_morphism, params=initial_params)
    ctrl_node = OpNode(
        m=control_morphism,
        deps={op_node.id},
        control=True,
    )

    g.nodes = {op_node.id: op_node, ctrl_node.id: ctrl_node}
    g.roots = {op_node.id}
    return g
```

### Control Morphism

```python
class ReactControlMorphism(Morphism):
    name = "react_control"
    requires = frozenset()
    provides = frozenset()

    max_rounds: int = 10
    operate_morphism: Morphism  # template for spawning next round
    current_round: int = 0

    async def apply(self, br, **kwargs):
        result = kwargs.get("result", {})
        actions = result.get("action_responses", [])
        self.current_round += 1

        if not actions or self.current_round >= self.max_rounds:
            return {"action": "halt", "reason": "ReAct complete"}

        # Spawn next round
        next_op = OpNode(m=self.operate_morphism, params={...})
        next_ctrl = OpNode(m=self, deps={next_op.id}, control=True)

        return {
            "action": "spawn",
            "nodes": [
                {"morphism": self.operate_morphism, "params": {...}},
                {"morphism": self, "deps": ["self_placeholder"], "control": True},
            ],
            "reason": f"ReAct round {self.current_round + 1}: {len(actions)} actions",
        }
```

### What this gives ReAct

- Each round is a visible node in telemetry — not buried in a `while` loop
- Runner policy_check fires per round — capability revocation mid-loop works
- Any control node can inject pause/human-review between rounds
- The ReAct subgraph can be a node in a larger workflow graph
- `max_dynamic_nodes` prevents infinite loops from hallucinated actions

## Other Patterns as Dynamic Graphs

### Self-Refine
```
[generate] → [critique (control)] → spawns [revise] → [critique (control)] → ...
```

### Tree of Thought
```
[generate] → [evaluate (control)] → spawns N × [generate] in parallel
                                   → [evaluate (control)] → selects best → spawns ...
```

### Debate
```
[agent_A] → [agent_B (sees A's result)] → [judge (control)]
    → spawns [agent_A (sees B's rebuttal)] → [agent_B] → [judge] → ...
```

## Implementation Order

1. Add `max_dynamic_nodes` to `Runner.__init__`, `_total_spawned` counter
   (reset at top of `run_stream`)
2. Add `spawn` case to `_apply_control_action`
3. Add `OpGraph.add_node()` helper
4. Add `graph.spawn` telemetry event
5. Add `on_spawn` observer in `_install_observers`
6. Write `ReactControlMorphism` — first consumer
7. Rewrite `operations/react.py` to compile ReAct as dynamic graph
8. Tests: spawn basic, spawn with deps, exceed limit, spawn in nested graph

## Non-Goals (This Phase)

- Dynamic node removal (nodes can be skipped but not deleted)
- Subgraph encapsulation (spawned nodes are flat in the parent graph)
- Cross-graph spawning (spawned nodes live in the same OpGraph)
- Streaming through spawned nodes (same limitation as static nodes)

## Risk

The main risk is a control morphism that spawns nodes which spawn more nodes
in an unbounded chain. `max_dynamic_nodes` is the hard ceiling. Additionally,
the `_total_spawned` counter is per-`run()` call, so a session-level Runner
resets between calls — no cross-call accumulation.

A subtler risk: spawned nodes that depend on each other in a cycle. Prevented
by the backward-only dep constraint — `add_node` rejects deps not already in
`g.nodes`. Since spawned nodes are added atomically (all in one control
action), circular deps within a spawn batch would need to be detected. The
simplest rule: within a spawn batch, deps are resolved in batch order — a node
can depend on a node earlier in the same batch, but not later.
