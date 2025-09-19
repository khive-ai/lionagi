from __future__ import annotations

import asyncio
from typing import Any, Dict, Set

from .eventbus import EventBus, emit_node_finish, emit_node_start
from .graph import OpGraph, OpNode
from .policy import policy_check
from .types import Branch, Observation


class Runner:
    def __init__(self, ipu, event_bus: EventBus | None = None):
        self.ipu = ipu
        self.bus = event_bus or EventBus()
        self._install_default_observers()

    def _install_default_observers(self):
        async def on_start(br: Branch, node: OpNode):
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="node.start",
                    payload={"node": str(node.id)},
                )
            )

        async def on_finish(br: Branch, node: OpNode, result: dict):
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="node.finish",
                    payload={
                        "node": str(node.id),
                        "keys": list(result.keys()),
                    },
                )
            )

        self.bus.subscribe("node.start", on_start)
        self.bus.subscribe("node.finish", on_finish)

    async def run(self, br: Branch, g: OpGraph):
        g.validate_dag()
        ready: Set = set(g.roots)
        done: Set = set()
        results: Dict[Any, Any] = {}

        while ready:
            batch = [
                g.nodes[n]
                for n in list(ready)
                if g.nodes[n].deps.issubset(done)
            ]
            if not batch:
                raise RuntimeError("No executable nodes (cycle or bad roots)")
            ready -= {n.id for n in batch}
            tasks = [
                asyncio.create_task(self._exec_node(br, n, results))
                for n in batch
            ]
            finished = await asyncio.gather(*tasks)
            for nid, _ in finished:
                done.add(nid)
                for cand in g.nodes.values():
                    if nid in cand.deps:
                        ready.add(cand.id)
        return results

    async def _exec_node(self, br: Branch, node: OpNode, results: dict):
        # 1) Build kwargs from node.params and br.ctx (params take precedence)
        kwargs: Dict[str, Any] = {}
        node_params = getattr(node, "params", None)
        if isinstance(node_params, dict):
            kwargs.update(node_params)
        for k, v in br.ctx.items():
            kwargs.setdefault(k, v)
        kwargs.setdefault(
            "prompt", ""
        )  # common default to satisfy simple pre()s

        # 2) Compute dynamic rights (if morphism exposes required_rights(**kwargs))
        override_reqs = None
        req_fn = getattr(node.m, "required_rights", None)
        if callable(req_fn):
            try:
                r = req_fn(**kwargs)
                if r:
                    override_reqs = set(r)
            except Exception:
                override_reqs = None  # fall back to static requires

        # 3) Gate by policy (using dynamic rights when available)
        if not policy_check(br, node.m, override_reqs=override_reqs):
            raise PermissionError(f"Policy denied: {node.m.name}")

        # 4) Pre-phase invariants
        await self.ipu.before_node(br, node)
        await emit_node_start(self.bus, br, node)

        # 5) Execute morphism
        assert await node.m.pre(br, **kwargs)
        res = await node.m.apply(br, **kwargs)
        assert await node.m.post(br, res)

        # 6) Post-phase invariants + observation
        await self.ipu.after_node(br, node, res)
        await emit_node_finish(self.bus, br, node, res)

        results[node.id] = res
        return node.id, res
