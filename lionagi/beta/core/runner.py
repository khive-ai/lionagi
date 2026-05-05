# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Runner: parallel DAG execution engine with wave-level atomicity.

Execution model:
    A "wave" is the maximal set of nodes whose dependencies are all satisfied.
    Nodes within a wave execute concurrently via asyncio.TaskGroup. If ANY
    node in a wave fails, the entire wave is cancelled and the error propagates.
    Successful waves are committed (ctx mutations persist). Failed waves leave
    ctx in an indeterminate state — callers should snapshot ctx before run()
    if rollback is needed.

Safety properties:
    - Policy is checked BEFORE any side effects (IPU, morphism, events)
    - IPU.after_node() is called in a finally block (even on failure)
    - Dynamic rights failures are DENIED (not silently fallen back to static)
    - Observer handlers are stored as strong refs to prevent GC
    - Morphism latency budgets are enforced via asyncio.wait_for() PRE-execution

Layering:
    Worker → WorkerEngine → Runner → IPU → Morphism
    Runner is the lowest execution layer. It runs a finite OpGraph one time.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from anyio import CapacityLimiter

from lionagi._errors import AccessError
from lionagi.ln.concurrency import create_task_group, fail_after
from lionagi.protocols.generic.eventbus import EventBus

from .graph import OpGraph, OpNode
from .ipu import IPU, LenientIPU, default_invariants
from .policy import policy_check
from .types import Observation, Principal

__all__ = ("Runner",)

try:
    _EGBase = BaseExceptionGroup
except NameError:
    from exceptiongroup import BaseExceptionGroup as _EGBase


def _unwrap_exception_group(exc: BaseException) -> BaseException:
    """Unwrap single-exception ExceptionGroups for clean catch patterns."""
    if isinstance(exc, _EGBase) and len(exc.exceptions) == 1:
        return exc.exceptions[0]
    return exc


class Runner:
    """Execute an OpGraph within a Principal context."""

    def __init__(
        self,
        ipu: IPU | None = None,
        event_bus: EventBus | None = None,
        max_concurrent: int | None = None,
        max_dynamic_nodes: int = 100,
    ) -> None:
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")
        self.ipu = ipu or LenientIPU(default_invariants())
        self.bus = event_bus or EventBus()
        self.max_concurrent = max_concurrent
        self.max_dynamic_nodes = max_dynamic_nodes
        # Strong refs to prevent GC when EventBus uses weakrefs (#9)
        self._handlers: list = []
        self._install_observers()

    def _install_observers(self) -> None:
        async def on_start(br: Principal, node: OpNode) -> None:
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="node.start",
                    payload={"node": str(node.id), "morphism": node.m.name},
                )
            )

        async def on_finish(br: Principal, node: OpNode, result: dict[str, Any]) -> None:
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="node.finish",
                    payload={"node": str(node.id), "keys": list(result.keys())},
                )
            )

        async def on_control(
            br: Principal,
            node: OpNode,
            action: dict[str, Any],
        ) -> None:
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="control.action",
                    payload={
                        "node": str(node.id),
                        "morphism": node.m.name,
                        "action": action.get("action"),
                        "targets": [str(i) for i in action.get("targets", [])],
                        "reason": action.get("reason", ""),
                    },
                )
            )

        async def on_spawn(br: Principal, control_node: OpNode, info: dict[str, Any]) -> None:
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="graph.spawn",
                    payload=info,
                )
            )

        self._handlers.extend([on_start, on_finish, on_control, on_spawn])
        self.bus.subscribe("node.start", on_start)
        self.bus.subscribe("node.finish", on_finish)
        self.bus.subscribe("control.action", on_control)
        self.bus.subscribe("graph.spawn", on_spawn)

    async def run(
        self,
        br: Principal,
        g: OpGraph,
    ) -> dict[UUID, dict[str, Any]]:
        results: dict[UUID, dict[str, Any]] = {}
        async for node_id, result in self.run_stream(br, g):
            results[node_id] = result
        return results

    async def run_stream(
        self,
        br: Principal,
        g: OpGraph,
    ) -> AsyncGenerator[tuple[UUID, dict[str, Any]], None]:
        """Execute graph and yield results as nodes complete; control nodes run serially after regular nodes per wave."""
        g.validate_dag()

        self._total_spawned = 0
        name_map: dict[str, UUID] = {}
        for nid, node in g.nodes.items():
            node_name = node.params.get("_lionagi_operation_name") or node.params.get("name")
            if node_name:
                name_map[node_name] = nid

        ready: set[UUID] = set(g.roots) if g.roots else {
            nid for nid, node in g.nodes.items() if not node.deps
        }
        done: set[UUID] = set()
        results: dict[UUID, dict[str, Any]] = {}
        accumulated_provides: set[str] = set()
        # Per-run effective reqs — never written to node.params to avoid cross-run contamination
        effective_reqs: dict[UUID, set[str] | None] = {}
        topo = g.validate_dag()

        while ready:
            batch = [
                g.nodes[n]
                for n in list(ready)
                if g.nodes[n].deps.issubset(done)
            ]
            if not batch:
                raise RuntimeError(
                    f"Runner stalled: ready={ready}, done={done}"
                )

            ready -= {n.id for n in batch}
            regular_batch = [node for node in batch if not node.control]
            control_batch = [node for node in batch if node.control]

            ctx_snapshot = dict(br.ctx)
            node_writes: dict[UUID, dict[str, Any]] = {}
            wave_results: dict[UUID, dict[str, Any]] = {}
            try:
                async for item in self._run_parallel_nodes(
                    br,
                    regular_batch,
                    wave_results,
                    ctx_snapshot,
                    accumulated_provides,
                    effective_reqs,
                    node_writes,
                ):
                    yield item
            except BaseException as exc:
                # Unwrap single-exception ExceptionGroups for clean catch
                unwrapped = _unwrap_exception_group(exc)
                if unwrapped is not exc:
                    raise unwrapped from unwrapped.__cause__
                raise

            control_actions: list[dict[str, Any]] = []
            halt_requested = False
            for node in control_batch:
                try:
                    await self._exec_node(
                        br,
                        node,
                        wave_results,
                        ctx_snapshot,
                        accumulated_provides,
                        effective_reqs,
                        node_writes,
                    )
                except BaseException as exc:
                    unwrapped = _unwrap_exception_group(exc)
                    if unwrapped is not exc:
                        raise unwrapped from unwrapped.__cause__
                    raise

                result = wave_results[node.id]
                yield node.id, result
                action = self._normalize_control_action(result, g, node)
                await self.bus.emit("control.action", br, node, action)
                control_actions.append(action)
                if action.get("action") == "halt":
                    halt_requested = True
                    break

            for nid in topo:
                if nid in node_writes:
                    br.ctx.update(node_writes[nid])

            results.update(wave_results)
            for nid in wave_results:
                done.add(nid)
                provides = getattr(g.nodes[nid].m, "provides", frozenset())
                if provides:
                    accumulated_provides.update(provides)

            if halt_requested:
                break

            for nid in wave_results:
                for cand_id, cand in g.nodes.items():
                    if nid in cand.deps and cand.deps.issubset(done):
                        ready.add(cand_id)

            for ca_idx, action in enumerate(control_actions):
                self._apply_control_action(action, g, ready, done, results, name_map)
                if action.get("action") == "spawn":
                    spawn_meta = action.get("metadata", {}).get("spawn_nodes", [])
                    ctrl_node = control_batch[ca_idx] if ca_idx < len(control_batch) else None
                    await self.bus.emit("graph.spawn", br, ctrl_node, {
                        "spawned_count": len(spawn_meta),
                        "total_dynamic": self._total_spawned,
                        "names": [s.get("name") for s in spawn_meta],
                    })

    async def _run_parallel_nodes(
        self,
        br: Principal,
        nodes: list[OpNode],
        results: dict[UUID, dict[str, Any]],
        ctx_snapshot: dict[str, Any],
        accumulated_provides: set[str],
        effective_reqs: dict[UUID, set[str] | None],
        node_writes: dict[UUID, dict[str, Any]],
    ) -> AsyncGenerator[tuple[UUID, dict[str, Any]], None]:
        if not nodes:
            return

        limiter = CapacityLimiter(self.max_concurrent) if self.max_concurrent else None
        completed: asyncio.Queue[tuple[UUID, dict[str, Any]]] = asyncio.Queue()

        async def _run_one(node: OpNode) -> None:
            if limiter is not None:
                async with limiter:
                    await self._exec_node(
                        br,
                        node,
                        results,
                        ctx_snapshot,
                        accumulated_provides,
                        effective_reqs,
                        node_writes,
                    )
            else:
                await self._exec_node(
                    br,
                    node,
                    results,
                    ctx_snapshot,
                    accumulated_provides,
                    effective_reqs,
                    node_writes,
                )
            await completed.put((node.id, results[node.id]))

        async with create_task_group() as tg:
            for node in nodes:
                tg.start_soon(_run_one, node)
            for _ in nodes:
                yield await completed.get()

    def _normalize_control_action(
        self,
        result: dict[str, Any],
        g: OpGraph,
        control_node: OpNode,
    ) -> dict[str, Any]:
        action = str(result.get("action", "proceed")).lower()
        if action == "abort":
            action = "halt"
        if action == "spawn":
            spawn_nodes = (
                result.get("nodes")
                or result.get("spawn_nodes")
                or []
            )
            return {
                "action": "spawn",
                "targets": [],
                "reason": str(result.get("reason", "")),
                "metadata": {"spawn_nodes": spawn_nodes},
            }
        targets = (
            result.get("targets")
            or result.get("target_ids")
            or result.get("nodes")
            or result.get("node_ids")
            or []
        )
        if isinstance(targets, (str, UUID)):
            targets = [targets]

        if action == "route":
            keep = {self._resolve_node_id(t, g) for t in targets}
            keep.discard(None)
            targets = [
                nid
                for nid, node in g.nodes.items()
                if control_node.id in node.deps and nid not in keep
            ]
            action = "skip"

        return {
            "action": action,
            "targets": [
                nid for target in targets
                if (nid := self._resolve_node_id(target, g)) is not None
            ],
            "reason": str(result.get("reason", "")),
            "metadata": result.get("metadata", {}),
        }

    def _resolve_node_id(self, value: Any, g: OpGraph) -> UUID | None:
        if isinstance(value, UUID):
            return value if value in g.nodes else None
        text = str(value)
        with contextlib.suppress(ValueError):
            node_id = UUID(text)
            if node_id in g.nodes:
                return node_id
        for node_id, node in g.nodes.items():
            if node.params.get("_lionagi_operation_name") == text:
                return node_id
            if node.params.get("name") == text:
                return node_id
        return None

    def _apply_control_action(
        self,
        action: dict[str, Any],
        g: OpGraph,
        ready: set[UUID],
        done: set[UUID],
        results: dict[UUID, dict[str, Any]],
        name_map: dict[str, UUID] | None = None,
    ) -> None:
        action_name = action.get("action")
        targets = set(action.get("targets") or [])

        if action_name == "skip":
            ready.difference_update(targets)
            return

        if action_name == "retry":
            for node_id in targets:
                if node_id not in g.nodes:
                    continue
                done.discard(node_id)
                results.pop(node_id, None)
                if g.nodes[node_id].deps.issubset(done):
                    ready.add(node_id)

        if action_name == "spawn":
            new_nodes = action.get("metadata", {}).get("spawn_nodes") or action.get("targets", [])
            if not new_nodes:
                return

            if self._total_spawned + len(new_nodes) > self.max_dynamic_nodes:
                from lionagi._errors import ExecutionError
                raise ExecutionError(
                    f"Dynamic node limit ({self.max_dynamic_nodes}) exceeded: "
                    f"tried to spawn {len(new_nodes)}, already spawned {self._total_spawned}",
                    retryable=False,
                )

            nm = name_map if name_map is not None else {}

            for spec in new_nodes:
                morphism = spec.get("morphism")
                if morphism is None:
                    raise ValueError("spawn spec missing 'morphism'")

                # Resolve deps — can be UUIDs, UUID strings, or names
                raw_deps = spec.get("deps", [])
                resolved_deps: set[UUID] = set()
                for d in raw_deps:
                    if isinstance(d, UUID):
                        resolved_deps.add(d)
                    else:
                        resolved = self._resolve_node_id(d, g)
                        if resolved is None:
                            # Check name_map for nodes added earlier in this batch
                            resolved = nm.get(str(d))
                        if resolved is None:
                            raise ValueError(f"spawn: unresolved dependency '{d}'")
                        resolved_deps.add(resolved)

                node = OpNode(
                    m=morphism,
                    deps=resolved_deps,
                    params=spec.get("params", {}),
                    control=spec.get("control", False),
                )

                node_name = spec.get("name")
                if node_name:
                    node.params["_lionagi_operation_name"] = node_name
                    nm[node_name] = node.id

                g.add_node(node)

                if node.deps.issubset(done):
                    ready.add(node.id)

            self._total_spawned += len(new_nodes)

    async def _exec_node(
        self,
        br: Principal,
        node: OpNode,
        results: dict[UUID, dict[str, Any]],
        ctx_snapshot: dict[str, Any] | None = None,
        accumulated_provides: set[str] | None = None,
        effective_reqs_map: dict[UUID, set[str] | None] | None = None,
        node_writes: dict[UUID, dict[str, Any]] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = dict(ctx_snapshot if ctx_snapshot is not None else br.ctx)
        kwargs.update(node.params)

        # Dynamic rights: failure is DENIAL, not fallback to static rights
        override_reqs: set[str] | None = None
        req_fn = getattr(node.m, "required_rights", None)
        if callable(req_fn):
            try:
                r = req_fn(**kwargs)
                if r:
                    override_reqs = set(r)
            except Exception as e:
                raise AccessError(
                    f"Dynamic rights computation failed for '{node.m.name}': {e}",
                    retryable=False,
                ) from e

        if effective_reqs_map is not None:
            effective_reqs_map[node.id] = override_reqs

        extra = frozenset(accumulated_provides) if accumulated_provides else None

        if not policy_check(br, node.m, override_reqs=override_reqs, extra_rights=extra):
            raise AccessError(
                f"Policy denied for morphism '{node.m.name}' in principal '{br.name}'",
                retryable=False,
            )

        budget_ms = getattr(node.m, "latency_budget_ms", None)
        timeout = budget_ms / 1000.0 if budget_ms else None
        local_ctx = {} if node_writes is not None else None
        res: dict[str, Any] = {}
        error: BaseException | None = None
        try:
            await self.ipu.before_node(br, node)
            await self.bus.emit("node.start", br, node)

            ok_pre = await node.m.pre(br, **kwargs)
            if not ok_pre:
                raise AssertionError(
                    f"Morphism '{node.m.name}' pre() returned False"
                )

            if timeout:
                with fail_after(timeout):
                    res = await node.m.apply(br, **kwargs)
            else:
                res = await node.m.apply(br, **kwargs)
            if not isinstance(res, dict):
                res = {"result": res}

            if local_ctx is not None and isinstance(res, dict):
                local_ctx.update(res)

            ok_post = await node.m.post(br, res)
            if not ok_post:
                raise AssertionError(
                    f"Morphism '{node.m.name}' post() returned False"
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                await self.ipu.after_node(br, node, res, error=error)
            except Exception:
                if error is None:
                    raise
                # Primary error takes precedence — don't mask it with IPU cleanup failure
                import logging
                logging.getLogger(__name__).warning(
                    "IPU after_node raised during error cleanup for '%s'",
                    node.m.name, exc_info=True,
                )
            if error is None:
                await self.bus.emit("node.finish", br, node, res)
            else:
                await self.bus.emit(
                    "node.error", br, node,
                    {"error": str(error), "type": type(error).__name__},
                )

        if node_writes is not None:
            node_writes[node.id] = local_ctx or {}
        results[node.id] = res
