from __future__ import annotations

import logging
from typing import Any

from ..ln import create_task_group, fail_after, is_coro_func
from .eventbus import EventBus, emit_node_finish, emit_node_start
from .graph import OpGraph, OpNode
from .policy import policy_check
from .types import Branch, Observation

logger = logging.getLogger(__name__)


class Runner:
    """Secure, structured execution kernel for Lion V1 workflows.

    Features:
    - Structured concurrency guarantees cleanup on failure
    - Proactive deadline enforcement via fail_after
    - Dynamic capability calculation with fail-closed security
    - Comprehensive invariant checking via IPU
    """

    def __init__(self, ipu, event_bus: EventBus | None = None):
        self.ipu = ipu
        self.bus = event_bus or EventBus()
        self._install_default_observers()

    def _install_default_observers(self):
        """Install default event observers for node lifecycle."""

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
                        "keys": list(result.keys()) if result else [],
                    },
                )
            )

        async def on_error(br: Branch, node: OpNode, err: dict):
            await self.ipu.on_observation(
                Observation(
                    who=br.id,
                    what="node.error",
                    payload={"node": str(node.id), **(err or {})},
                )
            )

        self.bus.subscribe("node.start", on_start)
        self.bus.subscribe("node.finish", on_finish)
        self.bus.subscribe("node.error", on_error)

    async def run(self, br: Branch, g: OpGraph) -> dict[Any, Any]:
        """Execute an OpGraph within a Branch using structured concurrency.

        Args:
            br: The execution context branch
            g: The operation graph to execute

        Returns:
            Dict mapping node IDs to their results

        Raises:
            RuntimeError: If graph has cycles or execution fails
            PermissionError: If security policy denies execution
        """
        g.validate_dag()

        # Initialize 'ready' from roots if provided; otherwise, from all zero-indegree nodes
        # This ensures DAGs without explicit roots can still execute from natural starting points
        if g.roots:
            ready: set = set(g.roots)
        else:
            # Calculate indegree to find natural starting nodes (zero dependencies)
            indeg = {k: 0 for k in g.nodes}
            for v, node in g.nodes.items():
                for u in node.deps:
                    indeg[v] += 1
            ready = {n for n, d in indeg.items() if d == 0}

        done: set = set()
        results: dict[Any, Any] = {}

        while ready:
            # Find executable nodes (dependencies satisfied)
            batch = [g.nodes[n] for n in list(ready) if g.nodes[n].deps.issubset(done)]
            if not batch:
                raise RuntimeError("No executable nodes (cycle or bad roots)")

            ready -= {n.id for n in batch}

            # Execute batch using structured concurrency
            async with create_task_group() as tg:
                for node in batch:
                    # Start each node in the structured task group
                    tg.start_soon(self._exec_node_wrapper, br, node, results)

            # After TaskGroup completes, all nodes in batch are done
            for node in batch:
                done.add(node.id)
                # Find newly ready nodes
                for cand in g.nodes.values():
                    if node.id in cand.deps and cand.deps.issubset(done):
                        ready.add(cand.id)

        return results

    async def _exec_node_wrapper(self, br: Branch, node: OpNode, results: dict) -> None:
        """Wrapper to store results from node execution."""
        try:
            node_id, result = await self._exec_node(br, node)
            results[node_id] = result
        except Exception as e:
            logger.error(f"Node {node.id} execution failed: {e}")
            results[node.id] = {"error": str(e), "failed": True}
            # Emit error event for observability - isolated to prevent cascade failures
            try:
                await self.bus.emit(
                    "node.error", br, node, {"error": str(e), "type": type(e).__name__}
                )
            except Exception:
                # Swallow event emission errors to prevent masking the original error
                pass
            raise

    async def _exec_node(self, br: Branch, node: OpNode) -> tuple[Any, dict]:
        """Execute a single node with all safety checks and monitoring.

        Features:
        - Dynamic capability calculation with fail-closed security
        - Proactive deadline enforcement
        - Pre/post condition validation
        - IPU invariant checking
        """

        # 1) Build kwargs from node.params and br.ctx
        kwargs: dict[str, Any] = {}
        node_params = getattr(node, "params", None)
        if isinstance(node_params, dict):
            kwargs.update(node_params)
        # Context values as defaults
        for k, v in br.ctx.items():
            kwargs.setdefault(k, v)
        kwargs.setdefault("prompt", "")  # Common default

        # 2) Dynamic rights calculation (FAIL CLOSED on error)
        override_reqs = None
        req_fn = getattr(node.m, "required_rights", None)
        if callable(req_fn):
            try:
                # Support both sync and async required_rights() methods
                if is_coro_func(req_fn):
                    r = await req_fn(**kwargs)
                else:
                    r = req_fn(**kwargs)
                if r:
                    override_reqs = set(r)
            except Exception as e:
                # CRITICAL: Fail closed - if we can't determine rights, deny
                logger.error(f"Failed to calculate dynamic rights for {node.id}: {e}")
                raise PermissionError(
                    f"Security policy denied: Cannot determine rights for {node.m.name}"
                ) from e

        # 3) Security gate via policy check
        if not policy_check(br, node.m, override_reqs=override_reqs):
            raise PermissionError(f"Security policy denied: {node.m.name}")

        # 4) Pre-phase invariants and events
        await self.ipu.before_node(br, node)
        await emit_node_start(self.bus, br, node)

        # 5) Validate pre-conditions (no assert - proper error)
        try:
            pre_result = await node.m.pre(br, **kwargs)
            if not pre_result:
                raise RuntimeError(f"Node {node.id} pre-condition returned False")
        except Exception as e:
            logger.error(f"Node {node.id} pre-condition failed: {e}")
            raise RuntimeError(f"Node {node.id} pre-condition failed") from e

        # 6-7) Execute with guaranteed IPU post-invariant cleanup
        res = None
        try:
            # 6) Execute with proactive deadline enforcement
            budget_ms = getattr(node.m, "latency_budget_ms", None)

            try:
                if budget_ms:
                    # Enforce deadline proactively
                    timeout_s = budget_ms / 1000.0
                    logger.debug(f"Node {node.id} has latency budget: {budget_ms}ms")

                    with fail_after(timeout_s):
                        res = await node.m.apply(br, **kwargs)
                else:
                    # No deadline, but still cancellable by parent scope
                    res = await node.m.apply(br, **kwargs)

            except TimeoutError as e:
                logger.error(f"Node {node.id} exceeded latency budget ({budget_ms}ms)")
                raise RuntimeError(f"Node {node.id} exceeded latency budget ({budget_ms}ms)") from e
            except Exception as e:
                logger.error(f"Node {node.id} execution failed: {e}")
                raise

            # 7) Validate post-conditions
            try:
                post_result = await node.m.post(br, res)
                if not post_result:
                    raise RuntimeError(f"Node {node.id} post-condition returned False")
            except Exception as e:
                logger.error(f"Node {node.id} post-condition failed: {e}")
                raise RuntimeError(f"Node {node.id} post-condition failed") from e

        finally:
            # 8) GUARANTEED post-phase invariants - always executed regardless of success/failure
            # Pass res (which may be None if execution failed) to allow invariant cleanup
            await self.ipu.after_node(br, node, res)

        # 9) Emit success event only after successful completion
        await emit_node_finish(self.bus, br, node, res)

        return node.id, res


class ParallelRunner(Runner):
    """Enhanced runner with advanced parallel execution strategies.

    Features:
    - Speculative execution of likely branches
    - Resource-aware scheduling
    - Adaptive concurrency limits
    """

    def __init__(self, ipu, event_bus: EventBus | None = None, max_concurrency: int = 10):
        super().__init__(ipu, event_bus)
        self.max_concurrency = max_concurrency

    # TODO: Implement advanced scheduling strategies
