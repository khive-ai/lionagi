from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from lionagi.core.morphism import Morphism
from lionagi.core.types import Principal
from lionagi.ln.concurrency import retry

MORPH_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Register morphism combinators by declared name."""
    MORPH_REGISTRY[getattr(cls, "name", cls.__name__)] = cls
    return cls


class BaseOp:
    """Structural base for Runner-native morphism combinators.

    Operations are session handlers that receive ``(params, RequestContext)``.
    Morph ops are lower-level executable decorators around ``Morphism`` objects;
    they live here so they cannot be confused with product operations.
    """

    name = "base"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    io: bool = False
    latency_budget_ms: int | None = None
    result_keys: set[str] | None = None
    result_schema = None
    ctx_writes: set[str] | None = None
    result_bytes_limit: int | None = None

    async def pre(self, br: Principal, **kw) -> bool:
        return True

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__}.apply() is not implemented")

    async def post(self, br: Principal, res: dict) -> bool:
        return True


def inherit_contract(target: BaseOp, inner: Any) -> None:
    target.requires = frozenset(getattr(inner, "requires", frozenset()))
    target.provides = frozenset(getattr(inner, "provides", frozenset()))
    target.io = bool(getattr(inner, "io", False))
    target.ctx_writes = getattr(inner, "ctx_writes", None)
    target.result_schema = getattr(inner, "result_schema", None)
    target.result_keys = getattr(inner, "result_keys", None)
    target.result_bytes_limit = getattr(inner, "result_bytes_limit", None)
    target.latency_budget_ms = getattr(inner, "latency_budget_ms", None)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@register
class CtxSet(BaseOp):
    """Return declared context writes for Runner's isolated ctx merge."""

    name = "ctx.set"

    def __init__(self, values: dict[str, Any], allowed_keys: set[str]):
        self.values = dict(values)
        self.ctx_writes = set(allowed_keys)
        self.requires = frozenset()
        self.provides = frozenset(self.values)

    async def pre(self, br: Principal, **kw) -> bool:
        return set(self.values.keys()).issubset(self.ctx_writes or set())

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        return dict(self.values)

    async def post(self, br: Principal, res: dict) -> bool:
        return all(res.get(k) == v for k, v in self.values.items())


@register
class SubgraphRun(BaseOp):
    """Run a nested OpGraph inside the same Principal."""

    name = "subgraph.run"

    def __init__(
        self,
        graph,
        *,
        runner=None,
        ipu=None,
        max_concurrent: int | None = None,
    ):
        from lionagi.core.graph import OpGraph

        if not isinstance(graph, OpGraph):
            raise ValueError("SubgraphRun requires an OpGraph instance")
        self.graph = graph
        self.runner = runner
        self.ipu = ipu
        self.max_concurrent = max_concurrent
        self.requires = frozenset({"graph.run"})
        self.provides = frozenset({"graph.result"})
        self.result_keys = {"ok", "results"}

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        from lionagi.core.runner import Runner

        runner = self.runner or Runner(
            ipu=self.ipu,
            max_concurrent=self.max_concurrent,
        )
        results = await runner.run(br, self.graph)
        return {"ok": True, "results": results}


@register
class WithRetry(BaseOp):
    """Retry a morphism's apply() using production structured-concurrency retry."""

    name = "with.retry"

    def __init__(
        self,
        inner: Morphism,
        attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 2.0,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        jitter: float = 0.1,
    ):
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self.inner = inner
        self.attempts = int(attempts)
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.retry_on = retry_on
        self.jitter = float(jitter)
        inherit_contract(self, inner)

    async def pre(self, br: Principal, **kw) -> bool:
        return bool(await maybe_await(self.inner.pre(br, **kw)))

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        return await retry(
            lambda: self.inner.apply(br, **kw),
            attempts=self.attempts,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            retry_on=self.retry_on,
            jitter=self.jitter,
        )

    async def post(self, br: Principal, res: dict) -> bool:
        return bool(await maybe_await(self.inner.post(br, res)))


@register
class WithTimeout(BaseOp):
    """Attach a Runner latency budget to a morphism."""

    name = "with.timeout"

    def __init__(self, inner: Morphism, timeout_ms: int):
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be >= 1")
        self.inner = inner
        self.timeout_ms = int(timeout_ms)
        inherit_contract(self, inner)
        self.latency_budget_ms = self.timeout_ms

    async def pre(self, br: Principal, **kw) -> bool:
        return bool(await maybe_await(self.inner.pre(br, **kw)))

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        return await maybe_await(self.inner.apply(br, **kw))

    async def post(self, br: Principal, res: dict) -> bool:
        return bool(await maybe_await(self.inner.post(br, res)))


@register
class OpThenPatch(BaseOp):
    """Expose selected result keys as Runner context writes."""

    name = "op.then_patch"

    def __init__(self, inner: Morphism, patch: Mapping[str, str] | Iterable[str]):
        self.inner = inner
        if isinstance(patch, dict):
            self.patch_map = dict(patch)
        else:
            self.patch_map = {k: k for k in patch}
        inherit_contract(self, inner)
        self.ctx_writes = set(self.patch_map.values())

    async def pre(self, br: Principal, **kw) -> bool:
        return bool(await maybe_await(self.inner.pre(br, **kw)))

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        res = await maybe_await(self.inner.apply(br, **kw))
        out = dict(res)
        for src_key, dst_key in self.patch_map.items():
            if src_key in res:
                out[dst_key] = res[src_key]
        return out

    async def post(self, br: Principal, result: dict[str, Any]) -> bool:
        return bool(await maybe_await(self.inner.post(br, result)))
