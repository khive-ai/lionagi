from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import msgspec
from anyio import Path

from ..base import Branch, Morphism, StrictIPU, default_invariants, register
from ..ln import fail_after, retry

# ----- BaseOp -------------------------------------------------------------


class BaseOp:
    """Convenience base that defines the attributes IPU looks for."""

    name = "base"
    requires: set[str] = set()
    io: bool = False
    latency_budget_ms: int | None = None
    result_keys: set[str] | None = None
    result_schema = None
    ctx_writes: set[str] | None = None
    result_bytes_limit: int | None = None

    async def pre(self, br: Branch, **kw) -> bool:
        return True

    async def post(self, br: Branch, res: dict) -> bool:
        return True


# ----- LLM ---------------------------------------------------------------


class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...


class TextOut(msgspec.Struct, kw_only=True):
    text: str


@register
class LLMGenerate(BaseOp):
    """Provider-agnostic LLM generation."""

    name = "llm.generate"
    io = True
    result_schema = TextOut
    ctx_writes = {"last_llm"}

    def __init__(
        self,
        provider: LLMProvider,
        host: str = "*",
        latency_budget_ms: int = 2000,
    ):
        self.provider = provider
        self.requires = {f"net.out:{host}"}
        self.latency_budget_ms = latency_budget_ms

    async def pre(self, br: Branch, **kw) -> bool:
        return "prompt" in kw and isinstance(kw["prompt"], str)

    async def apply(self, br: Branch, **kw) -> dict:
        text = await self.provider.generate(kw["prompt"])
        br.ctx["last_llm"] = text
        return {"text": text}

    async def post(self, br: Branch, res: dict) -> bool:
        return "text" in res and isinstance(res["text"], str)


# ----- HTTP (pluggable client) ------------------------------------------


class HttpClient(Protocol):
    async def get(self, url: str) -> tuple[int, str]: ...


@register
class HTTPGet(BaseOp):
    name = "http.get"
    io = True
    result_keys = {"status", "body"}

    def __init__(
        self,
        client: HttpClient,
        host: str = "*",
        result_bytes_limit: int = 1_000_000,
        latency_budget_ms: int = 5000,
    ):
        self.client = client
        self.requires = {f"net.out:{host}"}  # static default (used if URL parsing fails)
        self.result_bytes_limit = result_bytes_limit
        self.latency_budget_ms = latency_budget_ms

    def required_rights(self, **kw) -> set[str]:
        # derive host from URL to enforce per-request host capability
        try:
            from urllib.parse import urlparse

            url = kw.get("url", "")
            host = urlparse(url).netloc or "*"
            return {f"net.out:{host}"}
        except Exception:
            return set(self.requires)

    async def pre(self, br: Branch, **kw) -> bool:
        return "url" in kw and isinstance(kw["url"], str)

    async def apply(self, br: Branch, **kw) -> dict:
        status, body = await self.client.get(kw["url"])
        return {"status": int(status), "body": str(body)}

    async def post(self, br: Branch, res: dict) -> bool:
        return "status" in res and "body" in res


# ----- FS ----------------------------------------------------------------
@register
class FSRead(BaseOp):
    name = "fs.read"
    result_keys = {"path", "data"}

    def __init__(self, allow_pattern: str = "/*", result_bytes_limit: int = 10_000_000):
        """
        allow_pattern: e.g., "/tmp/*" for conservative policy coverage (static fallback).
        """
        self.requires = {f"fs.read:{allow_pattern}"}
        self.result_bytes_limit = result_bytes_limit

    def required_rights(self, **kw) -> set[str]:
        # Require the specific path being read (concrete resource), normalized.
        # Note: Using pathlib for path resolution in sync context is acceptable here
        from pathlib import Path

        path = kw.get("path")
        if not isinstance(path, str):
            return set(self.requires)
        try:
            # resolve() to collapse symlinks/.. and make absolute (macOS /private/tmp etc.)
            p = str(Path(path).expanduser().resolve())
        except Exception:
            p = str(Path(path).expanduser())
        return {f"fs.read:{p}"}

    async def pre(self, br: Branch, **kw) -> bool:
        return "path" in kw and isinstance(kw["path"], str)

    async def apply(self, br: Branch, **kw) -> dict:
        # Use anyio.Path for non-blocking I/O to prevent event loop blocking
        p = Path(kw["path"]).expanduser()
        data = await p.read_text(encoding="utf-8")
        return {"path": str(p), "data": data}


# ----- KV ----------------------------------------------------------------


@dataclass
class InMemoryKV:
    store: dict[str, dict[str, Any]]

    def __init__(self):
        self.store = {}

    def ensure_ns(self, ns: str) -> dict[str, Any]:
        return self.store.setdefault(ns, {})


@register
class KVSet(BaseOp):
    name = "kv.set"

    def __init__(self, kv: InMemoryKV, ns: str):
        self.kv = kv
        self.ns = ns
        self.requires = {f"kv.write:{ns}"}
        self.result_keys = {"key", "ok"}

    async def pre(self, br: Branch, **kw) -> bool:
        return "key" in kw and "value" in kw

    async def apply(self, br: Branch, **kw) -> dict:
        ns = self.kv.ensure_ns(self.ns)
        ns[kw["key"]] = kw["value"]
        return {"key": kw["key"], "ok": True}


@register
class KVGet(BaseOp):
    name = "kv.get"

    def __init__(self, kv: InMemoryKV, ns: str):
        self.kv = kv
        self.ns = ns
        self.requires = {f"kv.read:{ns}"}
        self.result_keys = {"key", "value"}

    async def pre(self, br: Branch, **kw) -> bool:
        return "key" in kw

    async def apply(self, br: Branch, **kw) -> dict:
        ns = self.kv.ensure_ns(self.ns)
        return {"key": kw["key"], "value": ns.get(kw["key"])}


# ----- Ctx.Set -----------------------------------------------------------


@register
class CtxSet(BaseOp):
    name = "ctx.set"

    def __init__(self, values: dict[str, Any], allowed_keys: set[str]):
        self.values = dict(values)
        self.ctx_writes = set(allowed_keys)
        self.requires = set()  # no IO

    async def pre(self, br: Branch, **kw) -> bool:
        # only allow writing declared keys
        return set(self.values.keys()).issubset(self.ctx_writes or set())

    async def apply(self, br: Branch, **kw) -> dict:
        br.ctx.update(self.values)
        return {"ok": True}

    async def post(self, br: Branch, res: dict) -> bool:
        return res.get("ok", False)


# ----- Subgraph.Run ------------------------------------------------------


@register
class SubgraphRun(BaseOp):
    """Run a nested OpGraph inside the same Branch."""

    name = "subgraph.run"

    def __init__(self, graph, ipu=None):
        from lionagi.base.graph import OpGraph  # local import to avoid cycles

        if not isinstance(graph, OpGraph):
            raise ValueError("SubgraphRun requires an OpGraph instance")
        self.graph = graph
        self.requires = {"graph.run"}
        self.result_keys = {"ok"}
        self.ipu = ipu  # allow injecting a custom IPU (defaults to StrictIPU)

    async def pre(self, br: Branch, **kw) -> bool:
        return True

    async def apply(self, br: Branch, **kw) -> dict:
        # Import here to avoid import cycle at module load
        from lionagi.base.runner import Runner

        ipu = self.ipu or StrictIPU(default_invariants())
        r = Runner(ipu=ipu)
        await r.run(br, self.graph)
        return {"ok": True}


# ----- Wrappers ----------------------------------------------------------


@register
class WithRetry(BaseOp):
    name = "with.retry"

    def __init__(
        self,
        inner: Morphism,
        retries: int = 3,
        backoff_ms: int = 100,
        jitter: bool = True,
    ):
        self.inner = inner
        self.retries = int(retries)
        self.backoff_ms = int(backoff_ms)
        self.jitter = bool(jitter)
        # inherit attrs
        self.requires = set(getattr(inner, "requires", set()))
        self.io = bool(getattr(inner, "io", False))
        self.ctx_writes = getattr(inner, "ctx_writes", None)
        self.result_schema = getattr(inner, "result_schema", None)
        self.result_keys = getattr(inner, "result_keys", None)
        self.result_bytes_limit = getattr(inner, "result_bytes_limit", None)
        self.latency_budget_ms = getattr(inner, "latency_budget_ms", None)

    async def pre(self, br: Branch, **kw) -> bool:
        return await self.inner.pre(br, **kw)

    async def apply(self, br: Branch, **kw) -> dict:
        # Use deadline-aware retry instead of manual loop with asyncio.sleep
        # This respects ambient timeouts and latency budgets
        async def attempt():
            return await self.inner.apply(br, **kw)

        return await retry(
            attempt,
            attempts=self.retries + 1,
            base_delay=self.backoff_ms / 1000.0,
            jitter=0.5 if self.jitter else 0.0,
        )

    async def post(self, br: Branch, res: dict) -> bool:
        return await self.inner.post(br, res)


@register
class WithTimeout(BaseOp):
    name = "with.timeout"

    def __init__(self, inner: Morphism, timeout_ms: int):
        self.inner = inner
        self.timeout_s = timeout_ms / 1000.0
        # inherit attrs
        self.requires = set(getattr(inner, "requires", set()))
        self.io = bool(getattr(inner, "io", False))
        self.ctx_writes = getattr(inner, "ctx_writes", None)
        self.result_schema = getattr(inner, "result_schema", None)
        self.result_keys = getattr(inner, "result_keys", None)
        self.result_bytes_limit = getattr(inner, "result_bytes_limit", None)
        self.latency_budget_ms = getattr(inner, "latency_budget_ms", None)

    async def pre(self, br: Branch, **kw) -> bool:
        return await self.inner.pre(br, **kw)

    async def apply(self, br: Branch, **kw) -> dict:
        # Use structured concurrency timeout instead of unsafe asyncio.wait_for
        with fail_after(self.timeout_s):
            return await self.inner.apply(br, **kw)

    async def post(self, br: Branch, res: dict) -> bool:
        return await self.inner.post(br, res)
