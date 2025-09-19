import asyncio

import pytest

from lionagi_v1.base.graph import OpGraph, OpNode
from lionagi_v1.base.ipu import StrictIPU, default_invariants
from lionagi_v1.base.runner import Runner
from lionagi_v1.base.types import Branch, Capability
from lionagi_v1.ops.core import (
    BaseOp,
    CtxSet,
    FSRead,
    HTTPGet,
    InMemoryKV,
    KVGet,
    KVSet,
    LLMGenerate,
    SubgraphRun,
    WithRetry,
    WithTimeout,
)

# ---------------- Stubs ----------------


class StubLLM:
    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(0.01)
        return f"echo:{prompt}"


class StubHTTP:
    async def get(self, url: str):
        await asyncio.sleep(0.01)
        return 200, f"body({url})"


class SleepOp(BaseOp):
    name = "sleep.op"
    requires = set()

    def __init__(self, delay):
        self.delay = delay

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        await asyncio.sleep(self.delay)
        return {"slept": self.delay}

    async def post(self, br, res):
        return "slept" in res


class FailNTimes(BaseOp):
    name = "fail.ntimes"
    requires = set()

    def __init__(self, n):
        self.n = n

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        if self.n > 0:
            self.n -= 1
            raise RuntimeError("boom")
        return {"ok": True}

    async def post(self, br, res):
        return res.get("ok", False)


# ---------------- Tests ----------------


@pytest.mark.asyncio
async def test_llm_generate_allow_and_deny():
    br_ok = Branch(name="llm-ok")
    br_ok.caps = (Capability(subject=br_ok.id, rights={"net.out:*"}),)
    n = OpNode(m=LLMGenerate(StubLLM(), host="*"), params={"prompt": "hi"})
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br_ok, g)
    assert br_ok.ctx["last_llm"].startswith("echo:")

    br_deny = Branch(name="llm-deny")
    n2 = OpNode(
        m=LLMGenerate(StubLLM(), host="example.com"), params={"prompt": "x"}
    )
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    with pytest.raises(PermissionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br_deny, g2)


@pytest.mark.asyncio
async def test_http_get_stub_client():
    br = Branch(name="http")
    br.caps = (Capability(subject=br.id, rights={"net.out:example.com"}),)
    op = HTTPGet(StubHTTP(), host="example.com")
    n = OpNode(m=op, params={"url": "https://example.com/x"})
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    res = await Runner(ipu=StrictIPU(default_invariants())).run(br, g)
    out = list(res.values())[0]
    assert out["status"] == 200 and "body(" in out["body"]

    br2 = Branch(name="http-deny")
    n2 = OpNode(
        m=HTTPGet(StubHTTP(), host="example.com"),
        params={"url": "https://example.com/y"},
    )
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    with pytest.raises(PermissionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br2, g2)


@pytest.mark.asyncio
async def test_fs_read_enforced(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("hello", encoding="utf-8")

    allow_pattern = str(tmp_path) + "/*"
    br = Branch(name="fs")
    br.caps = (Capability(subject=br.id, rights={f"fs.read:{allow_pattern}"}),)

    n = OpNode(m=FSRead(allow_pattern=allow_pattern), params={"path": str(p)})
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    res = await Runner(ipu=StrictIPU(default_invariants())).run(br, g)
    assert list(res.values())[0]["data"] == "hello"

    # deny when pattern does not match
    br_deny = Branch(name="fs-deny")
    br_deny.caps = (
        Capability(subject=br_deny.id, rights={"fs.read:/notmp/*"}),
    )
    n2 = OpNode(m=FSRead(allow_pattern="/notmp/*"), params={"path": str(p)})
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    with pytest.raises(PermissionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br_deny, g2)


@pytest.mark.asyncio
async def test_kv_set_get_rights():
    kv = InMemoryKV()
    br = Branch(name="kv")
    br.caps = (
        Capability(subject=br.id, rights={"kv.write:ns", "kv.read:ns"}),
    )

    set_node = OpNode(m=KVSet(kv, ns="ns"), params={"key": "k", "value": 42})
    get_node = OpNode(
        m=KVGet(kv, ns="ns"), deps={set_node.id}, params={"key": "k"}
    )
    g = OpGraph(
        nodes={set_node.id: set_node, get_node.id: get_node},
        roots={set_node.id},
    )
    res = await Runner(ipu=StrictIPU(default_invariants())).run(br, g)
    out = res[get_node.id]
    assert out["value"] == 42

    # denial on missing read right
    br2 = Branch(name="kv-deny")
    br2.caps = (Capability(subject=br2.id, rights={"kv.write:ns"}),)
    get_only = OpNode(m=KVGet(kv, ns="ns"), params={"key": "k"})
    g2 = OpGraph(nodes={get_only.id: get_only}, roots={get_only.id})
    with pytest.raises(PermissionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br2, g2)


@pytest.mark.asyncio
async def test_ctx_set_invariant():
    br = Branch(name="ctx")
    ok = OpNode(m=CtxSet(values={"a": 1}, allowed_keys={"a"}))
    g1 = OpGraph(nodes={ok.id: ok}, roots={ok.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g1)
    assert br.ctx["a"] == 1

    # Violate CtxWriteSet by attempting to write b (not allowed)
    class BadCtx(CtxSet):
        async def apply(self, br, **kw):
            br.ctx.update({"a": 2, "b": 3})
            return {"ok": True}

    bad = OpNode(m=BadCtx(values={"a": 2}, allowed_keys={"a"}))
    g2 = OpGraph(nodes={bad.id: bad}, roots={bad.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g2)


@pytest.mark.asyncio
async def test_subgraph_run_requires_right():
    # inner graph with a quick no-op
    class Quick(BaseOp):
        name = "quick"

        async def pre(self, br, **kw):
            return True

        async def apply(self, br, **kw):
            return {"ok": True}

        async def post(self, br, res):
            return res.get("ok", False)

    inner = OpNode(m=Quick())
    g_inner = OpGraph(nodes={inner.id: inner}, roots={inner.id})

    # missing right -> deny
    br_deny = Branch(name="sub-deny")
    n = OpNode(m=SubgraphRun(graph=g_inner))
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    with pytest.raises(PermissionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br_deny, g)

    # grant right -> ok
    br_ok = Branch(name="sub-ok")
    br_ok.caps = (Capability(subject=br_ok.id, rights={"graph.run"}),)
    n2 = OpNode(m=SubgraphRun(graph=g_inner))
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br_ok, g2)


@pytest.mark.asyncio
async def test_wrappers_retry_and_timeout():
    br = Branch(name="wrap")

    # Retry succeeds after failures
    inner = FailNTimes(2)
    node = OpNode(m=WithRetry(inner, retries=3, backoff_ms=1))
    g = OpGraph(nodes={node.id: node}, roots={node.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g)

    # Timeout triggers
    slow = SleepOp(0.2)
    node2 = OpNode(m=WithTimeout(slow, timeout_ms=50))
    g2 = OpGraph(nodes={node2.id: node2}, roots={node2.id})
    with pytest.raises(asyncio.TimeoutError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g2)
