import asyncio
import time

import pytest

from lionagi_v1.base.graph import OpGraph, OpNode
from lionagi_v1.base.ipu import StrictIPU, default_invariants
from lionagi_v1.base.runner import Runner
from lionagi_v1.base.types import Branch


class ParamOp:
    name = "param.op"
    requires = set()  # no IO

    async def pre(self, br, **kw):
        # Expect x present; node.params should override branch.ctx
        return "x" in kw

    async def apply(self, br, **kw):
        br.ctx["seen_x"] = kw["x"]
        br.ctx["from_ctx"] = kw.get("y", None)
        return {"ok": True}

    async def post(self, br, res):
        return res.get("ok", False)


@pytest.mark.asyncio
async def test_kwargs_merge_and_priority():
    br = Branch(name="k")
    br.ctx = {"x": "ctx", "y": "ctx_y"}
    n = OpNode(m=ParamOp())
    n.params = {"x": "param"}  # should win over br.ctx['x']
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    r = Runner(ipu=StrictIPU(default_invariants()))
    await r.run(br, g)
    assert br.ctx["seen_x"] == "param"
    assert br.ctx["from_ctx"] == "ctx_y"


class SleepOp:
    name = "sleep"
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


@pytest.mark.asyncio
async def test_concurrency_two_roots():
    br = Branch(name="conc")
    n1, n2 = OpNode(m=SleepOp(0.2)), OpNode(m=SleepOp(0.2))
    g = OpGraph(nodes={n1.id: n1, n2.id: n2}, roots={n1.id, n2.id})
    r = Runner(ipu=StrictIPU(default_invariants()))
    t0 = time.perf_counter()
    await r.run(br, g)
    elapsed = time.perf_counter() - t0
    # Should be significantly less than sequential 0.4s; allow margin
    assert elapsed < 0.35, f"Expected concurrency; took {elapsed:.3f}s"
