import pytest

from lionagi_v1.base.graph import OpGraph, OpNode
from lionagi_v1.base.ipu import StrictIPU, default_invariants
from lionagi_v1.base.runner import Runner
from lionagi_v1.base.types import Branch, Capability


class StubProv:
    async def generate(self, prompt: str) -> str:
        return f"ok:{prompt}"


class LLMCall:
    name = "llm"
    requires = {"net.out"}

    def __init__(self, prov):
        self.p = prov

    async def pre(self, br, **kw):
        return "prompt" in kw

    async def apply(self, br, **kw):
        t = await self.p.generate(kw["prompt"])
        br.ctx["last"] = t
        return {"text": t}

    async def post(self, br, res):
        return "text" in res


@pytest.mark.asyncio
async def test_runner_success():
    br = Branch(name="r")
    br.caps = (Capability(subject=br.id, rights={"net.out"}),)
    n1 = OpNode(m=LLMCall(StubProv()))
    g = OpGraph(nodes={n1.id: n1}, roots={n1.id})
    r = Runner(ipu=StrictIPU(default_invariants()))
    res = await r.run(br, g)
    assert list(res.values())[0]["text"].startswith("ok:")
    assert "last" in br.ctx


@pytest.mark.asyncio
async def test_runner_policy_denial():
    br = Branch(name="r2")  # no capabilities
    n1 = OpNode(m=LLMCall(StubProv()))
    g = OpGraph(nodes={n1.id: n1}, roots={n1.id})
    r = Runner(ipu=StrictIPU(default_invariants()))
    with pytest.raises(PermissionError):
        await r.run(br, g)
