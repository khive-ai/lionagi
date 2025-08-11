import asyncio

import msgspec
import pytest

from lionagi_v1.base.graph import OpGraph, OpNode
from lionagi_v1.base.ipu import StrictIPU, default_invariants
from lionagi_v1.base.runner import Runner
from lionagi_v1.base.types import Branch

# ---- Helpers


class Schema(msgspec.Struct, kw_only=True):
    text: str
    score: float


class Provider:
    async def generate(self, prompt: str) -> tuple[str, float]:
        await asyncio.sleep(0.06)  # ~60ms


# ---- Morphisms for tests


class LatencyOK:
    name = "lat.ok"
    requires = set()
    latency_budget_ms = 100  # 100ms budget

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        await asyncio.sleep(0.05)
        return {"x": 1}

    async def post(self, br, res):
        return True


class LatencyFail:
    name = "lat.fail"
    requires = set()
    latency_budget_ms = 20  # 20ms -> will fail

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        await asyncio.sleep(0.05)
        return {"x": 1}

    async def post(self, br, res):
        return True


class ShapeKeys:
    name = "shape.keys"
    requires = set()
    result_keys = {"text", "score"}

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {"text": "ok", "score": 0.9}

    async def post(self, br, res):
        return True


class ShapeKeysFail(ShapeKeys):
    async def apply(self, br, **kw):
        return {"text": "ok"}  # missing "score"


class ShapeSchema:
    name = "shape.schema"
    requires = set()
    result_schema = Schema

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {"text": "ok", "score": 0.5}

    async def post(self, br, res):
        return True


class ShapeSchemaFail(ShapeSchema):
    async def apply(self, br, **kw):
        return {"text": "ok", "score": "bad"}  # wrong type


class CtxWriteAllowed:
    name = "ctx.allowed"
    requires = set()
    ctx_writes = {"k1"}

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        br.ctx["k1"] = 1
        return {"ok": True}

    async def post(self, br, res):
        return True


class CtxWriteViolation(CtxWriteAllowed):
    async def apply(self, br, **kw):
        br.ctx["k1"] = 1
        br.ctx["k2"] = 2  # not allowed
        return {"ok": True}


class IOWithoutRequires:
    name = "io.ambient"
    io = True
    requires = set()  # <- should trigger NoAmbientAuthority pre violation

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {"ok": True}

    async def post(self, br, res):
        return True


class SizeBoundOK:
    name = "size.ok"
    requires = set()
    result_bytes_limit = 2000

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {"blob": "x" * 500}

    async def post(self, br, res):
        return True


class SizeBoundFail(SizeBoundOK):
    result_bytes_limit = 100

    async def apply(self, br, **kw):
        return {"blob": "x" * 500}


# ---- Tests


@pytest.mark.asyncio
async def test_latency_bound_pass_and_fail():
    br = Branch(name="lat")
    # pass
    g1 = OpGraph(nodes={(n := OpNode(m=LatencyOK())).id: n}, roots={n.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g1)
    # fail
    g2 = OpGraph(nodes={(m := OpNode(m=LatencyFail())).id: m}, roots={m.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g2)


@pytest.mark.asyncio
async def test_result_shape_keys_and_schema():
    br = Branch(name="shape")
    # keys ok
    n1 = OpNode(m=ShapeKeys())
    g1 = OpGraph(nodes={n1.id: n1}, roots={n1.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g1)
    # keys missing -> fail
    n2 = OpNode(m=ShapeKeysFail())
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g2)
    # schema ok
    n3 = OpNode(m=ShapeSchema())
    g3 = OpGraph(nodes={n3.id: n3}, roots={n3.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g3)
    # schema wrong type -> fail
    n4 = OpNode(m=ShapeSchemaFail())
    g4 = OpGraph(nodes={n4.id: n4}, roots={n4.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g4)


@pytest.mark.asyncio
async def test_ctx_write_set_allows_and_denies():
    br = Branch(name="ctx")
    # ok
    n1 = OpNode(m=CtxWriteAllowed())
    g1 = OpGraph(nodes={n1.id: n1}, roots={n1.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g1)
    assert "k1" in br.ctx and "k2" not in br.ctx
    # violation
    br2 = Branch(name="ctx2")
    n2 = OpNode(m=CtxWriteViolation())
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br2, g2)


@pytest.mark.asyncio
async def test_no_ambient_authority_pre_enforced():
    br = Branch(name="ambient")
    n = OpNode(m=IOWithoutRequires())
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g)


@pytest.mark.asyncio
async def test_result_size_bound():
    br = Branch(name="size")
    # ok
    n1 = OpNode(m=SizeBoundOK())
    g1 = OpGraph(nodes={n1.id: n1}, roots={n1.id})
    await Runner(ipu=StrictIPU(default_invariants())).run(br, g1)
    # violation
    n2 = OpNode(m=SizeBoundFail())
    g2 = OpGraph(nodes={n2.id: n2}, roots={n2.id})
    with pytest.raises(AssertionError):
        await Runner(ipu=StrictIPU(default_invariants())).run(br, g2)
