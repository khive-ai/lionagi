import pytest

from lionagi_v1.base.graph import OpGraph, OpNode
from lionagi_v1.base.ipu import default_invariants
from lionagi_v1.base.runner import Runner
from lionagi_v1.base.types import Branch, Observation


class QuickOp:
    name = "quick"
    requires = set()

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {"ok": True}

    async def post(self, br, res):
        return True


class RecordingIPU:
    """Minimal IPU to record observations and still enforce invariants."""

    name = "rec"

    def __init__(self, invariants):
        self.invariants = invariants
        self.events = []

    async def before_node(self, br, node):
        for inv in self.invariants:
            assert inv.pre(br, node)

    async def after_node(self, br, node, result):
        for inv in self.invariants:
            assert inv.post(br, node, result)

    async def on_observation(self, obs: Observation):
        self.events.append(obs)


@pytest.mark.asyncio
async def test_observations_emitted_start_and_finish():
    br = Branch(name="obs")
    n = OpNode(m=QuickOp())
    g = OpGraph(nodes={n.id: n}, roots={n.id})
    ipu = RecordingIPU(default_invariants())
    await Runner(ipu=ipu).run(br, g)
    kinds = [e.what for e in ipu.events]
    assert "node.start" in kinds and "node.finish" in kinds
