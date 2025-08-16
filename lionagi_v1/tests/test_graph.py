from lionagi_v1.base.graph import OpGraph, OpNode


class DummyM:
    name = "d"
    requires = set()

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {}

    async def post(self, br, res):
        return True


def test_graph_validate_dag():
    n1 = OpNode(m=DummyM())
    n2 = OpNode(m=DummyM(), deps={n1.id})
    g = OpGraph(nodes={n1.id: n1, n2.id: n2}, roots={n1.id})
    order = g.validate_dag()
    assert order[0] == n1.id and order[-1] == n2.id
