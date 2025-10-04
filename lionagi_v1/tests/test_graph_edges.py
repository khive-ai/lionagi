import pytest

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


def test_missing_dependency_raises():
    a = OpNode(m=DummyM())
    b = OpNode(m=DummyM(), deps={a.id, "nonexistent"})  # type: ignore
    g = OpGraph(nodes={a.id: a, b.id: b}, roots={a.id})
    with pytest.raises(ValueError):
        g.validate_dag()


def test_cycle_detection_raises():
    a = OpNode(m=DummyM())
    b = OpNode(m=DummyM(), deps={a.id})
    # create a cycle: a depends on b
    a.deps.add(b.id)
    g = OpGraph(nodes={a.id: a, b.id: b}, roots={a.id})
    with pytest.raises(ValueError):
        g.validate_dag()


def test_auto_roots_when_not_provided():
    a = OpNode(m=DummyM())  # indegree 0
    b = OpNode(m=DummyM(), deps={a.id})
    g = OpGraph(nodes={a.id: a, b.id: b}, roots=set())
    order = g.validate_dag()
    assert order[0] == a.id and order[-1] == b.id


def test_diamond_fan_in_respected_order():
    # a -> b, a -> c, b -> d, c -> d
    a = OpNode(m=DummyM())
    b = OpNode(m=DummyM(), deps={a.id})
    c = OpNode(m=DummyM(), deps={a.id})
    d = OpNode(m=DummyM(), deps={b.id, c.id})
    g = OpGraph(nodes={a.id: a, b.id: b, c.id: c, d.id: d}, roots={a.id})
    order = g.validate_dag()
    # a must appear before b and c; b and c before d
    assert order.index(a.id) < order.index(b.id)
    assert order.index(a.id) < order.index(c.id)
    assert order.index(b.id) < order.index(d.id)
    assert order.index(c.id) < order.index(d.id)
