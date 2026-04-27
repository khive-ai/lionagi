"""Graph behavior tests: add/remove nodes/edges, traversal, path-finding."""

import asyncio

import pytest

from lionagi._errors import RelationError
from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.protocols.graph.node import Node


@pytest.fixture()
def three_node_dag():
    g = Graph()
    a = Node(content="A")
    b = Node(content="B")
    c = Node(content="C")
    g.add_node(a)
    g.add_node(b)
    g.add_node(c)
    e1 = Edge(head=a.id, tail=b.id)
    e2 = Edge(head=b.id, tail=c.id)
    g.add_edge(e1)
    g.add_edge(e2)
    return g, a, b, c, e1, e2


@pytest.fixture()
def simple_graph():
    g = Graph()
    n1 = Node(content="X")
    n2 = Node(content="Y")
    g.add_node(n1)
    g.add_node(n2)
    e = Edge(head=n1.id, tail=n2.id)
    g.add_edge(e)
    return g, n1, n2, e


class TestGraphAddNode:
    def test_node_appears_in_internal_nodes(self):
        g = Graph()
        n = Node(content="hello")
        g.add_node(n)
        assert n.id in g.internal_nodes

    def test_mapping_initialised_empty(self):
        g = Graph()
        n = Node(content="map")
        g.add_node(n)
        assert g.node_edge_mapping[n.id] == {"in": {}, "out": {}}

    def test_non_relational_raises(self):
        g = Graph()
        with pytest.raises(RelationError):
            g.add_node("not-a-node")

    def test_duplicate_node_raises(self):
        g = Graph()
        n = Node(content="dup")
        g.add_node(n)
        with pytest.raises(RelationError):
            g.add_node(n)


class TestGraphAddEdge:
    def test_edge_appears_in_internal_edges(self, simple_graph):
        g, n1, n2, e = simple_graph
        assert e.id in g.internal_edges

    def test_mapping_updated(self, simple_graph):
        g, n1, n2, e = simple_graph
        assert e.id in g.node_edge_mapping[n1.id]["out"]
        assert e.id in g.node_edge_mapping[n2.id]["in"]

    def test_invalid_type_raises(self):
        g = Graph()
        with pytest.raises(RelationError):
            g.add_edge("not-an-edge")

    def test_missing_tail_node_raises(self):
        g = Graph()
        n1 = Node(content="present")
        n2 = Node(content="absent")
        g.add_node(n1)
        e = Edge(head=n1.id, tail=n2.id)
        with pytest.raises(RelationError):
            g.add_edge(e)


class TestGraphRemoveNode:
    def test_removes_from_internal_nodes(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        g.remove_node(a)
        assert a.id not in g.internal_nodes

    def test_removes_incident_edges(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        g.remove_node(b)
        assert e1.id not in g.internal_edges
        assert e2.id not in g.internal_edges

    def test_by_id_string(self, simple_graph):
        g, n1, n2, e = simple_graph
        g.remove_node(str(n1.id))
        assert n1.id not in g.internal_nodes

    def test_nonexistent_raises(self):
        g = Graph()
        ghost = Node(content="ghost")
        with pytest.raises(RelationError):
            g.remove_node(ghost)


class TestGraphRemoveEdge:
    def test_removes_from_internal_edges(self, simple_graph):
        g, n1, n2, e = simple_graph
        g.remove_edge(e)
        assert e.id not in g.internal_edges

    def test_cleans_mapping(self, simple_graph):
        g, n1, n2, e = simple_graph
        g.remove_edge(e)
        assert e.id not in g.node_edge_mapping[n1.id]["out"]
        assert e.id not in g.node_edge_mapping[n2.id]["in"]

    def test_by_edge_id(self, simple_graph):
        g, n1, n2, e = simple_graph
        g.remove_edge(e.id)
        assert e.id not in g.internal_edges

    def test_nonexistent_raises(self, simple_graph):
        g, n1, n2, existing_edge = simple_graph
        ghost_e = Edge(head=n1.id, tail=n2.id)
        with pytest.raises(RelationError):
            g.remove_edge(ghost_e)


class TestGraphGetHeads:
    def test_source_node_is_head(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        head_ids = [n.id for n in g.get_heads()]
        assert a.id in head_ids
        assert b.id not in head_ids
        assert c.id not in head_ids

    def test_isolated_node_is_head(self):
        g = Graph()
        n = Node(content="solo")
        g.add_node(n)
        assert n.id in [h.id for h in g.get_heads()]


class TestGraphGetTails:
    def test_sink_node_is_tail(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        tail_ids = [n.id for n in g.get_tails()]
        assert c.id in tail_ids
        assert a.id not in tail_ids
        assert b.id not in tail_ids

    def test_isolated_node_is_tail(self):
        g = Graph()
        n = Node(content="solo")
        g.add_node(n)
        assert n.id in [t.id for t in g.get_tails()]


class TestGraphIsAcyclic:
    def test_linear_dag_is_acyclic(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        assert g.is_acyclic() is True

    def test_graph_with_back_edge_is_not_acyclic(self):
        g = Graph()
        a = Node(content="A")
        b = Node(content="B")
        g.add_node(a)
        g.add_node(b)
        g.add_edge(Edge(head=a.id, tail=b.id))
        g.add_edge(Edge(head=b.id, tail=a.id))
        assert g.is_acyclic() is False

    def test_empty_graph_is_acyclic(self):
        assert Graph().is_acyclic() is True


class TestGraphTopologicalSort:
    def test_three_node_dag_order(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        ids = [n.id for n in g.topological_sort()]
        assert ids.index(a.id) < ids.index(b.id)
        assert ids.index(b.id) < ids.index(c.id)

    def test_single_node(self):
        g = Graph()
        n = Node(content="only")
        g.add_node(n)
        result = g.topological_sort()
        assert len(result) == 1 and result[0].id == n.id

    def test_cyclic_raises(self):
        g = Graph()
        a, b = Node(content="A"), Node(content="B")
        g.add_node(a)
        g.add_node(b)
        g.add_edge(Edge(head=a.id, tail=b.id))
        g.add_edge(Edge(head=b.id, tail=a.id))
        with pytest.raises(ValueError):
            g.topological_sort()


class TestGraphFindPath:
    def test_path_exists(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        path = asyncio.run(g.find_path(a, c))
        assert path is not None and len(path) == 2

    def test_direct_path(self, simple_graph):
        g, n1, n2, e = simple_graph
        path = asyncio.run(g.find_path(n1, n2))
        assert path is not None and path[0].id == e.id

    def test_no_path_returns_none(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        assert asyncio.run(g.find_path(c, a)) is None

    def test_same_node_returns_empty(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        assert asyncio.run(g.find_path(a, a)) == []

    def test_missing_start_raises(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        ghost = Node(content="ghost")
        with pytest.raises(RelationError):
            asyncio.run(g.find_path(ghost, c))

    def test_missing_end_raises(self, three_node_dag):
        g, a, b, c, e1, e2 = three_node_dag
        ghost = Node(content="ghost")
        with pytest.raises(RelationError):
            asyncio.run(g.find_path(a, ghost))


class TestGraphContains:
    def test_node_in_graph(self, simple_graph):
        g, n1, n2, e = simple_graph
        assert n1 in g

    def test_edge_in_graph(self, simple_graph):
        g, n1, n2, e = simple_graph
        assert e in g

    def test_unknown_node_not_in_graph(self, simple_graph):
        g, n1, n2, e = simple_graph
        assert Node(content="ghost") not in g
