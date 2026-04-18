"""
Coverage tests for:
  - lionagi/protocols/graph/graph.py       (~44 uncovered lines)
  - lionagi/protocols/generic/flow.py      (~38 uncovered lines)
  - lionagi/protocols/generic/progression.py (~36 uncovered lines)

Note: Flow.add_progression uses Pile.include internally. Pile.include
treats falsy objects (including empty Progressions) as "nothing to add".
Therefore all add_progression tests use non-empty Progressions backed
by items already present in the Flow.
"""

import asyncio

import pytest

from lionagi._errors import ItemExistsError, ItemNotFoundError, RelationError
from lionagi.protocols.generic.element import Element
from lionagi.protocols.generic.flow import Flow
from lionagi.protocols.generic.progression import Progression, prog
from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.protocols.graph.node import Node

# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture()
def three_node_dag():
    """Linear DAG: A -> B -> C with two edges."""
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
    """Two nodes, one edge: X -> Y."""
    g = Graph()
    n1 = Node(content="X")
    n2 = Node(content="Y")
    g.add_node(n1)
    g.add_node(n2)
    e = Edge(head=n1.id, tail=n2.id)
    g.add_edge(e)
    return g, n1, n2, e


@pytest.fixture()
def four_elem_prog():
    """Progression holding four distinct Elements."""
    elems = [Element() for _ in range(4)]
    p = Progression(order=[e.id for e in elems])
    return p, elems


def _flow_with_prog(name: str):
    """Flow with one item and a named non-empty Progression."""
    f = Flow()
    elem = Element()
    f.add_item(elem)
    p = Progression(order=[elem.id], name=name)
    f.add_progression(p)
    return f, elem, p


# ============================================================
# Graph — add_node
# ============================================================


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


# ============================================================
# Graph — add_edge
# ============================================================


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


# ============================================================
# Graph — remove_node
# ============================================================


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


# ============================================================
# Graph — remove_edge
# ============================================================


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


# ============================================================
# Graph — get_heads / get_tails
# ============================================================


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


# ============================================================
# Graph — is_acyclic
# ============================================================


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


# ============================================================
# Graph — topological_sort
# ============================================================


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


# ============================================================
# Graph — find_path  (async)
# ============================================================


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


# ============================================================
# Graph — __contains__
# ============================================================


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


# ============================================================
# Flow — instantiation
# ============================================================


class TestFlowInstantiation:
    def test_default_flow_empty(self):
        f = Flow()
        assert len(f.items) == 0
        assert len(f.progressions) == 0

    def test_named_flow(self):
        f = Flow(name="pipeline")
        assert f.name == "pipeline"

    def test_repr_contains_flow(self):
        assert "Flow" in repr(Flow())

    def test_len_equals_item_count(self):
        f = Flow()
        for _ in range(3):
            f.add_item(Element())
        assert len(f) == 3


# ============================================================
# Flow — add_progression
# Pile.include skips falsy values (empty Progression.__bool__ is False),
# so all tests here use non-empty Progressions referencing items in the Flow.
# ============================================================


class TestFlowAddProgression:
    def test_add_progression_with_items(self):
        f = Flow()
        elem = Element()
        f.add_item(elem)
        p = Progression(order=[elem.id], name="with-items")
        f.add_progression(p)
        assert len(f.progressions) == 1

    def test_progression_retrievable_by_name(self):
        f, elem, p = _flow_with_prog("stage1")
        assert f.get_progression("stage1").id == p.id

    def test_progression_name_indexed(self):
        f, elem, p = _flow_with_prog("idx-test")
        assert "idx-test" in f._progression_names

    def test_duplicate_name_raises(self):
        f, elem, p1 = _flow_with_prog("dup")
        elem2 = Element()
        f.add_item(elem2)
        p2 = Progression(order=[elem2.id], name="dup")
        with pytest.raises(ItemExistsError):
            f.add_progression(p2)

    def test_progression_referencing_unknown_item_raises(self):
        f = Flow()
        unknown_id = Element().id
        p = Progression(order=[unknown_id], name="missing")
        with pytest.raises(ItemNotFoundError):
            f.add_progression(p)

    def test_two_progressions_same_flow(self):
        f = Flow()
        e1, e2 = Element(), Element()
        f.add_item(e1)
        f.add_item(e2)
        p1 = Progression(order=[e1.id], name="p1")
        p2 = Progression(order=[e2.id], name="p2")
        f.add_progression(p1)
        f.add_progression(p2)
        assert len(f.progressions) == 2


# ============================================================
# Flow — remove_progression
# ============================================================


class TestFlowRemoveProgression:
    def test_remove_by_name(self):
        f, elem, p = _flow_with_prog("s1")
        f.remove_progression("s1")
        assert len(f.progressions) == 0

    def test_remove_by_uuid(self):
        f, elem, p = _flow_with_prog("s2")
        f.remove_progression(p.id)
        assert len(f.progressions) == 0

    def test_remove_by_instance(self):
        f, elem, p = _flow_with_prog("s3")
        f.remove_progression(p)
        assert len(f.progressions) == 0

    def test_name_removed_from_index(self):
        f, elem, p = _flow_with_prog("to-delete")
        f.remove_progression("to-delete")
        assert "to-delete" not in f._progression_names


# ============================================================
# Flow — get_progression
# ============================================================


class TestFlowGetProgression:
    def test_get_by_name(self):
        f, elem, p = _flow_with_prog("find-me")
        assert f.get_progression("find-me").id == p.id

    def test_get_by_uuid(self):
        f, elem, p = _flow_with_prog("by-id")
        assert f.get_progression(p.id).id == p.id

    def test_get_by_instance(self):
        f, elem, p = _flow_with_prog("by-inst")
        assert f.get_progression(p).id == p.id

    def test_get_missing_name_raises(self):
        f = Flow()
        with pytest.raises(Exception):
            f.get_progression("nope")


# ============================================================
# Flow — add_item
# ============================================================


class TestFlowAddItem:
    def test_item_added_to_pile(self):
        f = Flow()
        elem = Element()
        f.add_item(elem)
        assert elem.id in f.items

    def test_item_added_to_named_progression(self):
        f, initial_elem, p = _flow_with_prog("p1")
        new_elem = Element()
        f.add_item(new_elem, progressions="p1")
        assert new_elem.id in p

    def test_item_added_to_multiple_progressions(self):
        f = Flow()
        seed1, seed2 = Element(), Element()
        f.add_item(seed1)
        f.add_item(seed2)
        p1 = Progression(order=[seed1.id], name="p1")
        p2 = Progression(order=[seed2.id], name="p2")
        f.add_progression(p1)
        f.add_progression(p2)
        new_elem = Element()
        f.add_item(new_elem, progressions=["p1", "p2"])
        assert new_elem.id in p1
        assert new_elem.id in p2

    def test_item_added_without_progression(self):
        f = Flow()
        elem = Element()
        f.add_item(elem)
        assert len(f.items) == 1
        assert len(f.progressions) == 0


# ============================================================
# Flow — remove_item
# ============================================================


class TestFlowRemoveItem:
    def test_remove_from_pile(self):
        f = Flow()
        elem = Element()
        f.add_item(elem)
        f.remove_item(elem)
        assert elem.id not in f.items

    def test_remove_cleans_progressions(self):
        f, elem, p = _flow_with_prog("track")
        assert elem.id in p
        f.remove_item(elem)
        assert elem.id not in p

    def test_remove_by_uuid(self):
        f = Flow()
        elem = Element()
        f.add_item(elem)
        f.remove_item(elem.id)
        assert elem.id not in f.items


# ============================================================
# Flow — clear
# ============================================================


class TestFlowClear:
    def test_clear_empties_items_and_progressions(self):
        f, elem, p = _flow_with_prog("s")
        f.clear()
        assert len(f.items) == 0
        assert len(f.progressions) == 0
        assert f._progression_names == {}


# ============================================================
# Progression — move
# ============================================================


class TestProgressionMove:
    def test_move_to_later_position(self, four_elem_prog):
        p, elems = four_elem_prog
        first_id = p[0]
        p.move(0, 2)
        assert first_id in list(p.order)
        assert len(p) == 4

    def test_move_to_earlier_position(self, four_elem_prog):
        p, elems = four_elem_prog
        last_id = p[3]
        p.move(3, 0)
        assert list(p.order)[0] == last_id

    def test_move_preserves_length(self, four_elem_prog):
        p, elems = four_elem_prog
        p.move(0, 3)
        assert len(p) == 4

    def test_move_negative_from_index(self, four_elem_prog):
        p, elems = four_elem_prog
        last_id = p[-1]
        p.move(-1, 0)
        assert list(p.order)[0] == last_id

    def test_move_middle_element(self, four_elem_prog):
        p, elems = four_elem_prog
        mid_id = p[1]
        p.move(1, 3)
        assert mid_id in list(p.order)
        assert len(p) == 4


# ============================================================
# Progression — swap
# ============================================================


class TestProgressionSwap:
    def test_swap_adjacent(self, four_elem_prog):
        p, elems = four_elem_prog
        id0, id1 = p[0], p[1]
        p.swap(0, 1)
        assert p[0] == id1 and p[1] == id0

    def test_swap_first_and_last(self, four_elem_prog):
        p, elems = four_elem_prog
        id_first, id_last = p[0], p[3]
        p.swap(0, 3)
        assert p[0] == id_last and p[3] == id_first

    def test_swap_same_index_noop(self, four_elem_prog):
        p, elems = four_elem_prog
        original = list(p.order)
        p.swap(2, 2)
        assert list(p.order) == original

    def test_swap_negative_indices(self, four_elem_prog):
        p, elems = four_elem_prog
        id_last, id_penultimate = p[-1], p[-2]
        p.swap(-1, -2)
        assert p[-1] == id_penultimate and p[-2] == id_last

    def test_swap_preserves_length(self, four_elem_prog):
        p, elems = four_elem_prog
        p.swap(0, 2)
        assert len(p) == 4


# ============================================================
# Progression — __sub__
# ============================================================


class TestProgressionSub:
    # Note: validate_order treats a Progression as an Element and returns
    # [progression.id], not its contents. Pass explicit UUID lists instead.

    def test_sub_removes_shared_ids(self, four_elem_prog):
        p, elems = four_elem_prog
        # subtract first two IDs as a list
        result = p - [elems[0].id, elems[1].id]
        result_ids = list(result.order)
        assert elems[0].id not in result_ids
        assert elems[1].id not in result_ids
        assert elems[2].id in result_ids
        assert elems[3].id in result_ids

    def test_sub_returns_new_progression(self, four_elem_prog):
        p, elems = four_elem_prog
        result = p - [elems[0].id]
        assert isinstance(result, Progression)
        assert len(p) == 4  # original unchanged
        assert len(result) == 3

    def test_sub_single_element(self, four_elem_prog):
        p, elems = four_elem_prog
        result = p - elems[2]  # Element is valid: returns [element.id]
        result_ids = list(result.order)
        assert elems[2].id not in result_ids
        assert len(result) == 3

    def test_sub_all_elements(self, four_elem_prog):
        p, elems = four_elem_prog
        all_ids = [e.id for e in elems]
        result = p - all_ids
        assert len(result) == 0


# ============================================================
# Progression — __reversed__
# ============================================================


class TestProgressionReversed:
    def test_reversed_returns_progression(self, four_elem_prog):
        p, elems = four_elem_prog
        assert isinstance(reversed(p), Progression)

    def test_reversed_correct_order(self, four_elem_prog):
        p, elems = four_elem_prog
        original = list(p.order)
        assert list(reversed(p).order) == original[::-1]

    def test_reversed_does_not_mutate_original(self, four_elem_prog):
        p, elems = four_elem_prog
        original = list(p.order)
        _ = reversed(p)
        assert list(p.order) == original

    def test_last_id_via_reversed(self, four_elem_prog):
        p, elems = four_elem_prog
        rev = reversed(p)
        assert list(rev.order)[0] == elems[3].id


# ============================================================
# Progression — negative indexing
# ============================================================


class TestProgressionNegativeIndex:
    def test_neg1_is_last(self, four_elem_prog):
        p, elems = four_elem_prog
        assert p[-1] == elems[3].id

    def test_neg2_is_second_to_last(self, four_elem_prog):
        p, elems = four_elem_prog
        assert p[-2] == elems[2].id

    def test_neg_len_is_first(self, four_elem_prog):
        p, elems = four_elem_prog
        assert p[-4] == elems[0].id

    def test_out_of_range_raises(self, four_elem_prog):
        p, elems = four_elem_prog
        with pytest.raises(ItemNotFoundError):
            _ = p[-10]


# ============================================================
# Progression — _validate_index
# ============================================================


class TestProgressionValidateIndex:
    def test_negative_converted_to_positive(self):
        elems = [Element() for _ in range(3)]
        p = Progression(order=[e.id for e in elems])
        assert p._validate_index(-1) == 2

    def test_empty_progression_raises(self):
        p = Progression()
        with pytest.raises(ItemNotFoundError):
            p._validate_index(0)

    def test_out_of_range_raises(self):
        elems = [Element() for _ in range(2)]
        p = Progression(order=[e.id for e in elems])
        with pytest.raises(ItemNotFoundError):
            p._validate_index(5)

    def test_allow_end_permits_len(self):
        elems = [Element() for _ in range(3)]
        p = Progression(order=[e.id for e in elems])
        assert p._validate_index(3, allow_end=True) == 3


# ============================================================
# prog() factory
# ============================================================


class TestProgFactory:
    def test_creates_named_progression(self):
        elems = [Element() for _ in range(3)]
        p = prog([e.id for e in elems], "my-prog")
        assert isinstance(p, Progression)
        assert p.name == "my-prog"
        assert len(p) == 3

    def test_creates_unnamed_progression(self):
        p = prog([])
        assert p.name is None
        assert len(p) == 0
