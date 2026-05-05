# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.beta.work.builder: OperationGraphBuilder (alias Builder)."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from lionagi.protocols.graph.edge import Edge, EdgeCondition
from lionagi.protocols.graph.graph import Graph
from lionagi.beta.work.builder import Builder, OperationGraphBuilder
from lionagi.beta.work.node import Operation
from lionagi.ln.types._sentinel import Undefined

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        b = OperationGraphBuilder()
        assert b.name == "OperationGraph"
        assert isinstance(b.graph, Graph)
        assert b._nodes == {}
        assert b._executed == set()
        assert b._current_heads == []

    def test_string_arg_becomes_name(self):
        b = OperationGraphBuilder("my_graph")
        assert b.name == "my_graph"
        assert isinstance(b.graph, Graph)

    def test_explicit_name_kwarg(self):
        b = OperationGraphBuilder(name="custom")
        assert b.name == "custom"

    def test_builder_alias_is_same_class(self):
        assert Builder is OperationGraphBuilder

    def test_builder_alias_works(self):
        b = Builder()
        assert b.name == "OperationGraph"
        assert isinstance(b.graph, Graph)

    def test_existing_graph_accepted(self):
        g = Graph()
        b = OperationGraphBuilder(g)
        assert b.graph is g


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_single_node(self):
        b = OperationGraphBuilder()
        result = b.add("a", "op1")
        assert "a" in b._nodes
        assert isinstance(b._nodes["a"], Operation)
        assert result is b  # fluent

    def test_add_sets_current_heads(self):
        b = OperationGraphBuilder()
        b.add("a", "op1")
        assert b._current_heads == ["a"]

    def test_add_sequential_auto_link(self):
        b = OperationGraphBuilder()
        b.add("a", "op1")
        b.add("b", "op2")
        # b should depend on a via a sequential edge
        assert b._current_heads == ["b"]
        node_a = b._nodes["a"]
        node_b = b._nodes["b"]
        edges = b.graph.get_node_edges(node_b.id, direction="in")
        assert any(e.head == node_a.id for e in edges)

    def test_add_independent_depends_on_empty_list(self):
        b = OperationGraphBuilder()
        b.add("a", "op1")
        b.add("c", "op3", depends_on=[])
        # c should have no incoming edges
        node_c = b._nodes["c"]
        edges = b.graph.get_node_edges(node_c.id, direction="in")
        assert len(edges) == 0

    def test_add_explicit_depends_on(self):
        b = OperationGraphBuilder()
        b.add("a", "op1")
        b.add("b", "op2", depends_on=[])  # independent
        b.add("d", "op4", depends_on=["a"])
        node_a = b._nodes["a"]
        node_d = b._nodes["d"]
        edges = b.graph.get_node_edges(node_d.id, direction="in")
        assert any(e.head == node_a.id for e in edges)

    def test_add_duplicate_name_raises_value_error(self):
        b = OperationGraphBuilder()
        b.add("a", "op1")
        with pytest.raises(ValueError, match="already exists"):
            b.add("a", "op2")

    def test_add_nonexistent_dep_raises_value_error(self):
        b = OperationGraphBuilder()
        with pytest.raises(ValueError, match="not found"):
            b.add("x", "op", depends_on=["nonexistent"])

    def test_add_stores_metadata_name(self):
        b = OperationGraphBuilder()
        b.add("my_op", "some_operation")
        op = b._nodes["my_op"]
        assert op.metadata.get("name") == "my_op"

    def test_add_passes_parameters(self):
        b = OperationGraphBuilder()
        b.add("a", "op", parameters={"key": "value"})
        op = b._nodes["a"]
        assert op.parameters == {"key": "value"}

    def test_add_passes_metadata(self):
        b = OperationGraphBuilder()
        b.add("a", "op", metadata={"custom": 42})
        op = b._nodes["a"]
        assert op.metadata.get("custom") == 42

    def test_add_branch_str_stored_in_metadata(self):
        b = OperationGraphBuilder()
        b.add("a", "op", branch="my-branch")
        op = b._nodes["a"]
        assert op.metadata.get("branch") == "my-branch"

    def test_add_branch_uuid_stored_in_metadata(self):
        b = OperationGraphBuilder()
        uid = uuid4()
        b.add("a", "op", branch=uid)
        op = b._nodes["a"]
        assert op.metadata.get("branch") == uid

    def test_add_branch_uuid_string_stored_as_uuid(self):
        b = OperationGraphBuilder()
        uid = uuid4()
        b.add("a", "op", branch=str(uid))
        op = b._nodes["a"]
        assert op.metadata.get("branch") == uid

    def test_add_branch_invalid_raises_value_error(self):
        b = OperationGraphBuilder()
        with pytest.raises(ValueError, match="Invalid branch reference"):
            b.add("a", "op", branch=123)

    def test_add_control_type_stored(self):
        b = OperationGraphBuilder()
        b.add("ctrl", "op", control_type="halt")
        op = b._nodes["ctrl"]
        assert op.control_type == "halt"

    def test_add_morphism_stored(self):
        b = OperationGraphBuilder()

        class FakeMorphism:
            name = "fake"

        m = FakeMorphism()
        b.add("a", "op", morphism=m)
        op = b._nodes["a"]
        assert op.morphism is m

    def test_add_fluent_chain(self):
        b = OperationGraphBuilder()
        result = b.add("a", "op1").add("b", "op2").add("c", "op3")
        assert result is b
        assert len(b._nodes) == 3

    def test_add_node_registered_in_graph(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        op = b._nodes["a"]
        assert op.id in b.graph.internal_nodes


# ---------------------------------------------------------------------------
# add_operation()
# ---------------------------------------------------------------------------


class TestAddOperation:
    def test_add_operation_returns_uuid(self):
        b = OperationGraphBuilder()
        uid = b.add_operation("op1")
        assert isinstance(uid, UUID)

    def test_add_operation_name_deduplication(self):
        b = OperationGraphBuilder()
        b.add_operation("op1")
        b.add_operation("op1")  # should become op1_2
        assert "op1" in b._nodes
        assert "op1_2" in b._nodes

    def test_add_operation_triple_deduplication(self):
        b = OperationGraphBuilder()
        b.add_operation("op")
        b.add_operation("op")
        b.add_operation("op")
        assert "op" in b._nodes
        assert "op_2" in b._nodes
        assert "op_3" in b._nodes

    def test_add_operation_uuid_dep_resolved(self):
        b = OperationGraphBuilder()
        uid_a = b.add_operation("a")
        uid_b = b.add_operation("b", depends_on=[uid_a])
        node_b = b.get_by_id(uid_b)
        edges = b.graph.get_node_edges(node_b.id, direction="in")
        assert any(e.head == uid_a for e in edges)

    def test_add_operation_unknown_uuid_dep_raises(self):
        b = OperationGraphBuilder()
        with pytest.raises(ValueError, match="not found"):
            b.add_operation("a", depends_on=[uuid4()])


# ---------------------------------------------------------------------------
# add_control()
# ---------------------------------------------------------------------------


class TestAddControl:
    def test_add_control_returns_uuid(self):
        b = OperationGraphBuilder()
        uid = b.add_control("ctrl", "halt")
        assert isinstance(uid, UUID)

    def test_add_control_sets_control_type(self):
        b = OperationGraphBuilder()
        b.add_control("ctrl", "halt")
        op = b._nodes["ctrl"]
        assert op.control_type == "halt"

    def test_add_control_operation_type_prefixed(self):
        b = OperationGraphBuilder()
        b.add_control("ctrl", "halt")
        op = b._nodes["ctrl"]
        assert op.operation_type == "control.halt"

    def test_add_control_with_reason(self):
        b = OperationGraphBuilder()
        b.add_control("ctrl", "halt", reason="test reason")
        op = b._nodes["ctrl"]
        assert op.control_policy is not None
        assert op.control_policy.get("reason") == "test reason"

    def test_add_control_with_targets(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        node_a = b._nodes["a"]
        b.add_control("ctrl", "skip", targets=[node_a.id])
        op = b._nodes["ctrl"]
        assert op.control_policy is not None
        assert node_a.id in op.control_policy.get("targets", [])


# ---------------------------------------------------------------------------
# add_morphism()
# ---------------------------------------------------------------------------


class TestAddMorphism:
    def test_add_morphism_uses_morphism_name(self):
        b = OperationGraphBuilder()

        class FakeMorphism:
            name = "my_morphism_type"

        m = FakeMorphism()
        b.add_morphism("m1", m)
        op = b._nodes["m1"]
        assert op.operation_type == "my_morphism_type"

    def test_add_morphism_fallback_name(self):
        b = OperationGraphBuilder()

        class FakeMorphism:
            pass  # no name attribute

        m = FakeMorphism()
        b.add_morphism("m1", m)
        op = b._nodes["m1"]
        # fallback: "morphism.{name}"
        assert op.operation_type == "morphism.m1"

    def test_add_morphism_stores_morphism(self):
        b = OperationGraphBuilder()

        class FakeMorphism:
            name = "fake"

        m = FakeMorphism()
        b.add_morphism("m1", m)
        op = b._nodes["m1"]
        assert op.morphism is m

    def test_add_morphism_returns_self(self):
        b = OperationGraphBuilder()

        class FakeMorphism:
            name = "fake"

        result = b.add_morphism("m1", FakeMorphism())
        assert result is b


# ---------------------------------------------------------------------------
# depends_on()
# ---------------------------------------------------------------------------


class TestDependsOn:
    def test_depends_on_adds_edge(self):
        b = OperationGraphBuilder()
        b.add("a", "op", depends_on=[])
        b.add("b", "op", depends_on=[])
        b.depends_on("b", "a")
        node_a = b._nodes["a"]
        node_b = b._nodes["b"]
        edges = b.graph.get_node_edges(node_b.id, direction="in")
        assert any(e.head == node_a.id for e in edges)

    def test_depends_on_unknown_target_raises(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        with pytest.raises(ValueError, match="not found"):
            b.depends_on("nonexistent", "a")

    def test_depends_on_unknown_dep_raises(self):
        b = OperationGraphBuilder()
        b.add("b", "op")
        with pytest.raises(ValueError, match="not found"):
            b.depends_on("b", "nonexistent")

    def test_depends_on_returns_self(self):
        b = OperationGraphBuilder()
        b.add("a", "op", depends_on=[])
        b.add("b", "op", depends_on=[])
        result = b.depends_on("b", "a")
        assert result is b


# ---------------------------------------------------------------------------
# get() / get_by_id()
# ---------------------------------------------------------------------------


class TestGetMethods:
    def test_get_by_name_returns_operation(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        op = b.get("a")
        assert isinstance(op, Operation)

    def test_get_unknown_raises_value_error(self):
        b = OperationGraphBuilder()
        with pytest.raises(ValueError, match="not found"):
            b.get("nonexistent")

    def test_get_by_id_returns_operation(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        op = b._nodes["a"]
        result = b.get_by_id(op.id)
        assert result is op

    def test_get_by_id_unknown_returns_none(self):
        b = OperationGraphBuilder()
        result = b.get_by_id(uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# mark_executed() / get_unexecuted_nodes()
# ---------------------------------------------------------------------------


class TestExecutionTracking:
    def test_mark_executed_adds_to_set(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.mark_executed("a")
        op = b._nodes["a"]
        assert op.id in b._executed

    def test_mark_executed_returns_self(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        result = b.mark_executed("a")
        assert result is b

    def test_mark_executed_unknown_name_ignored(self):
        b = OperationGraphBuilder()
        b.mark_executed("nonexistent")  # should not raise
        assert len(b._executed) == 0

    def test_get_unexecuted_nodes_all(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.add("b", "op")
        unexec = b.get_unexecuted_nodes()
        assert len(unexec) == 2

    def test_get_unexecuted_nodes_excludes_executed(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.add("b", "op")
        b.mark_executed("a")
        unexec = b.get_unexecuted_nodes()
        ids = [op.id for op in unexec]
        assert b._nodes["a"].id not in ids
        assert b._nodes["b"].id in ids


# ---------------------------------------------------------------------------
# build() / get_graph()
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_returns_graph(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        g = b.build()
        assert isinstance(g, Graph)

    def test_build_acyclic_succeeds(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.add("b", "op")  # sequential -> a
        b.build()  # should not raise

    def test_build_with_cycle_raises_value_error(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.add("b", "op", depends_on=["a"])
        # Inject b->a edge to create cycle
        b.graph.add_edge(Edge(head=b._nodes["b"].id, tail=b._nodes["a"].id))
        with pytest.raises(ValueError, match="cycles"):
            b.build()

    def test_get_graph_same_as_build(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        g1 = b.build()
        g2 = b.get_graph()
        assert g1 is g2


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_nodes(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.clear()
        assert b._nodes == {}

    def test_clear_resets_executed(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.mark_executed("a")
        b.clear()
        assert b._executed == set()

    def test_clear_resets_heads(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.clear()
        assert b._current_heads == []

    def test_clear_resets_graph(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        old_graph = b.graph
        b.clear()
        assert b.graph is not old_graph
        assert len(b.graph) == 0

    def test_clear_returns_self(self):
        b = OperationGraphBuilder()
        result = b.clear()
        assert result is b


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_format(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        r = repr(b)
        assert "OperationGraphBuilder" in r
        assert "operations=1" in r
        assert "executed=0" in r

    def test_repr_reflects_state(self):
        b = OperationGraphBuilder()
        b.add("a", "op")
        b.add("b", "op")
        b.mark_executed("a")
        r = repr(b)
        assert "operations=2" in r
        assert "executed=1" in r
