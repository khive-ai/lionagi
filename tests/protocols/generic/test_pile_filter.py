# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Pile filter features (PR3).

Covers:
- pile.filter(predicate) named method
- pile[lambda x: ...] callable key dispatch in __getitem__
- _filter_by_function internal implementation
- Edge cases: empty results, all match, order preservation
- Type preservation: item_type and strict_type carried to filtered Pile
- Non-callable keys still dispatch normally (int, UUID, slice)
- Complex predicates (metadata checks)
"""

import pytest

from lionagi.protocols.generic.pile import Pile
from lionagi.protocols.graph.node import Node


class SampleNode(Node):
    value: int = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nodes_5():
    """Five TestNodes with values 0..4."""
    return [SampleNode(value=i) for i in range(5)]


@pytest.fixture
def pile_5(nodes_5):
    """Pile of five TestNodes."""
    return Pile(collections=nodes_5, item_type={SampleNode})


@pytest.fixture
def empty_pile():
    """Empty typed Pile."""
    return Pile(collections=[], item_type={SampleNode})


# ---------------------------------------------------------------------------
# pile.filter(predicate) - named method
# ---------------------------------------------------------------------------


class TestFilterMethod:
    """Tests for pile.filter(predicate)."""

    def test_filter_simple_predicate(self, pile_5, nodes_5):
        """Filter by a simple value comparison."""
        result = pile_5.filter(lambda x: x.value > 2)
        assert len(result) == 2
        assert all(item.value > 2 for item in result)

    def test_filter_returns_pile(self, pile_5):
        """filter() must return a Pile instance, not a list."""
        result = pile_5.filter(lambda x: x.value > 0)
        assert isinstance(result, Pile)

    def test_filter_returns_empty_pile_when_no_match(self, pile_5):
        """Predicate matching nothing returns an empty Pile."""
        result = pile_5.filter(lambda x: x.value > 100)
        assert isinstance(result, Pile)
        assert len(result) == 0
        assert result.is_empty()

    def test_filter_returns_all_items_when_all_match(self, pile_5, nodes_5):
        """Predicate matching everything returns a Pile with all items."""
        result = pile_5.filter(lambda x: x.value >= 0)
        assert len(result) == len(nodes_5)
        for node in nodes_5:
            assert node.id in result

    def test_filter_preserves_order(self, pile_5, nodes_5):
        """Filtered Pile respects insertion order of the source Pile."""
        result = pile_5.filter(lambda x: x.value % 2 == 0)
        values = [item.value for item in result]
        assert values == [0, 2, 4]

    def test_filter_does_not_mutate_source(self, pile_5):
        """Filtering creates a new Pile without altering the original."""
        original_len = len(pile_5)
        original_ids = list(pile_5.keys())
        _ = pile_5.filter(lambda x: x.value > 2)
        assert len(pile_5) == original_len
        assert list(pile_5.keys()) == original_ids

    def test_filter_on_empty_pile(self, empty_pile):
        """Filtering an empty Pile returns an empty Pile."""
        result = empty_pile.filter(lambda x: True)
        assert isinstance(result, Pile)
        assert len(result) == 0

    def test_filter_preserves_item_type(self, pile_5):
        """Filtered Pile carries over item_type from the source."""
        result = pile_5.filter(lambda x: x.value < 3)
        assert result.item_type == pile_5.item_type

    def test_filter_preserves_strict_type(self):
        """Filtered Pile carries over strict_type from the source."""
        nodes = [SampleNode(value=i) for i in range(3)]
        p = Pile(collections=nodes, item_type={SampleNode}, strict_type=True)
        result = p.filter(lambda x: x.value > 0)
        assert result.strict_type is True
        assert result.item_type == {SampleNode}

    def test_filter_single_match(self, pile_5, nodes_5):
        """Filtering down to exactly one item still returns a Pile."""
        result = pile_5.filter(lambda x: x.value == 3)
        assert isinstance(result, Pile)
        assert len(result) == 1
        assert list(result)[0].value == 3

    def test_filter_with_equality_predicate(self, pile_5, nodes_5):
        """Filter using identity / id-based predicate."""
        target_id = nodes_5[2].id
        result = pile_5.filter(lambda x: x.id == target_id)
        assert len(result) == 1
        assert list(result)[0].id == target_id


# ---------------------------------------------------------------------------
# pile[lambda x: ...] - callable key in __getitem__
# ---------------------------------------------------------------------------


class TestGetitemCallable:
    """Tests for callable key dispatch in __getitem__."""

    def test_getitem_lambda_returns_pile(self, pile_5):
        """pile[lambda] must return a Pile, not a list."""
        result = pile_5[lambda x: x.value > 2]
        assert isinstance(result, Pile)

    def test_getitem_lambda_filters_correctly(self, pile_5):
        """pile[lambda] filters with the same semantics as filter()."""
        result = pile_5[lambda x: x.value < 2]
        assert len(result) == 2
        values = [item.value for item in result]
        assert values == [0, 1]

    def test_getitem_lambda_empty_result(self, pile_5):
        """pile[lambda] returning no matches gives empty Pile."""
        result = pile_5[lambda x: x.value == 999]
        assert isinstance(result, Pile)
        assert len(result) == 0

    def test_getitem_lambda_all_match(self, pile_5, nodes_5):
        """pile[lambda] matching all items gives full Pile."""
        result = pile_5[lambda x: True]
        assert len(result) == len(nodes_5)

    def test_getitem_lambda_preserves_item_type(self, pile_5):
        """pile[lambda] carries over item_type."""
        result = pile_5[lambda x: x.value > 0]
        assert result.item_type == pile_5.item_type

    def test_getitem_lambda_preserves_strict_type(self):
        """pile[lambda] carries over strict_type."""
        nodes = [SampleNode(value=i) for i in range(3)]
        p = Pile(collections=nodes, item_type={SampleNode}, strict_type=True)
        result = p[lambda x: x.value == 1]
        assert result.strict_type is True

    def test_getitem_def_function(self, pile_5):
        """pile[func] works with a regular def, not just lambdas."""

        def is_even(item):
            return item.value % 2 == 0

        result = pile_5[is_even]
        assert isinstance(result, Pile)
        values = [item.value for item in result]
        assert values == [0, 2, 4]

    def test_getitem_lambda_equivalent_to_filter(self, pile_5):
        """pile[pred] and pile.filter(pred) yield identical results."""
        pred = lambda x: x.value in (1, 3)
        via_getitem = pile_5[pred]
        via_filter = pile_5.filter(pred)
        assert list(via_getitem.keys()) == list(via_filter.keys())
        assert [i.value for i in via_getitem] == [i.value for i in via_filter]


# ---------------------------------------------------------------------------
# Non-callable keys still work (regression guard)
# ---------------------------------------------------------------------------


class TestGetitemNonCallable:
    """Ensure non-callable __getitem__ paths are unaffected by callable dispatch."""

    def test_getitem_int_index(self, pile_5, nodes_5):
        """Integer index returns a single item."""
        item = pile_5[0]
        assert item.id == nodes_5[0].id

    def test_getitem_negative_index(self, pile_5, nodes_5):
        """Negative index works."""
        item = pile_5[-1]
        assert item.id == nodes_5[-1].id

    @pytest.mark.xfail(
        reason="Pre-existing Pile slice bug: Progression.__getitem__ returns "
        "Progression for slices but _getitem expects list",
        strict=True,
    )
    def test_getitem_slice(self, pile_5, nodes_5):
        """Slice returns multiple items."""
        items = pile_5[1:3]
        assert isinstance(items, list)
        assert len(items) == 2
        assert items[0].id == nodes_5[1].id
        assert items[1].id == nodes_5[2].id

    def test_getitem_uuid(self, pile_5, nodes_5):
        """UUID key returns the matching item."""
        target = nodes_5[3]
        item = pile_5[target.id]
        assert item.id == target.id
        assert item.value == target.value

    def test_getitem_element_ref(self, pile_5, nodes_5):
        """Element reference (the object itself) returns the item."""
        target = nodes_5[2]
        item = pile_5[target]
        assert item.id == target.id


# ---------------------------------------------------------------------------
# _filter_by_function internals
# ---------------------------------------------------------------------------


class TestFilterByFunction:
    """Direct tests on the private _filter_by_function method."""

    def test_filter_by_function_basic(self, pile_5):
        """_filter_by_function works the same as filter()."""
        result = pile_5._filter_by_function(lambda x: x.value >= 3)
        assert isinstance(result, Pile)
        assert len(result) == 2

    def test_filter_by_function_returns_new_pile(self, pile_5):
        """_filter_by_function returns a distinct Pile object."""
        result = pile_5._filter_by_function(lambda x: True)
        assert result is not pile_5

    def test_filter_by_function_preserves_order(self, pile_5):
        """Items in filtered Pile follow the source progression order."""
        result = pile_5._filter_by_function(lambda x: x.value in (4, 1, 2))
        values = [item.value for item in result]
        # source order is 0,1,2,3,4 so filtered order is 1,2,4
        assert values == [1, 2, 4]


# ---------------------------------------------------------------------------
# Complex predicates
# ---------------------------------------------------------------------------


class TestComplexPredicates:
    """Filter with predicates that inspect metadata or multiple fields."""

    def test_filter_by_metadata(self):
        """Filter using metadata attributes."""
        nodes = []
        for i in range(5):
            n = SampleNode(value=i)
            n.metadata["tag"] = "important" if i % 2 == 0 else "normal"
            nodes.append(n)

        p = Pile(collections=nodes)
        result = p.filter(lambda x: x.metadata.get("tag") == "important")
        assert len(result) == 3  # values 0, 2, 4
        for item in result:
            assert item.metadata["tag"] == "important"

    def test_filter_by_content(self):
        """Filter Node by content field."""
        nodes = [
            Node(content="alpha"),
            Node(content="beta"),
            Node(content="alpha-2"),
            Node(content="gamma"),
        ]
        p = Pile(collections=nodes)
        result = p.filter(lambda x: isinstance(x.content, str) and x.content.startswith("alpha"))
        assert len(result) == 2
        contents = [item.content for item in result]
        assert "alpha" in contents
        assert "alpha-2" in contents

    def test_filter_compound_predicate(self, pile_5):
        """Filter with a compound boolean predicate."""
        result = pile_5.filter(lambda x: x.value > 1 and x.value < 4)
        values = [item.value for item in result]
        assert values == [2, 3]

    def test_filter_by_id_substring(self, pile_5, nodes_5):
        """Filter by partial UUID string match."""
        target = nodes_5[0]
        prefix = str(target.id)[:8]
        result = pile_5.filter(lambda x: str(x.id).startswith(prefix))
        assert len(result) == 1
        assert list(result)[0].id == target.id

    def test_filter_chaining(self, pile_5):
        """Filters can be chained (filter result is itself a Pile)."""
        step1 = pile_5.filter(lambda x: x.value > 0)
        step2 = step1.filter(lambda x: x.value < 4)
        values = [item.value for item in step2]
        assert values == [1, 2, 3]

    def test_getitem_lambda_chaining(self, pile_5):
        """Callable __getitem__ results can also be chained."""
        step1 = pile_5[lambda x: x.value >= 2]
        step2 = step1[lambda x: x.value <= 3]
        values = [item.value for item in step2]
        assert values == [2, 3]


# ---------------------------------------------------------------------------
# Mixed-type Pile filtering
# ---------------------------------------------------------------------------


class OtherNode(Node):
    label: str = ""


class TestMixedTypePile:
    """Filter on Piles with heterogeneous types (no item_type constraint)."""

    def test_filter_by_isinstance(self):
        """Filter a mixed Pile to keep only TestNode instances."""
        items = [
            SampleNode(value=1),
            OtherNode(label="a"),
            SampleNode(value=2),
            OtherNode(label="b"),
        ]
        p = Pile(collections=items)
        result = p.filter(lambda x: isinstance(x, SampleNode))
        assert len(result) == 2
        assert all(isinstance(item, SampleNode) for item in result)

    def test_filter_mixed_preserves_order(self):
        """Filter on mixed Pile preserves insertion order."""
        items = [
            OtherNode(label="first"),
            SampleNode(value=10),
            OtherNode(label="second"),
            SampleNode(value=20),
            OtherNode(label="third"),
        ]
        p = Pile(collections=items)
        result = p.filter(lambda x: isinstance(x, OtherNode))
        labels = [item.label for item in result]
        assert labels == ["first", "second", "third"]
