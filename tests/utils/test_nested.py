# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionagi.libs.nested import (
    flatten,
    unflatten,
    nget,
    nset,
    npop,
    ninsert,
    nmerge,
    nfilter,
    get_target_container,
    ensure_list_index,
    is_homogeneous,
    is_same_dtype,
    is_structure_homogeneous,
    deep_update,
)


# Test flatten/unflatten
def test_flatten_basic():
    """Test basic flattening of nested dictionaries."""
    nested = {"a": 1, "b": {"c": 2, "d": [3, 4]}}
    expected = {'a': 1, 'b|c': 2, 'b|d|0': 3, 'b|d|1': 4}
    assert flatten(nested) == expected


def test_flatten_with_max_depth():
    """Test flattening with max_depth parameter."""
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    expected = {'a': 1, 'b|c': 2, 'b|d': {"e": 3}}
    assert flatten(nested, max_depth=2) == expected


def test_unflatten_basic():
    """Test basic unflattening of flat dictionaries."""
    flat = {'a': 1, 'b|c': 2, 'b|d|0': 3, 'b|d|1': 4}
    expected = {"a": 1, "b": {"c": 2, "d": [3, 4]}}
    assert unflatten(flat) == expected


def test_unflatten_inplace():
    """Test unflattening with inplace=True."""
    flat = {'a': 1, 'b|c': 2, 'b|d|0': 3, 'b|d|1': 4}
    expected = {"a": 1, "b": {"c": 2, "d": [3, 4]}}
    result = unflatten(flat, inplace=True)
    assert result == expected
    # The flat dictionary is modified in place to match the result
    assert flat == expected


# Test nget/nset/npop
def test_nget_basic():
    """Test basic nested get operation."""
    nested = {"a": {"b": {"c": 3}}}
    assert nget(nested, ["a", "b", "c"]) == 3
    assert nget(nested, ["a", "b"]) == {"c": 3}


def test_nget_with_default():
    """Test nested get with default value."""
    nested = {"a": {"b": 2}}
    assert nget(nested, ["a", "c"], default=10) == 10


def test_nset_basic():
    """Test basic nested set operation."""
    nested = {"a": {"b": 2}}
    nset(nested, ["a", "c"], 3)
    assert nested == {"a": {"b": 2, "c": 3}}


def test_npop_basic():
    """Test basic nested pop operation."""
    nested = {"a": {"b": 2, "c": 3}}
    result = npop(nested, ["a", "c"])
    assert result == 3
    assert nested == {"a": {"b": 2}}


def test_npop_with_default():
    """Test nested pop with default value."""
    nested = {"a": {"b": 2}}
    result = npop(nested, ["a", "c"], default=10)
    assert result == 10
    assert nested == {"a": {"b": 2}}


# Test ninsert/nmerge/nfilter
def test_ninsert_basic():
    """Test basic nested insert operation."""
    nested = {"a": [1, 2]}
    ninsert(nested, ["a", 1], 3)
    assert nested == {"a": [1, 3, 2]}


def test_nmerge_basic():
    """Test basic nested merge operation."""
    nested1 = {"a": {"b": 1}}
    nested2 = {"a": {"c": 2}}
    result = nmerge(nested1, nested2)
    assert result == {"a": {"b": 1, "c": 2}}
    assert result is nested1  # inplace by default


def test_nmerge_not_inplace():
    """Test nested merge with inplace=False."""
    nested1 = {"a": {"b": 1}}
    nested2 = {"a": {"c": 2}}
    result = nmerge(nested1, nested2, inplace=False)
    assert result == {"a": {"b": 1, "c": 2}}
    assert result is not nested1


def test_nfilter_basic():
    """Test basic nested filter operation."""
    nested = {"a": 1, "b": {"c": 2, "d": 3}}
    result = nfilter(nested, lambda p, k, v: isinstance(v, int) and v > 1)
    assert result == {"b": {"c": 2, "d": 3}}


# Test edge cases
def test_empty_structures():
    """Test handling of empty structures."""
    assert flatten({}) == {}
    assert unflatten({}) == {}
    
    # Test with empty nested structures
    nested = {"a": {}, "b": []}
    flat = flatten(nested)
    assert unflatten(flat) == nested


def test_nested_lists():
    """Test handling of nested lists."""
    nested = [1, [2, [3, 4]]]
    flat = flatten(nested, dynamic=True)
    assert flat == {'0': 1, '1|0': 2, '1|1|0': 3, '1|1|1': 4}
    
    # Test with coerce_sequence="dict"
    flat_dict = flatten(nested, dynamic=True, coerce_sequence="dict")
    assert flat_dict == {'0': 1, '1|0': 2, '1|1|0': 3, '1|1|1': 4}


def test_mixed_types():
    """Test handling of mixed types."""
    nested = {"a": 1, "b": [2, {"c": 3}]}
    flat = flatten(nested)
    assert flat == {'a': 1, 'b|0': 2, 'b|1|c': 3}
    assert unflatten(flat) == nested


# Test helper functions
def test_get_target_container():
    """Test get_target_container function."""
    nested = {"a": {"b": [1, 2, 3]}}
    container = get_target_container(nested, ["a", "b"])
    assert container == [1, 2, 3]


def test_ensure_list_index():
    """Test ensure_list_index function."""
    lst = [1, 2]
    ensure_list_index(lst, 4)
    assert lst == [1, 2, None, None, None]


def test_is_homogeneous():
    """Test is_homogeneous function."""
    assert is_homogeneous([1, 2, 3], int) is True
    assert is_homogeneous([1, "2", 3], int) is False
    assert is_homogeneous({"a": 1, "b": 2}, int) is True
    assert is_homogeneous({"a": 1, "b": "2"}, int) is False


def test_is_same_dtype():
    """Test is_same_dtype function."""
    assert is_same_dtype([1, 2, 3]) is True
    assert is_same_dtype([1, "2", 3]) is False
    assert is_same_dtype({"a": 1, "b": 2}) is True
    assert is_same_dtype({"a": 1, "b": "2"}) is False


def test_is_structure_homogeneous():
    """Test is_structure_homogeneous function."""
    assert is_structure_homogeneous({"a": {"b": 1}, "c": {"d": 2}}) is True
    assert is_structure_homogeneous({"a": {"b": 1}, "c": [1, 2]}) is False


def test_deep_update():
    """Test deep_update function."""
    original = {"a": 1, "b": {"c": 2}}
    update = {"b": {"d": 3}}
    result = deep_update(original, update)
    assert result == {"a": 1, "b": {"c": 2, "d": 3}}
    assert result is original  # inplace by default