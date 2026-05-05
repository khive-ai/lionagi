# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.libs.nested — get_target_container, nget, nset, npop,
flatten, unflatten covering previously uncovered branches."""

import pytest

from lionagi.libs.nested import (
    flatten,
    get_target_container,
    nget,
    npop,
    nset,
    unflatten,
)
from lionagi.utils import UNDEFINED

# ---------------------------------------------------------------------------
# get_target_container
# ---------------------------------------------------------------------------


class TestGetTargetContainer:
    def test_simple_dict_walk(self):
        data = {"a": {"b": 42}}
        result = get_target_container(data, ["a"])
        assert result == {"b": 42}

    def test_simple_list_walk(self):
        data = [[1, 2], [3, 4]]
        result = get_target_container(data, [0])
        assert result == [1, 2]

    def test_string_digit_index_on_list(self):
        """String digit "1" is coerced to int when traversing a list."""
        data = [10, 20, 30]
        result = get_target_container(data, ["1"])
        assert result == 20

    def test_invalid_list_index_raises_index_error(self):
        """Out-of-range integer index raises IndexError."""
        data = [1, 2, 3]
        with pytest.raises(IndexError):
            get_target_container(data, [99])

    def test_non_digit_string_on_list_raises_index_error(self):
        """Non-digit string index on a list raises IndexError."""
        data = [1, 2, 3]
        with pytest.raises(IndexError):
            get_target_container(data, ["notanindex"])

    def test_missing_dict_key_raises_key_error(self):
        """Missing dict key raises KeyError."""
        data = {"a": 1}
        with pytest.raises(KeyError):
            get_target_container(data, ["missing"])

    def test_non_list_non_dict_raises_type_error(self):
        """Traversing into a scalar raises TypeError."""
        data = {"a": 42}
        with pytest.raises(TypeError):
            get_target_container(data, ["a", "nested"])

    def test_empty_indices_returns_data(self):
        """Empty indices list returns the data unchanged."""
        data = {"x": 1}
        assert get_target_container(data, []) is data


# ---------------------------------------------------------------------------
# nget
# ---------------------------------------------------------------------------


class TestNget:
    def test_empty_indices_raises_value_error(self):
        with pytest.raises(ValueError):
            nget({"a": 1}, [])

    def test_basic_get(self):
        data = {"a": {"b": 7}}
        assert nget(data, ["a", "b"]) == 7

    def test_list_with_string_digit_index(self):
        data = {"items": [10, 20, 30]}
        assert nget(data, ["items", "2"]) == 30

    def test_not_found_with_default(self):
        data = {"a": 1}
        assert nget(data, ["missing"], default="fallback") == "fallback"

    def test_not_found_without_default_raises_key_error(self):
        data = {"a": 1}
        with pytest.raises(KeyError):
            nget(data, ["missing"])

    def test_type_error_in_traversal_with_default(self):
        """Traversing into a non-container returns default when provided."""
        data = {"a": 42}
        assert nget(data, ["a", "nested"], default="safe") == "safe"

    def test_type_error_in_traversal_without_default_raises_key_error(self):
        data = {"a": 42}
        with pytest.raises(KeyError):
            nget(data, ["a", "nested"])

    def test_index_error_with_default(self):
        data = [1, 2]
        assert nget(data, [99], default=None) is None

    def test_deep_nested(self):
        data = {"x": {"y": {"z": "deep"}}}
        assert nget(data, ["x", "y", "z"]) == "deep"


# ---------------------------------------------------------------------------
# nset
# ---------------------------------------------------------------------------


class TestNset:
    def test_basic_set_dict(self):
        data = {"a": {}}
        nset(data, ["a", "b"], 99)
        assert data["a"]["b"] == 99

    def test_basic_set_list(self):
        data = [0, 0, 0]
        nset(data, [1], 42)
        assert data[1] == 42

    def test_non_integer_index_on_list_raises_type_error(self):
        data = [1, 2, 3]
        with pytest.raises(TypeError):
            nset(data, ["bad_key"], 99)

    def test_integer_key_on_dict_raises_type_error(self):
        data = {}
        with pytest.raises(TypeError):
            nset(data, [0], "value")

    def test_non_list_non_dict_target_raises_type_error(self):
        """Trying to nest into a scalar container raises TypeError."""
        data = {"a": 42}
        with pytest.raises(TypeError):
            nset(data, ["a", "b"], "value")

    def test_non_integer_last_index_on_list_raises_type_error(self):
        data = [[1, 2, 3]]
        with pytest.raises(TypeError):
            nset(data, [0, "notint"], 99)

    def test_non_string_last_index_on_dict_raises_type_error(self):
        data = {"a": {}}
        with pytest.raises(TypeError):
            nset(data, ["a", 0], 99)

    def test_auto_creates_intermediate_dict(self):
        """nset creates intermediate dicts for missing string keys."""
        data = {}
        nset(data, ["x", "y", "z"], "created")
        assert data["x"]["y"]["z"] == "created"

    def test_auto_creates_intermediate_list(self):
        """nset creates intermediate list when next index is int."""
        data = {}
        nset(data, ["arr", 0], "first")
        assert data["arr"][0] == "first"

    def test_extends_list_with_nones(self):
        """nset extends list with None placeholders when index > len."""
        data = {"items": []}
        nset(data, ["items", 3], "fourth")
        assert data["items"][3] == "fourth"
        assert data["items"][0] is None

    def test_empty_indices_raises_value_error(self):
        with pytest.raises(ValueError):
            nset({}, [], "value")


# ---------------------------------------------------------------------------
# npop
# ---------------------------------------------------------------------------


class TestNpop:
    def test_basic_pop_dict(self):
        data = {"a": {"b": 99}}
        result = npop(data, ["a", "b"])
        assert result == 99
        assert "b" not in data["a"]

    def test_basic_pop_list(self):
        data = [10, 20, 30]
        result = npop(data, [1])
        assert result == 20
        assert data == [10, 30]

    def test_missing_key_with_default(self):
        data = {"a": 1}
        result = npop(data, ["missing"], default="fallback")
        assert result == "fallback"

    def test_missing_key_without_default_raises(self):
        data = {"a": 1}
        with pytest.raises(KeyError):
            npop(data, ["missing"])

    def test_index_out_of_range_with_default(self):
        data = [1, 2]
        result = npop(data, [99], default="safe")
        assert result == "safe"

    def test_index_out_of_range_without_default_raises(self):
        data = [1, 2]
        with pytest.raises(IndexError):
            npop(data, [99])

    def test_non_list_non_dict_raises_type_error(self):
        """npop on a non-list/dict final container raises TypeError."""
        data = {"a": 42}
        with pytest.raises(TypeError):
            npop(data, ["a", "nested"])

    def test_list_pop_with_string_digit(self):
        """String digit last index is coerced to int for list pop."""
        data = [10, 20, 30]
        result = npop(data, ["1"])
        assert result == 20

    def test_non_integer_list_index_raises_type_error(self):
        """Non-integer, non-digit string index raises TypeError for list pop."""
        data = [1, 2, 3]
        with pytest.raises(TypeError):
            npop(data, ["notdigit"])

    def test_empty_indices_raises_value_error(self):
        with pytest.raises(ValueError):
            npop({}, [])


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flat_dict(self):
        data = {"a": 1, "b": 2}
        result = flatten(data)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        data = {"a": {"b": {"c": 3}}}
        result = flatten(data)
        assert result["a|b|c"] == 3

    def test_nested_list(self):
        data = {"items": [10, 20, 30]}
        result = flatten(data)
        assert result["items|0"] == 10
        assert result["items|2"] == 30

    def test_max_depth_stops_early(self):
        """max_depth=1 stops traversal at depth 1."""
        data = {"a": {"b": {"c": 3}}}
        result = flatten(data, max_depth=1)
        # depth 1 reached at "a" level, so nested dict not expanded
        assert "a" in result
        assert isinstance(result["a"], dict)

    def test_preserve_empty_true_with_empty_mapping(self):
        """preserve_empty=True keeps empty dicts."""
        data = {"a": {}}
        result = flatten(data, preserve_empty=True)
        assert "a" in result
        assert result["a"] == {}

    def test_preserve_empty_true_with_empty_list(self):
        """preserve_empty=True keeps empty lists."""
        data = {"b": []}
        result = flatten(data, preserve_empty=True)
        assert "b" in result
        assert result["b"] == []

    def test_non_container_scalar(self):
        """A bare scalar produces a single entry with empty-string key."""
        result = flatten(42)
        assert result == {"": 42}

    def test_string_not_traversed(self):
        """Strings are treated as scalars, not sequences."""
        data = {"msg": "hello"}
        result = flatten(data)
        assert result == {"msg": "hello"}

    def test_bytes_not_traversed(self):
        """bytes are treated as scalars."""
        data = {"raw": b"\x00\x01"}
        result = flatten(data)
        assert result["raw"] == b"\x00\x01"

    def test_nested_list_with_nested_dict(self):
        """List containing dicts is recursively flattened."""
        data = {"rows": [{"x": 1}, {"x": 2}]}
        result = flatten(data)
        assert result["rows|0|x"] == 1
        assert result["rows|1|x"] == 2

    def test_custom_separator(self):
        data = {"a": {"b": 5}}
        result = flatten(data, sep=".")
        assert "a.b" in result

    def test_preserve_empty_false_skips_empty(self):
        """preserve_empty=False omits empty containers."""
        data = {"a": {}, "b": 1}
        result = flatten(data, preserve_empty=False)
        assert "a" not in result
        assert result["b"] == 1


# ---------------------------------------------------------------------------
# unflatten
# ---------------------------------------------------------------------------


class TestUnflatten:
    def test_basic_unflatten(self):
        flat = {"a|b": 1, "a|c": 2}
        result = unflatten(flat)
        assert result == {"a": {"b": 1, "c": 2}}

    def test_list_inference(self):
        """Consecutive integer string keys at non-root level become a list."""
        flat = {"items|0": "x", "items|1": "y", "items|2": "z"}
        result = unflatten(flat)
        assert result["items"] == ["x", "y", "z"]

    def test_non_consecutive_int_keys_not_converted_to_list(self):
        """Non-consecutive integer keys should NOT convert to list."""
        flat = {"m|0": "a", "m|2": "b"}
        result = unflatten(flat)
        # keys 0 and 2 are not consecutive (missing 1) → stays dict
        assert isinstance(result["m"], dict)

    def test_root_level_int_keys_not_converted_to_list(self):
        """root=True means the top level is always a dict, even for integer keys."""
        flat = {"0": "a", "1": "b", "2": "c"}
        result = unflatten(flat)
        # root level is never converted to list
        assert isinstance(result, dict)

    def test_custom_separator(self):
        flat = {"a.b.c": 99}
        result = unflatten(flat, sep=".")
        assert result["a"]["b"]["c"] == 99

    def test_infer_lists_false_keeps_dicts(self):
        """infer_lists=False prevents any integer-key dicts from becoming lists."""
        flat = {"a|0": "x", "a|1": "y"}
        result = unflatten(flat, infer_lists=False)
        assert isinstance(result["a"], dict)

    def test_roundtrip_dict(self):
        original = {"x": {"y": {"z": 42}}, "a": [1, 2, 3]}
        flat = flatten(original)
        restored = unflatten(flat)
        assert restored["x"]["y"]["z"] == 42
        assert restored["a"] == [1, 2, 3]
