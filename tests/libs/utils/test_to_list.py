"""Test suite for to_list - TDD Specification Implementation."""

import msgspec
import pytest
from pydantic import BaseModel

from lionagi.ln._to_list import to_list
from lionagi.ln.types import Undefined, Unset


class TestToListCoercion:
    """TestSuite: ToListCoercion - Correct coercion and complex interactions."""

    def test_handling_none_and_sentinels(self):
        """Test: HandlingNoneAndSentinels

        GIVEN input=None, Undefined, or Unset
        THEN to_list(input) returns [].
        """
        assert to_list(None) == [], "to_list(None) must return []"
        assert to_list(Undefined) == [], "to_list(Undefined) must return []"
        assert to_list(Unset) == [], "to_list(Unset) must return []"

    def test_handling_scalars_and_strings(self):
        """Test: HandlingScalarsAndStrings

        GIVEN input=1 THEN returns [1].
        GIVEN input="abc" THEN returns ["abc"] (not ['a', 'b', 'c']).
        """
        assert to_list(1) == [1], "to_list(1) must return [1]"
        assert to_list(3.14) == [3.14], "to_list(3.14) must return [3.14]"
        assert to_list(True) == [True], "to_list(True) must return [True]"
        assert to_list("abc") == [
            "abc"
        ], "to_list('abc') must return ['abc'] not ['a', 'b', 'c']"

        # Test that use_values affects strings
        assert to_list("abc", use_values=True) == [
            "a",
            "b",
            "c",
        ], "to_list('abc', use_values=True) must return list of chars"

    def test_handling_mappings_and_structs(self):
        """Test: HandlingMappingsAndStructs

        GIVEN input={"a": 1} or MyStruct(a=1)
        WHEN use_values=False THEN returns [input] (treated as atom).
        WHEN use_values=True (for mappings) THEN returns [1].
        """
        # Test with dict
        dict_input = {"a": 1, "b": 2}
        assert to_list(dict_input, use_values=False) == [
            dict_input
        ], "Dict with use_values=False must return [dict]"
        assert to_list(dict_input, use_values=True) == [
            1,
            2,
        ], "Dict with use_values=True must return list of values"

        # Test with msgspec.Struct
        class MyStruct(msgspec.Struct):
            a: int
            b: int

        struct_input = MyStruct(a=1, b=2)
        assert to_list(struct_input) == [
            struct_input
        ], "msgspec.Struct must return [struct]"

        # Test with Pydantic model
        class MyModel(BaseModel):
            a: int
            b: int

        model_input = MyModel(a=1, b=2)
        assert to_list(model_input) == [
            model_input
        ], "Pydantic model must return [model]"

    def test_handling_iterables(self):
        """Test handling of various iterable types."""
        # List
        assert to_list([1, 2, 3]) == [
            1,
            2,
            3,
        ], "List input should return same list"

        # Tuple
        assert to_list((1, 2, 3)) == [
            1,
            2,
            3,
        ], "Tuple input should return list"

        # Set
        result = to_list({1, 2, 3})
        assert set(result) == {
            1,
            2,
            3,
        }, "Set input should return list with same elements"
        assert len(result) == 3, "Set input should preserve all elements"

        # Range
        assert to_list(range(3)) == [0, 1, 2], "Range input should return list"

        # Generator
        gen = (x for x in [1, 2, 3])
        assert to_list(gen) == [1, 2, 3], "Generator input should return list"


class TestToListTransformations:
    """TestSuite: ToListTransformations - flatten, dropna, unique transformations."""

    def test_flatten_recursive(self):
        """Test: FlattenRecursive

        GIVEN input=[1, [2, [3]]]
        WHEN flatten=True THEN returns [1, 2, 3].
        """
        assert to_list([1, [2, [3]]], flatten=True) == [
            1,
            2,
            3,
        ], "Nested lists must flatten completely"
        assert to_list([[1, 2], [3, [4, 5]]], flatten=True) == [
            1,
            2,
            3,
            4,
            5,
        ], "Complex nested lists must flatten"

        # Test that flatten=False preserves structure
        assert to_list([1, [2, [3]]], flatten=False) == [
            1,
            [2, [3]],
        ], "flatten=False must preserve structure"

    def test_flatten_skip_types(self):
        """Test: FlattenSkipTypes (CRITICAL)

        # Strings, Mappings, Structs should not be recursively flattened
        GIVEN input=["ab", {"k": 1}, [MyStruct(f=1)]]
        WHEN flatten=True THEN returns ["ab", {"k": 1}, MyStruct(f=1)].
        """

        class MyStruct(msgspec.Struct):
            f: int

        struct_obj = MyStruct(f=1)
        input_list = ["ab", {"k": 1}, [struct_obj]]

        result = to_list(input_list, flatten=True)
        assert result == [
            "ab",
            {"k": 1},
            struct_obj,
        ], "Strings, mappings, and structs must not be flattened"

        # Test with Pydantic model too
        class MyModel(BaseModel):
            f: int

        model_obj = MyModel(f=1)
        input_with_model = ["xy", {"m": 2}, [model_obj]]

        result2 = to_list(input_with_model, flatten=True)
        assert result2 == [
            "xy",
            {"m": 2},
            model_obj,
        ], "Pydantic models must not be flattened"

    def test_flatten_tuple_set_control(self):
        """Test: FlattenTupleSetControl

        GIVEN input=[(1, 2)]
        WHEN flatten=True, flatten_tuple_set=False THEN returns [(1, 2)].
        WHEN flatten=True, flatten_tuple_set=True THEN returns [1, 2].
        """
        # Test with tuples
        assert to_list([(1, 2)], flatten=True, flatten_tuple_set=False) == [
            (1, 2)
        ], "Tuples not flattened when flatten_tuple_set=False"
        assert to_list([(1, 2)], flatten=True, flatten_tuple_set=True) == [
            1,
            2,
        ], "Tuples flattened when flatten_tuple_set=True"

        # Test with sets
        input_with_set = [{1, 2}]
        result1 = to_list(
            input_with_set, flatten=True, flatten_tuple_set=False
        )
        assert result1 == [
            {1, 2}
        ], "Sets not flattened when flatten_tuple_set=False"

        result2 = to_list(input_with_set, flatten=True, flatten_tuple_set=True)
        assert set(result2) == {
            1,
            2,
        }, "Sets flattened when flatten_tuple_set=True"

        # Test with frozensets
        input_with_frozenset = [frozenset([3, 4])]
        result3 = to_list(
            input_with_frozenset, flatten=True, flatten_tuple_set=False
        )
        assert result3 == [
            frozenset([3, 4])
        ], "Frozensets not flattened when flatten_tuple_set=False"

        result4 = to_list(
            input_with_frozenset, flatten=True, flatten_tuple_set=True
        )
        assert set(result4) == {
            3,
            4,
        }, "Frozensets flattened when flatten_tuple_set=True"

    def test_dropna_behavior(self):
        """Test: DropNaBehavior

        GIVEN input=[1, None, 2, Undefined, 3, Unset]
        WHEN dropna=True THEN returns [1, 2, 3].
        """
        input_list = [1, None, 2, Undefined, 3, Unset]
        assert to_list(input_list, dropna=True) == [
            1,
            2,
            3,
        ], "dropna must remove None and sentinels"

        # Test with nested structures
        nested = [1, [None, 2], [[Undefined, 3]]]
        assert to_list(nested, flatten=True, dropna=True) == [
            1,
            2,
            3,
        ], "dropna must work with flatten"

        # Test that dropna=False preserves all values
        assert (
            to_list(input_list, dropna=False) == input_list
        ), "dropna=False must preserve all values"


class TestToListUniqueness:
    """TestSuite: ToListUniqueness (CRITICAL) - Testing unique parameter with complex types."""

    def test_unique_requires_flatten_validation(self):
        """Test: UniqueRequiresFlattenValidation

        WHEN unique=True, flatten=False
        THEN must raise ValueError.
        """
        with pytest.raises(
            ValueError, match="unique=True requires flatten=True"
        ):
            to_list([1, 2, 1], unique=True, flatten=False)

    def test_unique_hashable_types(self):
        """Test unique with simple hashable types."""
        assert to_list([1, 2, 1, 3, 2], flatten=True, unique=True) == [
            1,
            2,
            3,
        ], "Unique must remove duplicates"
        assert to_list(["a", "b", "a", "c"], flatten=True, unique=True) == [
            "a",
            "b",
            "c",
        ], "Unique must work with strings"

        # Test preservation of first occurrence
        assert to_list([3, 2, 1, 2, 3], flatten=True, unique=True) == [
            3,
            2,
            1,
        ], "Unique must preserve first occurrence order"

    def test_unique_unhashable_types(self):
        """Test: UniqueUnhashableTypes (Integration with hash_dict)

        GIVEN input=[{"a": 1}, {"b": 2}, {"a": 1}, MyStruct(f=1), MyStruct(f=1)]
        WHEN flatten=True, unique=True
        THEN returns [{"a": 1}, {"b": 2}, MyStruct(f=1)].
        """

        class MyStruct(msgspec.Struct):
            f: int

        struct_obj1 = MyStruct(f=1)
        struct_obj2 = MyStruct(f=1)  # Same content

        input_list = [{"a": 1}, {"b": 2}, {"a": 1}, struct_obj1, struct_obj2]
        result = to_list(input_list, flatten=True, unique=True)

        assert (
            len(result) == 3
        ), "Unique must remove duplicates of unhashable types"
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}
        assert isinstance(result[2], MyStruct) and result[2].f == 1

        # Test with Pydantic models
        class MyModel(BaseModel):
            f: int

        model1 = MyModel(f=1)
        model2 = MyModel(f=1)

        input_with_models = [model1, {"x": 1}, model2, {"x": 1}]
        result2 = to_list(input_with_models, flatten=True, unique=True)

        assert len(result2) == 2, "Unique must handle Pydantic models"

    def test_unique_unhashable_order_insensitivity(self):
        """Test: UniqueUnhashableOrderInsensitivity

        GIVEN input=[{"k1": 1, "k2": 2}, {"k2": 2, "k1": 1}]
        WHEN flatten=True, unique=True
        THEN returns [{"k1": 1, "k2": 2}] (only the first occurrence).
        """
        input_list = [{"k1": 1, "k2": 2}, {"k2": 2, "k1": 1}]
        result = to_list(input_list, flatten=True, unique=True)

        assert (
            len(result) == 1
        ), "Dicts with same content but different key order must be considered duplicates"
        assert result[0] == {
            "k1": 1,
            "k2": 2,
        }, "First occurrence must be preserved"

        # Test with nested dicts
        nested1 = {"outer": {"a": 1, "b": 2}, "c": 3}
        nested2 = {"c": 3, "outer": {"b": 2, "a": 1}}

        result2 = to_list([nested1, nested2], flatten=True, unique=True)
        assert (
            len(result2) == 1
        ), "Nested dicts with same content must be considered duplicates"

    def test_complex_transformation_combinations(self):
        """Test complex combinations of transformations."""
        # Complex nested structure with duplicates and None values
        input_data = [
            1,
            None,
            [2, None, [3, 1]],
            {"a": 1},
            [{"a": 1}],
            "test",
            ["test"],
            Undefined,
            [Unset],
        ]

        # Apply all transformations
        result = to_list(input_data, flatten=True, dropna=True, unique=True)

        # Should have: 1, 2, 3, {"a": 1}, "test"
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert {"a": 1} in result
        assert "test" in result
        assert None not in result
        assert Undefined not in result
        assert Unset not in result
        assert (
            len(result) == 5
        ), f"Expected 5 unique elements, got {len(result)}: {result}"

    def test_error_handling_unhashable(self):
        """Test error handling for types that can't be hashed."""

        # Create a truly unhashable object that also can't be handled by hash_dict
        class BadObject:
            def __hash__(self):
                raise TypeError("Cannot hash")

            def __eq__(self, other):
                raise TypeError("Cannot compare")

        bad_obj = BadObject()

        # This should raise a clear error
        with pytest.raises(ValueError, match="Unhashable type encountered"):
            to_list([bad_obj, bad_obj], flatten=True, unique=True)
