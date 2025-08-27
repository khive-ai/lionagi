"""Test suite for hash_dict - TDD Specification Implementation."""

import msgspec
import pytest
from pydantic import BaseModel

from lionagi.ln._hash import hash_dict


class TestHashDictStabilityAndSensitivity:
    """TestSuite: HashDictStabilityAndSensitivity - Stability, order-insensitivity, type differentiation, msgspec integration."""

    def test_idempotence(self):
        """Test: Idempotence

        GIVEN a complex input object D
        WHEN calling hash_dict(D) multiple times
        THEN the result must be identical each time.
        """
        complex_obj = {
            "nested": {"a": 1, "b": [2, 3]},
            "list": [1, 2, {"inner": True}],
            "set": {4, 5, 6},
            "tuple": (7, 8, 9),
        }

        hash1 = hash_dict(complex_obj)
        hash2 = hash_dict(complex_obj)
        hash3 = hash_dict(complex_obj)

        assert hash1 == hash2 == hash3, "hash_dict must be idempotent"

    def test_dictionary_order_insensitivity(self):
        """Test: DictionaryOrderInsensitivity (CRITICAL)

        GIVEN D1 = {"a": 1, "b": 2} AND D2 = {"b": 2, "a": 1}
        WHEN hashed
        THEN hash_dict(D1) must equal hash_dict(D2).
        """
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}

        assert hash_dict(d1) == hash_dict(d2), "Dictionary order must not affect hash"

        # Test with nested dictionaries
        d3 = {"outer": {"x": 1, "y": 2}, "z": 3}
        d4 = {"z": 3, "outer": {"y": 2, "x": 1}}

        assert hash_dict(d3) == hash_dict(d4), "Nested dictionary order must not affect hash"

    def test_set_order_insensitivity(self):
        """Test: SetOrderInsensitivity

        GIVEN S1 = {1, 2} AND S2 = {2, 1}
        WHEN hashed
        THEN hash_dict(S1) must equal hash_dict(S2).
        """
        s1 = {1, 2, 3}
        s2 = {3, 1, 2}

        assert hash_dict(s1) == hash_dict(s2), "Set order must not affect hash"

        # Test frozenset as well
        fs1 = frozenset([1, 2, 3])
        fs2 = frozenset([3, 1, 2])

        assert hash_dict(fs1) == hash_dict(fs2), "Frozenset order must not affect hash"

    def test_list_order_sensitivity(self):
        """Test: List/TupleOrderSensitivity

        GIVEN L1 = [1, 2] AND L2 = [2, 1]
        WHEN hashed
        THEN hash_dict(L1) must NOT equal hash_dict(L2).
        """
        l1 = [1, 2]
        l2 = [2, 1]

        assert hash_dict(l1) != hash_dict(l2), "List order must affect hash"

    def test_tuple_order_sensitivity(self):
        """Test: TupleOrderSensitivity

        GIVEN T1 = (1, 2) AND T2 = (2, 1)
        WHEN hashed
        THEN hash_dict(T1) must NOT equal hash_dict(T2).
        """
        t1 = (1, 2)
        t2 = (2, 1)

        assert hash_dict(t1) != hash_dict(t2), "Tuple order must affect hash"

    def test_type_differentiation(self):
        """Test: TypeDifferentiation (CRITICAL: Collision Avoidance)

        GIVEN L = [1, 2] AND T = (1, 2)
        WHEN hashed
        THEN hash_dict(L) must NOT equal hash_dict(T) (Verify type markers).
        """
        list_obj = [1, 2]
        tuple_obj = (1, 2)

        assert hash_dict(list_obj) != hash_dict(
            tuple_obj
        ), "List and tuple with same content must have different hashes"

        # Also test set vs frozenset
        set_obj = {1, 2}
        frozenset_obj = frozenset([1, 2])

        assert hash_dict(set_obj) != hash_dict(
            frozenset_obj
        ), "Set and frozenset with same content must have different hashes"

        # Test dict vs list of tuples
        dict_obj = {"a": 1, "b": 2}
        list_of_tuples = [("a", 1), ("b", 2)]

        assert hash_dict(dict_obj) != hash_dict(
            list_of_tuples
        ), "Dict and list of tuples must have different hashes"

    def test_msgspec_integration(self):
        """Test: MsgspecIntegration (V1 Requirement)

        GIVEN MyStruct = msgspec.Struct(a=1, b=2)
        AND DictD = {"a": 1, "b": 2}
        WHEN hashed
        THEN hash_dict(MyStruct) must NOT equal hash_dict(DictD) (Structs are distinct).
        AND two instances of the same Struct content must hash equally.
        """

        # Define a msgspec Struct
        class MyStruct(msgspec.Struct):
            a: int
            b: int

        struct1 = MyStruct(a=1, b=2)
        struct2 = MyStruct(a=1, b=2)
        dict_obj = {"a": 1, "b": 2}

        # Structs are distinct from dicts
        assert hash_dict(struct1) != hash_dict(
            dict_obj
        ), "msgspec.Struct must be distinct from dict"

        # Two instances with same content must hash equally
        assert hash_dict(struct1) == hash_dict(
            struct2
        ), "Same msgspec.Struct content must hash equally"

        # Different content must hash differently
        struct3 = MyStruct(a=2, b=1)
        assert hash_dict(struct1) != hash_dict(
            struct3
        ), "Different msgspec.Struct content must hash differently"

    def test_pydantic_model_handling(self):
        """Test that Pydantic models are handled correctly."""

        class MyModel(BaseModel):
            a: int
            b: int

        model1 = MyModel(a=1, b=2)
        model2 = MyModel(a=1, b=2)
        dict_obj = {"a": 1, "b": 2}

        # Pydantic models are distinct from dicts
        assert hash_dict(model1) != hash_dict(dict_obj), "Pydantic model must be distinct from dict"

        # Two instances with same content must hash equally
        assert hash_dict(model1) == hash_dict(
            model2
        ), "Same Pydantic model content must hash equally"

    def test_handling_mixed_type_sets(self):
        """Test: HandlingMixedTypeSets (Edge Case)

        Tests fallback sorting when elements aren't directly comparable
        GIVEN S1 = {1, "a"} AND S2 = {"a", 1}
        WHEN hashed
        THEN hash_dict(S1) must equal hash_dict(S2).
        """
        s1 = {1, "a", 2.5, True}
        s2 = {True, 2.5, "a", 1}

        assert hash_dict(s1) == hash_dict(
            s2
        ), "Mixed type sets must hash consistently regardless of order"

        # Test with more complex mixed types
        s3 = {1, "hello", (1, 2), None}
        s4 = {None, (1, 2), 1, "hello"}

        assert hash_dict(s3) == hash_dict(s4), "Complex mixed type sets must hash consistently"

    def test_deep_nesting(self):
        """Test hashing of deeply nested structures."""
        nested1 = {"level1": {"level2": {"level3": {"data": [1, 2, {"inner": True}]}}}}

        nested2 = {"level1": {"level2": {"level3": {"data": [1, 2, {"inner": True}]}}}}

        assert hash_dict(nested1) == hash_dict(
            nested2
        ), "Identical deep structures must hash equally"

        # Change something deep
        nested3 = {"level1": {"level2": {"level3": {"data": [1, 2, {"inner": False}]}}}}  # Changed

        assert hash_dict(nested1) != hash_dict(
            nested3
        ), "Different deep structures must hash differently"

    def test_strict_mode(self):
        """Test the strict mode parameter."""
        original = {"a": [1, 2], "b": {"c": 3}}

        # Non-strict mode should not modify original
        hash1 = hash_dict(original, strict=False)
        assert original == {"a": [1, 2], "b": {"c": 3}}

        # Strict mode should deepcopy first (original unchanged)
        hash2 = hash_dict(original, strict=True)
        assert original == {"a": [1, 2], "b": {"c": 3}}

        # Hashes should be the same
        assert hash1 == hash2, "Strict and non-strict modes should produce same hash"

    def test_error_handling(self):
        """Test that proper errors are raised for invalid inputs."""

        # Custom object that can't be hashed properly
        class UnhashableCustom:
            def __str__(self):
                raise Exception("Cannot stringify")

            def __repr__(self):
                raise Exception("Cannot repr")

        # This should handle the error gracefully and still produce a hash
        # (through the fallback mechanism in _generate_hashable_representation)
        obj = UnhashableCustom()
        try:
            hash_dict(obj)
        except TypeError as e:
            assert "not hashable" in str(e).lower()
