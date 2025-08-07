# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""Test the singleton sentinel infrastructure."""

import copy
import pickle
from typing import Any

import pytest

from lionagi.utils import (
    UNDEFINED,
    SingletonType,
    Undefined,
    UndefinedType,
    Unset,
    UnsetType,
    is_sentinel,
    is_undefined,
    is_unset,
    not_sentinel,
)


class TestSingletonMeta:
    """Test the singleton metaclass behavior."""

    def test_singleton_identity(self):
        """Test that singleton instances maintain identity."""
        # Create multiple "instances"
        undefined1 = UndefinedType()
        undefined2 = UndefinedType()
        unset1 = UnsetType()
        unset2 = UnsetType()

        # All should be the same object
        assert undefined1 is undefined2
        assert undefined1 is Undefined
        assert unset1 is unset2
        assert unset1 is Unset

    def test_singleton_deepcopy(self):
        """Test that deepcopy returns the same singleton."""
        undefined_copy = copy.deepcopy(Undefined)
        unset_copy = copy.deepcopy(Unset)

        assert undefined_copy is Undefined
        assert unset_copy is Unset

    def test_singleton_copy(self):
        """Test that copy returns the same singleton."""
        undefined_copy = copy.copy(Undefined)
        unset_copy = copy.copy(Unset)

        assert undefined_copy is Undefined
        assert unset_copy is Unset


class TestUndefinedType:
    """Test the UndefinedType sentinel."""

    def test_bool_evaluation(self):
        """Test that Undefined evaluates to False."""
        assert not Undefined
        assert not bool(Undefined)

    def test_string_representation(self):
        """Test string representations."""
        assert repr(Undefined) == "Undefined"
        assert str(Undefined) == "Undefined"

    def test_backward_compatibility(self):
        """Test that UNDEFINED is the same as Undefined."""
        assert UNDEFINED is Undefined

    def test_identity_checks(self):
        """Test identity comparisons."""
        value = Undefined
        assert value is Undefined
        assert value is not None
        assert value is not Unset

    def test_in_conditionals(self):
        """Test using Undefined in conditional expressions."""

        def func(param=Undefined):
            if param is Undefined:
                return "not provided"
            return param

        assert func() == "not provided"
        assert func(None) is None
        assert func(0) == 0
        assert func("") == ""


class TestUnsetType:
    """Test the UnsetType sentinel."""

    def test_bool_evaluation(self):
        """Test that Unset evaluates to False."""
        assert not Unset
        assert not bool(Unset)

    def test_string_representation(self):
        """Test string representations."""
        assert repr(Unset) == "Unset"
        assert str(Unset) == "Unset"

    def test_identity_checks(self):
        """Test identity comparisons."""
        value = Unset
        assert value is Unset
        assert value is not None
        assert value is not Undefined

    def test_in_conditionals(self):
        """Test using Unset in conditional expressions."""

        def func(param=Unset):
            if param is Unset:
                return "not set"
            return param

        assert func() == "not set"
        assert func(None) is None
        assert func(0) == 0
        assert func("") == ""


class TestSentinelUtilities:
    """Test the sentinel utility functions."""

    def test_is_undefined(self):
        """Test is_undefined function."""
        assert is_undefined(Undefined) is True
        assert is_undefined(UNDEFINED) is True
        assert is_undefined(Unset) is False
        assert is_undefined(None) is False
        assert is_undefined(0) is False
        assert is_undefined("") is False

    def test_is_unset(self):
        """Test is_unset function."""
        assert is_unset(Unset) is True
        assert is_unset(Undefined) is False
        assert is_unset(None) is False
        assert is_unset(0) is False
        assert is_unset("") is False

    def test_is_sentinel(self):
        """Test is_sentinel function."""
        assert is_sentinel(Undefined) is True
        assert is_sentinel(Unset) is True
        assert is_sentinel(None) is False
        assert is_sentinel(0) is False
        assert is_sentinel("") is False

    def test_not_sentinel(self):
        """Test not_sentinel function."""
        assert not_sentinel(Undefined) is False
        assert not_sentinel(Unset) is False
        assert not_sentinel(None) is True
        assert not_sentinel(0) is True
        assert not_sentinel("") is True

    def test_filtering_with_sentinels(self):
        """Test using sentinels in filtering operations."""
        values = [1, Undefined, 2, Unset, 3, None, 4]

        # Filter out sentinels
        filtered = list(filter(not_sentinel, values))
        assert filtered == [1, 2, 3, None, 4]

        # Filter out undefined only
        filtered_undefined = [v for v in values if not is_undefined(v)]
        assert filtered_undefined == [1, 2, Unset, 3, None, 4]

        # Filter out unset only
        filtered_unset = [v for v in values if not is_unset(v)]
        assert filtered_unset == [1, Undefined, 2, 3, None, 4]


class TestSentinelUsagePatterns:
    """Test common usage patterns with sentinels."""

    def test_dict_with_sentinels(self):
        """Test using sentinels in dictionaries."""
        d = {
            "present": 1,
            "none": None,
            "undefined": Undefined,
            "unset": Unset,
        }

        # Test get with default
        assert d.get("missing", Undefined) is Undefined
        assert d.get("missing", Unset) is Unset

        # Test filtering
        filtered = {k: v for k, v in d.items() if not_sentinel(v)}
        assert filtered == {"present": 1, "none": None}

    def test_function_defaults(self):
        """Test using sentinels as function defaults."""

        def process(
            required: str,
            optional: Any = Unset,
            retry_default: Any = Undefined,
        ):
            result = {"required": required}

            if optional is not Unset:
                result["optional"] = optional

            if retry_default is not Undefined:
                result["retry_default"] = retry_default

            return result

        # Test with only required
        assert process("test") == {"required": "test"}

        # Test with optional provided
        assert process("test", "opt") == {
            "required": "test",
            "optional": "opt",
        }

        # Test with retry_default provided
        assert process("test", retry_default="default") == {
            "required": "test",
            "retry_default": "default",
        }

        # Test with None values
        assert process("test", None, None) == {
            "required": "test",
            "optional": None,
            "retry_default": None,
        }

    def test_class_attributes(self):
        """Test using sentinels in class attributes."""

        class Config:
            required: str
            optional: Any = Unset
            default_value: Any = Undefined

            def __init__(self, required: str, **kwargs):
                self.required = required
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

            def get_config(self):
                config = {"required": self.required}

                if self.optional is not Unset:
                    config["optional"] = self.optional

                if self.default_value is not Undefined:
                    config["default_value"] = self.default_value

                return config

        # Test basic config
        cfg = Config("test")
        assert cfg.get_config() == {"required": "test"}

        # Test with optional set
        cfg = Config("test", optional="value")
        assert cfg.get_config() == {"required": "test", "optional": "value"}

    def test_list_operations(self):
        """Test sentinels in list operations."""

        def process_list(items: list, default=Undefined):
            result = []
            for item in items:
                if item is Undefined and default is not Undefined:
                    result.append(default)
                elif not is_sentinel(item):
                    result.append(item)
            return result

        # Test without default
        items = [1, Undefined, 2, Unset, 3]
        assert process_list(items) == [1, 2, 3]

        # Test with default
        assert process_list(items, default=0) == [1, 0, 2, 3]

    def test_error_handling(self):
        """Test using sentinels in error handling."""

        def safe_divide(a: float, b: float, on_error=Undefined):
            try:
                return a / b
            except ZeroDivisionError:
                if on_error is not Undefined:
                    return on_error
                raise

        # Test normal operation
        assert safe_divide(10, 2) == 5

        # Test error with default
        assert safe_divide(10, 0, on_error=None) is None
        assert safe_divide(10, 0, on_error=float("inf")) == float("inf")

        # Test error without default
        with pytest.raises(ZeroDivisionError):
            safe_divide(10, 0)


class TestSentinelComparison:
    """Test that sentinels are properly distinct."""

    def test_sentinel_uniqueness(self):
        """Test that each sentinel is unique."""
        # Each sentinel should be distinct
        assert Undefined is not Unset
        assert Undefined is not None
        assert Unset is not None

        # But identical to themselves
        assert Undefined is Undefined
        assert Unset is Unset

    def test_sentinel_in_collections(self):
        """Test sentinels in sets and as dict keys."""
        # Sentinels should be hashable
        s = {Undefined, Unset, None}
        assert len(s) == 3

        # Should work as dict keys
        d = {Undefined: "undefined", Unset: "unset", None: "none"}
        assert d[Undefined] == "undefined"
        assert d[Unset] == "unset"
        assert d[None] == "none"

    def test_type_checking(self):
        """Test type checking with sentinels."""
        assert isinstance(Undefined, UndefinedType)
        assert isinstance(Undefined, SingletonType)
        assert not isinstance(Undefined, UnsetType)

        assert isinstance(Unset, UnsetType)
        assert isinstance(Unset, SingletonType)
        assert not isinstance(Unset, UndefinedType)
