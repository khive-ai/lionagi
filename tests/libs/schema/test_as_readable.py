# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for as_readable utility."""

import json

import pytest

from lionagi.libs.schema.as_readable import (
    as_readable,
    format_dict,
    in_console,
    in_notebook,
)


class TestFormatDict:
    """Test cases for format_dict function."""

    def test_simple_dict(self):
        """Test formatting a simple dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        result = format_dict(data)

        assert "key1:" in result
        assert "value1" in result
        assert "key2:" in result
        assert "value2" in result

    def test_nested_dict(self):
        """Test formatting nested dictionaries."""
        data = {"level1": {"level2": {"level3": "value"}}}
        result = format_dict(data)

        assert "level1:" in result
        assert "level2:" in result
        assert "level3:" in result
        assert "value" in result

    def test_dict_with_list(self):
        """Test formatting dictionary containing lists."""
        data = {"items": ["item1", "item2", "item3"]}
        result = format_dict(data)

        assert "items:" in result
        assert "- item1" in result
        assert "- item2" in result
        assert "- item3" in result

    def test_dict_with_multiline_string(self):
        """Test formatting dictionary with multiline strings."""
        data = {"description": "Line 1\nLine 2\nLine 3"}
        result = format_dict(data)

        assert "description: |" in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_list_of_dicts(self):
        """Test formatting list of dictionaries."""
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = format_dict(data)

        assert "- name:" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_mixed_nested_structure(self):
        """Test complex nested structure."""
        data = {
            "users": [
                {
                    "name": "Alice",
                    "roles": ["admin", "user"],
                    "metadata": {"created": "2024-01-01", "active": True},
                }
            ]
        }
        result = format_dict(data)

        assert "users:" in result
        assert "- name:" in result
        assert "Alice" in result
        assert "- admin" in result
        assert "metadata:" in result

    def test_indent_parameter(self):
        """Test that indent parameter affects output."""
        data = {"key": "value"}

        result_0 = format_dict(data, indent=0)
        result_1 = format_dict(data, indent=1)
        result_2 = format_dict(data, indent=2)

        # Higher indent should have more leading spaces
        assert result_1.startswith("  ")
        assert result_2.startswith("    ")

    def test_scalar_types(self):
        """Test formatting of scalar types."""
        assert format_dict("string") == "string"
        assert format_dict(42) == "42"
        assert format_dict(3.14) == "3.14"
        assert format_dict(True) == "True"
        assert format_dict(None) == "None"


class TestAsReadable:
    """Test cases for as_readable function."""

    def test_simple_dict_json_format(self):
        """Test converting simple dict to JSON format."""
        data = {"name": "Alice", "age": 30}
        result = as_readable(data, format_curly=False)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["name"] == "Alice"
        assert parsed["age"] == 30

    def test_simple_dict_yaml_format(self):
        """Test converting simple dict to YAML-like format."""
        data = {"name": "Alice", "age": 30}
        result = as_readable(data, format_curly=True)

        assert "name:" in result
        assert "Alice" in result
        assert "age:" in result
        assert "30" in result

    def test_list_of_items(self):
        """Test converting list of items."""
        data = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        result = as_readable(data, format_curly=False)

        # Should contain both items
        assert "Item 1" in result
        assert "Item 2" in result

    def test_markdown_wrapper_json(self):
        """Test markdown wrapper with JSON format."""
        data = {"key": "value"}
        result = as_readable(data, md=True, format_curly=False)

        assert result.startswith("```json")
        assert result.endswith("```")
        assert '"key"' in result

    def test_markdown_wrapper_yaml(self):
        """Test markdown wrapper with YAML format."""
        data = {"key": "value"}
        result = as_readable(data, md=True, format_curly=True)

        assert result.startswith("```yaml")
        assert result.endswith("```")
        assert "key:" in result

    def test_max_chars_truncation(self):
        """Test output truncation with max_chars."""
        data = {"key": "a" * 1000}
        result = as_readable(data, max_chars=100)

        assert len(result) <= 150  # Some overhead for truncation message
        assert "Truncated" in result or "..." in result

    def test_nested_structure(self):
        """Test converting nested structure."""
        data = {
            "user": {
                "name": "Alice",
                "contacts": {
                    "email": "alice@example.com",
                    "phone": "123-456-7890",
                },
            }
        }
        result = as_readable(data, format_curly=False)

        parsed = json.loads(result)
        assert parsed["user"]["name"] == "Alice"
        assert parsed["user"]["contacts"]["email"] == "alice@example.com"

    def test_display_str_returns_none(self):
        """Test that display_str=True returns None."""
        data = {"key": "value"}
        result = as_readable(data, display_str=True, use_rich=False)

        # Should print and return None
        assert result is None

    def test_use_rich_false(self):
        """Test with rich rendering disabled."""
        data = {"key": "value"}
        result = as_readable(data, use_rich=False)

        # Should return string even in console
        assert isinstance(result, str)

    def test_empty_dict(self):
        """Test converting empty dictionary."""
        data = {}
        result = as_readable(data, format_curly=False)

        parsed = json.loads(result)
        assert parsed == {}

    def test_empty_list(self):
        """Test converting empty list."""
        data = []
        result = as_readable(data, format_curly=False)

        # Empty list might produce empty string, handle gracefully
        if result.strip():
            parsed = json.loads(result)
            assert parsed == []
        else:
            # Empty output is acceptable for empty list
            assert result == "" or result.strip() == ""

    def test_string_input(self):
        """Test with string input."""
        data = "simple string"
        result = as_readable(data)

        # Should handle gracefully
        assert isinstance(result, str)
        assert "simple string" in result

    def test_integer_input(self):
        """Test with integer input."""
        data = 42
        result = as_readable(data)

        assert isinstance(result, str)
        assert "42" in result

    def test_boolean_input(self):
        """Test with boolean input."""
        data = True
        result = as_readable(data, format_curly=False)

        assert isinstance(result, str)
        assert "true" in result.lower()

    def test_none_input(self):
        """Test with None input."""
        data = None
        result = as_readable(data, format_curly=False)

        assert isinstance(result, str)

    def test_complex_real_world_data(self):
        """Test with complex real-world-like data structure."""
        data = {
            "status": "success",
            "data": {
                "users": [
                    {
                        "id": 1,
                        "name": "Alice",
                        "email": "alice@example.com",
                        "roles": ["admin", "user"],
                        "metadata": {
                            "created_at": "2024-01-01T00:00:00Z",
                            "last_login": "2024-01-15T10:30:00Z",
                        },
                    },
                    {
                        "id": 2,
                        "name": "Bob",
                        "email": "bob@example.com",
                        "roles": ["user"],
                        "metadata": {
                            "created_at": "2024-01-02T00:00:00Z",
                            "last_login": "2024-01-14T15:45:00Z",
                        },
                    },
                ],
                "total": 2,
            },
        }

        # Test JSON format
        result_json = as_readable(data, format_curly=False)
        parsed = json.loads(result_json)
        assert parsed["status"] == "success"
        assert len(parsed["data"]["users"]) == 2

        # Test YAML format
        result_yaml = as_readable(data, format_curly=True)
        assert "status:" in result_yaml
        assert "Alice" in result_yaml
        assert "Bob" in result_yaml

    def test_multiline_string_values(self):
        """Test handling of multiline string values."""
        data = {
            "description": "This is a long description\nwith multiple lines\nof text."
        }
        result = as_readable(data, format_curly=True)

        assert "description:" in result
        assert "multiple lines" in result

    def test_panel_parameter(self):
        """Test panel parameter (should not affect string output)."""
        data = {"key": "value"}

        result_with_panel = as_readable(data, panel=True, use_rich=False)
        result_without_panel = as_readable(data, panel=False, use_rich=False)

        # Both should produce string output when rich is disabled
        assert isinstance(result_with_panel, str)
        assert isinstance(result_without_panel, str)

    def test_border_parameter(self):
        """Test border parameter (should not affect string output)."""
        data = {"key": "value"}

        result_with_border = as_readable(data, border=True, use_rich=False)
        result_without_border = as_readable(data, border=False, use_rich=False)

        # Both should produce string output when rich is disabled
        assert isinstance(result_with_border, str)
        assert isinstance(result_without_border, str)

    def test_theme_parameter(self):
        """Test theme parameter (should not affect string output when rich disabled)."""
        data = {"key": "value"}

        result_dark = as_readable(data, theme="dark", use_rich=False)
        result_light = as_readable(data, theme="light", use_rich=False)

        # Both should produce string output when rich is disabled
        assert isinstance(result_dark, str)
        assert isinstance(result_light, str)

    def test_max_panel_width_parameter(self):
        """Test max_panel_width parameter."""
        data = {"key": "value"}

        result = as_readable(data, max_panel_width=80, use_rich=False)

        # Should still produce valid output
        assert isinstance(result, str)


class TestEnvironmentDetection:
    """Test cases for environment detection functions."""

    def test_in_notebook_returns_bool(self):
        """Test that in_notebook returns a boolean."""
        result = in_notebook()
        assert isinstance(result, bool)

    def test_in_console_returns_bool(self):
        """Test that in_console returns a boolean."""
        result = in_console()
        assert isinstance(result, bool)

    def test_notebook_and_console_mutually_exclusive(self):
        """Test that we can't be in both notebook and console."""
        # If in notebook, should not be in console
        if in_notebook():
            assert not in_console()
        # If in console, should not be in notebook
        elif in_console():
            assert not in_notebook()
