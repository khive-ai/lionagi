"""Tests for JSON utilities migrated to msgspec."""

import pytest

from lionagi.ln._extract_json import extract_json
from lionagi.ln._fuzzy_json import fix_json_string, fuzzy_json


class TestFuzzyJson:
    """Test fuzzy JSON parsing with msgspec."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        result = fuzzy_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_single_quotes(self):
        """Test parsing JSON with single quotes."""
        result = fuzzy_json("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_unquoted_keys(self):
        """Test parsing JSON with unquoted keys."""
        result = fuzzy_json("{key: 'value'}")
        assert result == {"key": "value"}

    def test_missing_closing_bracket(self):
        """Test fixing missing closing bracket."""
        result = fuzzy_json('{"key": "value"')
        assert result == {"key": "value"}

    def test_nested_structure(self):
        """Test parsing nested JSON structure."""
        result = fuzzy_json('{"outer": {"inner": "value"}}')
        assert result == {"outer": {"inner": "value"}}

    def test_array(self):
        """Test parsing JSON array."""
        result = fuzzy_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_invalid_json_raises(self):
        """Test that truly invalid JSON raises ValueError."""
        with pytest.raises(ValueError):
            fuzzy_json("not json at all")

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            fuzzy_json("")


class TestExtractJson:
    """Test JSON extraction from markdown and strings."""

    def test_direct_json(self):
        """Test extracting direct JSON."""
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_json_block(self):
        """Test extracting JSON from markdown code block."""
        text = """
        Some text here
        ```json
        {"key": "value"}
        ```
        More text
        """
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_multiple_json_blocks(self):
        """Test extracting multiple JSON blocks."""
        text = """
        ```json
        {"first": 1}
        ```
        Some text
        ```json
        {"second": 2}
        ```
        """
        result = extract_json(text, return_one_if_single=False)
        assert result == [{"first": 1}, {"second": 2}]

    def test_fuzzy_parse_in_markdown(self):
        """Test fuzzy parsing within markdown blocks."""
        text = """
        ```json
        {'key': 'value'}
        ```
        """
        result = extract_json(text, fuzzy_parse=True)
        assert result == {"key": "value"}

    def test_list_input(self):
        """Test extracting JSON from list of strings."""
        lines = ["Some text", "```json", '{"key": "value"}', "```"]
        result = extract_json(lines)
        assert result == {"key": "value"}

    def test_no_json_returns_empty(self):
        """Test that text without JSON returns empty list."""
        result = extract_json("No JSON here")
        assert result == []


class TestFixJsonString:
    """Test JSON string fixing utilities."""

    def test_fix_missing_closing_brace(self):
        """Test fixing missing closing brace."""
        result = fix_json_string('{"key": "value"')
        assert result == '{"key": "value"}'

    def test_fix_missing_closing_bracket(self):
        """Test fixing missing closing bracket."""
        result = fix_json_string("[1, 2, 3")
        assert result == "[1, 2, 3]"

    def test_fix_nested_missing_brackets(self):
        """Test fixing nested missing brackets."""
        result = fix_json_string('{"outer": {"inner": "value"')
        assert result == '{"outer": {"inner": "value"}}'

    def test_extra_closing_bracket_raises(self):
        """Test that extra closing bracket raises error."""
        with pytest.raises(ValueError, match="Extra closing bracket"):
            fix_json_string('{"key": "value"}}')

    def test_mismatched_brackets_raises(self):
        """Test that mismatched brackets raise error."""
        with pytest.raises(ValueError, match="Mismatched brackets"):
            fix_json_string('{"key": "value"]')
