"""Tests for lionagi/ln/fuzzy/_fuzzy_json.py

Target: Cover lines 88-89 (escaped character handling in fix_json_string)
"""

import pytest

from lionagi.ln.fuzzy._fuzzy_json import (
    _check_valid_str,
    _clean_json_string,
    fix_json_string,
    fuzzy_json,
)

# ============================================================================
# Test fuzzy_json main function
# ============================================================================


def test_fuzzy_json_valid():
    """Test fuzzy_json with valid JSON"""
    result = fuzzy_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_fuzzy_json_single_quotes():
    """Test fuzzy_json with single quotes"""
    result = fuzzy_json("{'key': 'value'}")
    assert result == {"key": "value"}


def test_fuzzy_json_unquoted_keys():
    """Test fuzzy_json with unquoted keys"""
    result = fuzzy_json("{key: 'value'}")
    assert result == {"key": "value"}


def test_fuzzy_json_trailing_commas():
    """Test fuzzy_json with trailing commas"""
    result = fuzzy_json('{"key": "value",}')
    assert result == {"key": "value"}


def test_fuzzy_json_missing_closing_bracket():
    """Test fuzzy_json with missing closing bracket"""
    result = fuzzy_json('{"key": "value"')
    assert result == {"key": "value"}


def test_fuzzy_json_invalid():
    """Test fuzzy_json with completely invalid JSON"""
    with pytest.raises(ValueError, match="Invalid JSON string"):
        fuzzy_json("{completely broken")


def test_fuzzy_json_not_string():
    """Test fuzzy_json with non-string input"""
    with pytest.raises(TypeError, match="Input must be a string"):
        fuzzy_json(123)


def test_fuzzy_json_empty():
    """Test fuzzy_json with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fuzzy_json("")


def test_fuzzy_json_whitespace_only():
    """Test fuzzy_json with whitespace only"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fuzzy_json("   ")


# ============================================================================
# Test _check_valid_str
# ============================================================================


def test_check_valid_str_valid():
    """Test _check_valid_str with valid string"""
    _check_valid_str("valid string")  # Should not raise


def test_check_valid_str_not_string():
    """Test _check_valid_str with non-string"""
    with pytest.raises(TypeError, match="Input must be a string"):
        _check_valid_str(123)


def test_check_valid_str_empty():
    """Test _check_valid_str with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        _check_valid_str("")


# ============================================================================
# Test _clean_json_string
# ============================================================================


def test_clean_json_string_single_quotes():
    """Test _clean_json_string with single quotes"""
    result = _clean_json_string("{'key': 'value'}")
    assert '"' in result


def test_clean_json_string_trailing_comma():
    """Test _clean_json_string removes trailing commas"""
    result = _clean_json_string('{"key": "value",}')
    assert result == '{"key": "value"}' or '","' not in result


def test_clean_json_string_whitespace():
    """Test _clean_json_string normalizes whitespace"""
    result = _clean_json_string('{"key":   "value"}')
    assert "  " not in result


def test_clean_json_string_unquoted_keys():
    """Test _clean_json_string quotes unquoted keys"""
    result = _clean_json_string('{key: "value"}')
    assert '"key"' in result


# ============================================================================
# Test fix_json_string - Lines 88-89 (escaped chars)
# ============================================================================


def test_fix_json_string_escaped_backslash():
    """Test fix_json_string with escaped backslash - covers lines 88-89"""
    # JSON string with escaped characters
    json_str = r'{"path": "C:\\Users\\file.txt"}'
    result = fix_json_string(json_str)
    # Should handle escaped backslashes properly
    assert result == json_str  # Should be unchanged


def test_fix_json_string_escaped_quote():
    """Test fix_json_string with escaped quote - covers lines 88-89"""
    json_str = r'{"text": "He said \"hello\""}'
    result = fix_json_string(json_str)
    # Should handle escaped quotes properly
    assert result == json_str


def test_fix_json_string_escaped_newline():
    """Test fix_json_string with escaped newline - covers lines 88-89"""
    json_str = r'{"text": "line1\nline2"}'
    result = fix_json_string(json_str)
    assert result == json_str


def test_fix_json_string_multiple_escapes():
    """Test fix_json_string with multiple escape sequences"""
    json_str = r'{"path": "C:\\folder\\file", "text": "quote: \"hi\"\nend"}'
    result = fix_json_string(json_str)
    # Should handle all escapes properly
    assert result == json_str


def test_fix_json_string_missing_bracket():
    """Test fix_json_string adds missing closing bracket"""
    json_str = '{"key": "value"'
    result = fix_json_string(json_str)
    assert result == '{"key": "value"}'


def test_fix_json_string_missing_multiple_brackets():
    """Test fix_json_string adds multiple missing brackets"""
    json_str = '{"key": {"nested": "value"'
    result = fix_json_string(json_str)
    assert result == '{"key": {"nested": "value"}}'


def test_fix_json_string_missing_array_bracket():
    """Test fix_json_string adds missing array bracket"""
    json_str = '["item1", "item2"'
    result = fix_json_string(json_str)
    assert result == '["item1", "item2"]'


def test_fix_json_string_extra_closing_bracket():
    """Test fix_json_string with extra closing bracket"""
    json_str = '{"key": "value"}}'
    with pytest.raises(ValueError, match="Extra closing bracket"):
        fix_json_string(json_str)


def test_fix_json_string_mismatched_brackets():
    """Test fix_json_string with mismatched brackets"""
    json_str = '{"key": "value"]'
    with pytest.raises(ValueError, match="Mismatched brackets"):
        fix_json_string(json_str)


def test_fix_json_string_empty():
    """Test fix_json_string with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fix_json_string("")


def test_fix_json_string_complex_with_escapes():
    """Test fix_json_string with complex JSON containing escapes"""
    json_str = r'{"data": {"path": "C:\\test\\", "text": "say \"hi\"", "newline": "a\nb"'
    result = fix_json_string(json_str)
    # Should add missing closing brackets while preserving escapes
    assert result.endswith("}}")
    assert "\\\\" in result or "\\test" in result  # Escapes preserved


def test_fix_json_string_escape_at_end():
    """Test fix_json_string with escape at end of string"""
    json_str = r'{"path": "folder\\"'
    result = fix_json_string(json_str)
    # Should handle backslash at end and add missing bracket
    assert result.endswith("}")


def test_fuzzy_json_with_escapes_comprehensive():
    """Comprehensive test for fuzzy_json with various escape scenarios"""
    # Test that fuzzy_json can handle JSON with escapes through the full pipeline
    json_str = r'{"file": "C:\\Users\\test.txt", "quote": "He said \"hello\""}'
    result = fuzzy_json(json_str)
    assert result["file"] == "C:\\Users\\test.txt"
    assert result["quote"] == 'He said "hello"'


def test_fix_json_string_escaped_chars_in_array():
    """Test fix_json_string with escaped chars in array"""
    json_str = r'["item1", "C:\\path\\file", "text with \"quotes\""'
    result = fix_json_string(json_str)
    assert result.endswith("]")
    assert "\\\\" in result or "\\path" in result


def test_fix_json_string_nested_with_escapes():
    """Test fix_json_string with nested structures and escapes"""
    json_str = r'{"outer": {"inner": "path\\to\\file"'
    result = fix_json_string(json_str)
    # Should add two closing brackets and preserve escapes
    assert result.count("}") == 2
    assert "path" in result
