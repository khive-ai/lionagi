# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.libs.schema.typescript — typescript_schema and helpers."""

import pytest

from lionagi.libs.schema.typescript import (
    _extract_type_signature,
    _format_enum_union,
    _type_map,
    typescript_schema,
)

# ---------------------------------------------------------------------------
# _type_map
# ---------------------------------------------------------------------------


class TestTypeMap:
    def test_known_types(self):
        assert _type_map("string") == "string"
        assert _type_map("integer") == "int"
        assert _type_map("number") == "float"
        assert _type_map("boolean") == "bool"
        assert _type_map("array") == "array"
        assert _type_map("object") == "object"
        assert _type_map("null") == "null"

    def test_unknown_type_passthrough(self):
        assert _type_map("foobar") == "foobar"


# ---------------------------------------------------------------------------
# _format_enum_union
# ---------------------------------------------------------------------------


class TestFormatEnumUnion:
    def test_string_values(self):
        result = _format_enum_union(["active", "inactive"])
        assert result == '"active" | "inactive"'

    def test_null_value(self):
        result = _format_enum_union(["a", None])
        assert result == '"a" | null'

    def test_integer_values(self):
        result = _format_enum_union([1, 2, 3])
        assert result == "1 | 2 | 3"

    def test_mixed_str_int_none(self):
        result = _format_enum_union(["x", 42, None])
        assert result == '"x" | 42 | null'

    def test_single_value(self):
        result = _format_enum_union(["only"])
        assert result == '"only"'


# ---------------------------------------------------------------------------
# typescript_schema — basic cases
# ---------------------------------------------------------------------------


class TestTypescriptSchemaEmpty:
    def test_empty_properties(self):
        assert typescript_schema({"properties": {}}) == ""

    def test_no_properties_key(self):
        assert typescript_schema({}) == ""


# ---------------------------------------------------------------------------
# Required vs optional fields
# ---------------------------------------------------------------------------


class TestRequiredOptional:
    def test_required_string_field(self):
        schema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = typescript_schema(schema)
        assert result == "name: string"

    def test_optional_integer_field(self):
        schema = {
            "properties": {"age": {"type": "integer"}},
        }
        result = typescript_schema(schema)
        assert result == "age?: int"

    def test_required_boolean_field(self):
        schema = {
            "properties": {"active": {"type": "boolean"}},
            "required": ["active"],
        }
        result = typescript_schema(schema)
        assert result == "active: bool"


# ---------------------------------------------------------------------------
# Enum fields
# ---------------------------------------------------------------------------


class TestEnumFields:
    def test_enum_values_optional(self):
        schema = {"properties": {"status": {"enum": ["active", "inactive"]}}}
        result = typescript_schema(schema)
        assert "status" in result
        assert '"active"' in result
        assert '"inactive"' in result
        assert "?" in result  # not required → optional

    def test_enum_with_null(self):
        schema = {"properties": {"x": {"enum": ["a", None]}}}
        result = typescript_schema(schema)
        assert "null" in result
        assert '"a"' in result


# ---------------------------------------------------------------------------
# anyOf fields
# ---------------------------------------------------------------------------


class TestAnyOfFields:
    def test_anyof_string_int(self):
        schema = {
            "properties": {"val": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["val"],
        }
        result = typescript_schema(schema)
        assert "string" in result
        assert "int" in result

    def test_anyof_including_null_makes_optional(self):
        schema = {
            "properties": {"val": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": ["val"],
        }
        result = typescript_schema(schema)
        assert "?" in result
        assert "null" in result

    def test_anyof_with_ref(self):
        schema = {"properties": {"parent": {"anyOf": [{"$ref": "#/$defs/ParentType"}]}}}
        result = typescript_schema(schema)
        assert "ParentType" in result

    def test_anyof_with_array_items_type(self):
        schema = {
            "properties": {
                "vals": {"anyOf": [{"type": "array", "items": {"type": "string"}}]}
            }
        }
        result = typescript_schema(schema)
        assert "string[]" in result

    def test_anyof_with_array_items_ref(self):
        schema = {
            "properties": {
                "nodes": {
                    "anyOf": [{"type": "array", "items": {"$ref": "#/$defs/NodeType"}}]
                }
            }
        }
        result = typescript_schema(schema)
        assert "NodeType[]" in result

    def test_anyof_with_enum_option(self):
        schema = {"properties": {"mode": {"anyOf": [{"enum": ["fast", "slow"]}]}}}
        result = typescript_schema(schema)
        assert '"fast"' in result
        assert '"slow"' in result


# ---------------------------------------------------------------------------
# Array fields
# ---------------------------------------------------------------------------


class TestArrayFields:
    def test_array_with_items_type(self):
        schema = {
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        result = typescript_schema(schema)
        assert "items: string[]" in result

    def test_array_with_items_ref(self):
        schema = {
            "properties": {
                "children": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/ChildType"},
                }
            }
        }
        result = typescript_schema(schema)
        assert "ChildType[]" in result

    def test_array_with_no_items_is_any(self):
        schema = {
            "properties": {"stuff": {"type": "array"}},
        }
        result = typescript_schema(schema)
        assert "any[]" in result

    def test_array_with_enum_items(self):
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"enum": ["a", "b"]},
                }
            }
        }
        result = typescript_schema(schema)
        assert '("a" | "b")[]' in result


# ---------------------------------------------------------------------------
# $ref fields
# ---------------------------------------------------------------------------


class TestRefField:
    def test_ref_field_optional(self):
        schema = {"properties": {"parent": {"$ref": "#/$defs/ParentType"}}}
        result = typescript_schema(schema)
        assert "ParentType" in result
        assert "?" in result  # optional because not required

    def test_ref_field_required(self):
        schema = {
            "properties": {"parent": {"$ref": "#/$defs/ParentType"}},
            "required": ["parent"],
        }
        result = typescript_schema(schema)
        assert "parent: ParentType" in result
        assert "?" not in result


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaultValues:
    def test_default_string(self):
        schema = {"properties": {"name": {"type": "string", "default": "world"}}}
        result = typescript_schema(schema)
        assert '= "world"' in result

    def test_default_none(self):
        schema = {"properties": {"x": {"type": "string", "default": None}}}
        result = typescript_schema(schema)
        assert "= null" in result

    def test_default_true(self):
        schema = {"properties": {"flag": {"type": "boolean", "default": True}}}
        result = typescript_schema(schema)
        assert "= true" in result

    def test_default_false(self):
        schema = {"properties": {"flag": {"type": "boolean", "default": False}}}
        result = typescript_schema(schema)
        assert "= false" in result

    def test_default_int(self):
        schema = {"properties": {"count": {"type": "integer", "default": 42}}}
        result = typescript_schema(schema)
        assert "= 42" in result


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


class TestDescription:
    def test_description_appended(self):
        schema = {
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The user email address",
                }
            },
            "required": ["email"],
        }
        result = typescript_schema(schema)
        assert "The user email address" in result
        assert result.startswith("email: string")


# ---------------------------------------------------------------------------
# type ending with "?"
# ---------------------------------------------------------------------------


class TestQuestionMarkType:
    def test_type_ending_question_mark(self):
        """A type string ending in '?' should be stripped and mark field optional."""
        spec = {"type": "string?"}
        type_sig, is_optional = _extract_type_signature(spec, required=True)
        assert type_sig == "string"
        assert is_optional is True


# ---------------------------------------------------------------------------
# indent > 0
# ---------------------------------------------------------------------------


class TestIndent:
    def test_indent_adds_extra_spaces(self):
        schema = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        result = typescript_schema(schema, indent=2)
        # indent=2 means 4 extra spaces (2*2) prepended before the standard 2
        assert result.startswith("    " + "  x: int") or result.startswith(
            "  " * 2 + "x: int"
        )
        # Verify it has at least 4 spaces of leading whitespace
        assert result.lstrip() == "x: int"
        assert result != "  x: int"
