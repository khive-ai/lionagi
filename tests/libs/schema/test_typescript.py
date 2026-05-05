# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for lionagi.libs.schema.typescript."""

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
    def test_string(self):
        assert _type_map("string") == "string"

    def test_integer(self):
        assert _type_map("integer") == "int"

    def test_number(self):
        assert _type_map("number") == "float"

    def test_boolean(self):
        assert _type_map("boolean") == "bool"

    def test_array(self):
        assert _type_map("array") == "array"

    def test_object(self):
        assert _type_map("object") == "object"

    def test_null(self):
        assert _type_map("null") == "null"

    def test_unknown_passthrough(self):
        assert _type_map("foobar") == "foobar"

    def test_empty_string_passthrough(self):
        assert _type_map("") == ""

    def test_custom_type_passthrough(self):
        assert _type_map("DateTime") == "DateTime"


# ---------------------------------------------------------------------------
# _format_enum_union
# ---------------------------------------------------------------------------


class TestFormatEnumUnion:
    def test_string_values_quoted(self):
        result = _format_enum_union(["active", "inactive"])
        assert result == '"active" | "inactive"'

    def test_none_becomes_null(self):
        result = _format_enum_union([None])
        assert result == "null"

    def test_string_and_none(self):
        result = _format_enum_union(["a", None])
        assert result == '"a" | null'

    def test_integer_values(self):
        result = _format_enum_union([1, 2, 3])
        assert result == "1 | 2 | 3"

    def test_mixed_string_int_none(self):
        result = _format_enum_union(["x", 42, None])
        assert result == '"x" | 42 | null'

    def test_single_string(self):
        result = _format_enum_union(["only"])
        assert result == '"only"'

    def test_empty_list(self):
        result = _format_enum_union([])
        assert result == ""

    def test_float_value(self):
        result = _format_enum_union([3.14])
        assert result == "3.14"


# ---------------------------------------------------------------------------
# _extract_type_signature
# ---------------------------------------------------------------------------


class TestExtractTypeSignature:
    # --- enum ---
    def test_enum_required_no_null(self):
        spec = {"enum": ["a", "b"]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert '"a"' in sig
        assert '"b"' in sig
        assert opt is False  # required, no null

    def test_enum_optional_because_not_required(self):
        spec = {"enum": ["a", "b"]}
        sig, opt = _extract_type_signature(spec, required=False)
        assert opt is True

    def test_enum_with_null_makes_optional(self):
        spec = {"enum": ["a", None]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert "null" in sig
        assert opt is True

    # --- anyOf ---
    def test_anyof_basic_types(self):
        spec = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert "string" in sig
        assert "int" in sig
        assert opt is False

    def test_anyof_with_null_is_optional(self):
        spec = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert "null" in sig
        assert opt is True

    def test_anyof_with_ref(self):
        spec = {"anyOf": [{"$ref": "#/$defs/MyModel"}]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert "MyModel" in sig

    def test_anyof_with_array_items_type(self):
        spec = {"anyOf": [{"type": "array", "items": {"type": "string"}}]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert "string[]" in sig

    def test_anyof_with_array_items_ref(self):
        spec = {"anyOf": [{"type": "array", "items": {"$ref": "#/$defs/Node"}}]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert "Node[]" in sig

    def test_anyof_with_enum_option(self):
        spec = {"anyOf": [{"enum": ["x", "y"]}]}
        sig, opt = _extract_type_signature(spec, required=True)
        assert '"x"' in sig

    def test_anyof_empty_becomes_any(self):
        spec = {"anyOf": [{"type": "null"}]}
        sig, opt = _extract_type_signature(spec, required=True)
        # only null → type_parts is empty before null is appended
        assert "null" in sig

    # --- plain type ---
    def test_plain_type_required(self):
        spec = {"type": "string"}
        sig, opt = _extract_type_signature(spec, required=True)
        assert sig == "string"
        assert opt is False

    def test_plain_type_optional(self):
        spec = {"type": "integer"}
        sig, opt = _extract_type_signature(spec, required=False)
        assert sig == "int"
        assert opt is True

    def test_plain_type_ending_question_mark(self):
        spec = {"type": "string?"}
        sig, opt = _extract_type_signature(spec, required=True)
        assert sig == "string"
        assert opt is True

    # --- array type ---
    def test_array_with_typed_items(self):
        spec = {"type": "array", "items": {"type": "number"}}
        sig, opt = _extract_type_signature(spec, required=True)
        assert sig == "float[]"
        assert opt is False

    def test_array_with_ref_items(self):
        spec = {"type": "array", "items": {"$ref": "#/$defs/Item"}}
        sig, opt = _extract_type_signature(spec, required=True)
        assert sig == "Item[]"

    def test_array_with_enum_items(self):
        spec = {"type": "array", "items": {"enum": ["x", "y"]}}
        sig, opt = _extract_type_signature(spec, required=True)
        assert '("x" | "y")[]' == sig

    def test_array_no_items_defaults_to_any(self):
        spec = {"type": "array"}
        sig, opt = _extract_type_signature(spec, required=True)
        assert sig == "any[]"

    # --- $ref ---
    def test_ref_required(self):
        spec = {"$ref": "#/$defs/Address"}
        sig, opt = _extract_type_signature(spec, required=True)
        assert sig == "Address"
        assert opt is False

    def test_ref_optional(self):
        spec = {"$ref": "#/$defs/Address"}
        sig, opt = _extract_type_signature(spec, required=False)
        assert sig == "Address"
        assert opt is True

    # --- fallback ---
    def test_empty_spec_returns_any(self):
        sig, opt = _extract_type_signature({}, required=True)
        assert sig == "any"
        assert opt is False


# ---------------------------------------------------------------------------
# typescript_schema — integration tests
# ---------------------------------------------------------------------------


class TestTypescriptSchemaEmpty:
    def test_no_properties_key_returns_empty(self):
        assert typescript_schema({}) == ""

    def test_empty_properties_returns_empty(self):
        assert typescript_schema({"properties": {}}) == ""

    def test_non_object_schema_without_properties(self):
        assert typescript_schema({"type": "string"}) == ""


class TestTypescriptSchemaIndent:
    def test_indent_zero_no_leading_spaces(self):
        schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        result = typescript_schema(schema, indent=0)
        assert result == "x: string"

    def test_indent_one_two_spaces(self):
        schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        result = typescript_schema(schema, indent=1)
        assert result == "  x: string"

    def test_indent_two_four_spaces(self):
        schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        result = typescript_schema(schema, indent=2)
        assert result == "    x: string"


class TestTypescriptSchemaRequiredOptional:
    def test_required_field_no_question_mark(self):
        schema = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = typescript_schema(schema)
        assert result == "name: string"
        assert "?" not in result

    def test_optional_field_has_question_mark(self):
        schema = {"properties": {"age": {"type": "integer"}}}
        result = typescript_schema(schema)
        assert result == "age?: int"

    def test_multiple_fields_mixed(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = typescript_schema(schema)
        assert "name: string" in result
        assert "age?: int" in result


class TestTypescriptSchemaEnumField:
    def test_enum_required(self):
        schema = {
            "properties": {"status": {"enum": ["active", "inactive"]}},
            "required": ["status"],
        }
        result = typescript_schema(schema)
        assert '"active"' in result
        assert '"inactive"' in result
        assert "?" not in result

    def test_enum_optional_not_required(self):
        schema = {"properties": {"mode": {"enum": ["fast", "slow"]}}}
        result = typescript_schema(schema)
        assert "?" in result

    def test_enum_with_null_always_optional(self):
        schema = {
            "properties": {"x": {"enum": ["a", None]}},
            "required": ["x"],
        }
        result = typescript_schema(schema)
        assert "null" in result
        assert "?" in result


class TestTypescriptSchemaAnyOf:
    def test_anyof_union_types(self):
        schema = {
            "properties": {"val": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["val"],
        }
        result = typescript_schema(schema)
        assert "string" in result
        assert "int" in result

    def test_anyof_with_null_optional(self):
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


class TestTypescriptSchemaArrayField:
    def test_array_typed_items(self):
        schema = {
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        }
        result = typescript_schema(schema)
        assert "string[]" in result

    def test_array_ref_items(self):
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

    def test_array_no_items_any(self):
        schema = {"properties": {"stuff": {"type": "array"}}}
        result = typescript_schema(schema)
        assert "any[]" in result

    def test_array_enum_items(self):
        schema = {
            "properties": {
                "roles": {"type": "array", "items": {"enum": ["admin", "user"]}}
            }
        }
        result = typescript_schema(schema)
        assert '("admin" | "user")[]' in result


class TestTypescriptSchemaRef:
    def test_ref_optional(self):
        schema = {"properties": {"parent": {"$ref": "#/$defs/ParentType"}}}
        result = typescript_schema(schema)
        assert "ParentType" in result
        assert "?" in result

    def test_ref_required(self):
        schema = {
            "properties": {"parent": {"$ref": "#/$defs/ParentType"}},
            "required": ["parent"],
        }
        result = typescript_schema(schema)
        assert "parent: ParentType" in result
        assert "?" not in result


class TestTypescriptSchemaDescription:
    def test_description_appended_after_type(self):
        schema = {
            "properties": {
                "email": {
                    "type": "string",
                    "description": "User email address",
                }
            },
            "required": ["email"],
        }
        result = typescript_schema(schema)
        assert "User email address" in result
        assert "email: string" in result
        assert result.index("email: string") < result.index("User email address")

    def test_no_description_no_dash(self):
        schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        result = typescript_schema(schema)
        assert " - " not in result


class TestTypescriptSchemaDefaultValues:
    def test_default_string(self):
        schema = {"properties": {"name": {"type": "string", "default": "world"}}}
        result = typescript_schema(schema)
        assert '= "world"' in result

    def test_default_none_is_null(self):
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

    def test_default_integer(self):
        schema = {"properties": {"count": {"type": "integer", "default": 42}}}
        result = typescript_schema(schema)
        assert "= 42" in result


class TestTypescriptSchemaMultipleFields:
    def test_two_line_output(self):
        schema = {
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        }
        result = typescript_schema(schema)
        lines = result.split("\n")
        assert len(lines) == 2

    def test_preserves_all_fields(self):
        schema = {
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["id", "count", "active"],
        }
        result = typescript_schema(schema)
        assert "id: string" in result
        assert "count: int" in result
        assert "active: bool" in result


class TestTypescriptSchemaComplexSchema:
    def test_schema_with_nested_defs_ref(self):
        """Fields referencing $defs render the model name correctly."""
        schema = {
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
                "tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}},
            },
            "required": ["address"],
        }
        result = typescript_schema(schema)
        assert "address: Address" in result
        assert "Tag[]" in result
        assert "?" not in result.split("\n")[0]  # address is required

    def test_anyof_array_with_null(self):
        schema = {
            "properties": {
                "items": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ]
                }
            },
            "required": ["items"],
        }
        result = typescript_schema(schema)
        assert "string[]" in result
        assert "null" in result
        assert "?" in result
