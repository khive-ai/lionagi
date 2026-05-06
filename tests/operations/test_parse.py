# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/parse.py — _direct_parse logic (no LLM required)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lionagi._errors import ConfigurationError, ExecutionError, ValidationError
from lionagi.operations.parse import ParseParams, _direct_parse
from lionagi.ln.fuzzy import HandleUnmatched
from lionagi.protocols.messages.rendering import StructureFormat

# ---------------------------------------------------------------------------
# _direct_parse — JSON path
# ---------------------------------------------------------------------------


class TestDirectParseJSON:
    def test_extracts_json_from_text(self):
        text = '{"name": "Alice", "age": 30}'
        result = _direct_parse(text=text, target_keys=["name", "age"])
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_extracts_json_from_fenced_text(self):
        text = '```json\n{"name": "Bob"}\n```'
        result = _direct_parse(text=text, target_keys=["name"])
        assert result["name"] == "Bob"

    def test_with_fill_value_for_missing_keys(self):
        text = '{"name": "Alice"}'
        result = _direct_parse(
            text=text,
            target_keys=["name", "missing_key"],
            handle_unmatched=HandleUnmatched.FORCE,
            fill_value="N/A",
        )
        assert result["name"] == "Alice"

    def test_empty_json_object_raises(self):
        # {} is treated as empty sentinel by is_sentinel(..., additions={"none","empty"})
        text = "{}"
        with pytest.raises(ExecutionError):
            _direct_parse(text=text, target_keys=["a"])

    def test_no_json_raises_execution_error(self):
        text = "This has no JSON at all"
        with pytest.raises(ExecutionError):
            _direct_parse(text=text, target_keys=["key"])

    def test_similarity_threshold_used(self):
        # key name close to target with high threshold should match
        text = '{"nme": "Alice"}'
        result = _direct_parse(
            text=text,
            target_keys=["name"],
            similarity_threshold=0.5,  # low threshold, nme ≈ name
        )
        assert isinstance(result, dict)

    def test_fill_mapping(self):
        text = '{"a": 1}'
        result = _direct_parse(
            text=text,
            target_keys=["a", "b"],
            handle_unmatched=HandleUnmatched.FORCE,
            fill_mapping={"b": "default_b"},
        )
        assert result.get("b") == "default_b"

    def test_sentinel_target_keys_raises(self):
        from lionagi.ln.types._sentinel import Unset

        with pytest.raises(ValidationError):
            _direct_parse(text="hi", target_keys=Unset)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValidationError):
            _direct_parse(
                text="test",
                target_keys=["k"],
                structure_format="unsupported_format",
            )


# ---------------------------------------------------------------------------
# _direct_parse — CUSTOM path
# ---------------------------------------------------------------------------


class TestDirectParseCustom:
    def test_custom_parser_called(self):
        def my_parser(text, keys):
            return {k: "val" for k in keys}

        result = _direct_parse(
            text="anything",
            target_keys=["x"],
            structure_format=StructureFormat.CUSTOM,
            custom_parser=my_parser,
        )
        assert result["x"] == "val"

    def test_custom_parser_none_raises(self):
        with pytest.raises(ConfigurationError):
            _direct_parse(
                text="anything",
                target_keys=["x"],
                structure_format=StructureFormat.CUSTOM,
                custom_parser=None,
            )

    def test_custom_parser_exception_raises_execution_error(self):
        def bad_parser(text, keys):
            raise RuntimeError("parser failed")

        with pytest.raises(ExecutionError):
            _direct_parse(
                text="anything",
                target_keys=["x"],
                structure_format=StructureFormat.CUSTOM,
                custom_parser=bad_parser,
            )


# ---------------------------------------------------------------------------
# _direct_parse — LNDL path
# ---------------------------------------------------------------------------


class TestDirectParseLNDL:
    def test_lndl_requires_operable(self):
        from lionagi.lndl.errors import MissingOutBlockError

        with pytest.raises(ConfigurationError):
            _direct_parse(
                text="<lvar x>hello</lvar>",
                target_keys=["x"],
                structure_format=StructureFormat.LNDL,
                operable=None,
            )

    def test_lndl_with_operable_and_out_block(self):
        from lionagi.ln.types import Operable, Spec

        op = Operable([Spec(str, name="greeting")])
        text = "<lvar greeting>Hello world</lvar>\nOUT{\n  greeting: greeting\n}"
        result = _direct_parse(
            text=text,
            target_keys=["greeting"],
            structure_format=StructureFormat.LNDL,
            operable=op,
        )
        assert isinstance(result, dict)

    def test_lndl_missing_out_block_raises(self):
        from lionagi.lndl.errors import MissingOutBlockError
        from lionagi.ln.types import Operable, Spec

        op = Operable([Spec(str, name="x")])
        text = "<lvar x>value</lvar>"  # no OUT{} block
        with pytest.raises(MissingOutBlockError):
            _direct_parse(
                text=text,
                target_keys=["x"],
                structure_format=StructureFormat.LNDL,
                operable=op,
            )


# ---------------------------------------------------------------------------
# ParseParams construction
# ---------------------------------------------------------------------------


class TestParseParams:
    def test_basic_construction(self):
        p = ParseParams(text='{"a": 1}', target_keys=["a"])
        assert p.text == '{"a": 1}'

    def test_default_structure_format(self):
        p = ParseParams(text="t")
        assert p.structure_format == StructureFormat.JSON

    def test_default_max_retries(self):
        p = ParseParams(text="t")
        assert p.max_retries == 3

    def test_sentinel_target_keys(self):
        from lionagi.ln.types._sentinel import Unset, is_sentinel

        p = ParseParams(text="t")
        assert is_sentinel(p.target_keys)

    def test_sentinel_imodel(self):
        p = ParseParams(text="t")
        assert p.imodel is None  # default, not sentinel
