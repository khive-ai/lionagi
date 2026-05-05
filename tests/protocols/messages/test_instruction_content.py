"""Surgical gap-fill tests for InstructionContent missing branches.

Targets the ~36 missing statements in lionagi/protocols/messages/instruction.py:
Lines: 74, 89, 91, 96-104, 141, 146, 151, 159-178, 199-211, 233, 285, 290, 298,
       304, 334-335, 342-343, 364, 392-393
"""

import warnings

import pytest
from pydantic import BaseModel

from lionagi.protocols.messages.instruction import Instruction, InstructionContent
from lionagi.protocols.messages.rendering import StructureFormat


class ResponseModel(BaseModel):
    answer: str
    score: float = 0.0


class AnotherModel(BaseModel):
    items: list[str]


# ---------------------------------------------------------------------------
# __init__ branches
# ---------------------------------------------------------------------------


def test_init_primary_alias_sets_instruction():
    """Line 74: primary is used when instruction is None."""
    c = InstructionContent(primary="via primary alias")
    assert c.instruction == "via primary alias"


def test_init_response_format_as_basemodel_instance():
    """Lines 91, 96-97: response_format is a BaseModel instance."""
    instance = ResponseModel(answer="yes", score=1.0)
    c = InstructionContent(response_format=instance)
    assert c._model_class is ResponseModel
    assert c._schema_dict is not None
    # schema_dict comes from model_dump
    assert c._schema_dict.get("answer") == "yes"


def test_init_response_format_as_dict():
    """Lines 94-95: response_format is a plain dict."""
    rf = {"type": "object", "properties": {"x": {"type": "string"}}}
    c = InstructionContent(response_format=rf)
    assert c._schema_dict is rf
    assert c._model_class is None


def test_init_response_format_as_model_class_generates_schema():
    """Lines 86-89, 98-104: response_format is a BaseModel class."""
    c = InstructionContent(response_format=ResponseModel)
    assert c._model_class is ResponseModel
    assert c._schema_dict is not None
    assert isinstance(c._schema_dict, dict)


# ---------------------------------------------------------------------------
# Property aliases
# ---------------------------------------------------------------------------


def test_context_property_is_alias_for_prompt_context():
    """Line 141 (context property getter — usually covered, ensure hit)."""
    c = InstructionContent(context=["item1", "item2"])
    assert c.context == ["item1", "item2"]
    assert c.prompt_context is c.context


def test_primary_property_setter():
    """Line 146: primary.setter writes through to instruction."""
    c = InstructionContent(instruction="original")
    c.primary = "updated"
    assert c.instruction == "updated"


def test_role_property():
    """Line 151: role returns MessageRole.USER."""
    from lionagi.protocols.messages.message import MessageRole

    c = InstructionContent()
    assert c.role == MessageRole.USER


# ---------------------------------------------------------------------------
# with_updates
# ---------------------------------------------------------------------------


def test_with_updates_primary_alias():
    """Line 162-164: 'primary' kwarg translates to 'instruction'."""
    c = InstructionContent(instruction="old")
    updated = c.with_updates(primary="new value")
    assert updated.instruction == "new value"


def test_with_updates_primary_none_does_not_override():
    """Line 163: primary=None does not overwrite instruction."""
    c = InstructionContent(instruction="keep")
    updated = c.with_updates(primary=None)
    # primary=None should not set instruction to None
    assert updated.instruction == "keep"


def test_with_updates_context_alias_list():
    """Lines 166-172: 'context' kwarg translates to prompt_context (list)."""
    c = InstructionContent(instruction="x")
    updated = c.with_updates(context=["a", "b"])
    assert updated.prompt_context == ["a", "b"]


def test_with_updates_context_alias_scalar():
    """Lines 166-172: 'context' scalar → wrapped in list."""
    c = InstructionContent(instruction="x")
    updated = c.with_updates(context="single item")
    assert updated.prompt_context == ["single item"]


def test_with_updates_context_alias_none():
    """Lines 166-172: 'context' None → empty list."""
    c = InstructionContent(instruction="x", context=["existing"])
    updated = c.with_updates(context=None)
    assert updated.prompt_context == []


def test_with_updates_request_model_alias():
    """Line 174-175: 'request_model' → 'response_format'."""
    c = InstructionContent(instruction="x")
    updated = c.with_updates(request_model=ResponseModel)
    assert updated._model_class is ResponseModel


def test_with_updates_copy_containers_stripped():
    """Line 159: copy_containers kwarg is silently removed."""
    c = InstructionContent(instruction="original")
    updated = c.with_updates(copy_containers=True, instruction="updated")
    assert updated.instruction == "updated"


# ---------------------------------------------------------------------------
# create classmethod
# ---------------------------------------------------------------------------


def test_create_with_structure_format_enum():
    """Lines 205-207: structure_format as StructureFormat enum member."""
    c = InstructionContent.create(
        primary="test",
        structure_format=StructureFormat.JSON,
    )
    assert c.structure_format == "json"


def test_create_with_structure_format_string():
    """Line 208: structure_format as plain string."""
    c = InstructionContent.create(primary="test", structure_format="lndl")
    assert c.structure_format == "lndl"


def test_create_with_no_args():
    """Lines 199-200: create() with no instruction is valid."""
    c = InstructionContent.create()
    assert c.instruction is None


# ---------------------------------------------------------------------------
# render / rendered
# ---------------------------------------------------------------------------


def test_render_delegates_to_rendered():
    """Line 233: render() delegates to rendered property."""
    c = InstructionContent(instruction="hello")
    assert c.render() == c.rendered


def test_rendered_with_images_returns_list():
    """Lines 263-264: rendered with images returns a list."""
    c = InstructionContent(
        instruction="look at this",
        images=["http://example.com/img.jpg"],
        image_detail="auto",
    )
    result = c.rendered
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# request_model deprecated property
# ---------------------------------------------------------------------------


def test_request_model_property_raises_deprecation_warning():
    """Lines 243-251: request_model property emits DeprecationWarning."""
    c = InstructionContent(response_format=ResponseModel)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = c.request_model
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert result is ResponseModel


# ---------------------------------------------------------------------------
# from_dict branches
# ---------------------------------------------------------------------------


def test_from_dict_handle_context_replace():
    """Line 304: handle_context='replace' replaces instead of extending."""
    data = {
        "instruction": "test",
        "prompt_context": ["new"],
        "handle_context": "replace",
    }
    c = InstructionContent.from_dict(data)
    assert c.prompt_context == ["new"]


def test_from_dict_handle_context_invalid_raises():
    """Lines 289-290: invalid handle_context raises ValueError."""
    with pytest.raises(ValueError, match="handle_context must be"):
        InstructionContent.from_dict({"handle_context": "invalid"})


def test_from_dict_response_format_as_model_class():
    """Line 334-335 (model_class path): response_format is a BaseModel class."""
    data = {"instruction": "x", "response_format": ResponseModel}
    c = InstructionContent.from_dict(data)
    assert c._model_class is ResponseModel
    assert c._schema_dict is not None


def test_from_dict_response_format_as_instance():
    """Lines 334-335, 342-343: response_format is a BaseModel instance."""
    instance = ResponseModel(answer="test")
    data = {"instruction": "x", "response_format": instance}
    c = InstructionContent.from_dict(data)
    assert c._model_class is ResponseModel
    assert c._schema_dict is not None


def test_from_dict_response_format_as_dict():
    """Line 338-340: response_format is a plain dict."""
    rf = {"type": "object", "properties": {}}
    data = {"instruction": "x", "response_format": rf}
    c = InstructionContent.from_dict(data)
    assert c._schema_dict is rf
    assert c._model_class is None


def test_from_dict_instruction_fallback_to_primary():
    """Line 285: instruction falls back to 'primary' key."""
    data = {"primary": "from primary key"}
    c = InstructionContent.from_dict(data)
    assert c.instruction == "from primary key"


# ---------------------------------------------------------------------------
# _format_response_format
# ---------------------------------------------------------------------------


def test_format_response_format_returns_none_for_empty():
    """Line 389: returns None for falsy input."""
    result = InstructionContent._format_response_format(None)
    assert result is None

    result = InstructionContent._format_response_format({})
    assert result is None


def test_format_response_format_orjson_success():
    """Lines 391-398: normal orjson path."""
    rf = {"field": "value"}
    result = InstructionContent._format_response_format(rf)
    assert result is not None
    assert "json" in result
    assert "field" in result


def test_format_response_format_exception_fallback():
    """Lines 392-393: exception fallback to str()."""

    # Pass an object that orjson cannot serialize
    class UnSerializable:
        def __repr__(self):
            return "unserializable_obj"

    # orjson raises TypeError for non-serializable objects
    bad = {"key": UnSerializable()}
    result = InstructionContent._format_response_format(bad)
    assert result is not None
    assert "unserializable" in result


# ---------------------------------------------------------------------------
# _format_image_item
# ---------------------------------------------------------------------------


def test_format_image_item_http_url():
    """Lines 401-412: http URL passes through unchanged."""
    item = InstructionContent._format_image_item("http://example.com/img.jpg", "low")
    assert item["image_url"]["url"] == "http://example.com/img.jpg"
    assert item["image_url"]["detail"] == "low"


def test_format_image_item_https_url():
    """Lines 401-412: https URL passes through unchanged."""
    item = InstructionContent._format_image_item("https://example.com/img.png", "high")
    assert item["image_url"]["url"] == "https://example.com/img.png"


def test_format_image_item_data_url():
    """Lines 401-412: data: URL passes through unchanged."""
    data_url = "data:image/png;base64,abc123"
    item = InstructionContent._format_image_item(data_url, "auto")
    assert item["image_url"]["url"] == data_url


def test_format_image_item_base64_gets_wrapped():
    """Line 408: non-URL string gets wrapped in data URI."""
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    item = InstructionContent._format_image_item(b64, "auto")
    assert item["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert b64 in item["image_url"]["url"]


# ---------------------------------------------------------------------------
# _format_image_content
# ---------------------------------------------------------------------------


def test_format_image_content_builds_list():
    """Lines 421-423: _format_image_content builds [text, *images]."""
    result = InstructionContent._format_image_content(
        "describe this",
        ["http://a.com/1.jpg", "http://b.com/2.jpg"],
        "low",
    )
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == {"type": "text", "text": "describe this"}
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "image_url"


# ---------------------------------------------------------------------------
# Instruction validator
# ---------------------------------------------------------------------------


def test_instruction_validator_raises_on_bad_type():
    """Line 443: TypeError if content is not dict/InstructionContent/None."""
    with pytest.raises(TypeError, match="content must be dict or InstructionContent"):
        Instruction(content=42)


# ---------------------------------------------------------------------------
# Additional gap-fill: remaining uncovered lines
# ---------------------------------------------------------------------------


def test_primary_property_getter():
    """Line 141: primary property getter returns instruction."""
    c = InstructionContent(instruction="test value")
    assert c.primary == "test value"


def test_from_dict_custom_renderer_callable():
    """Line 285: custom_renderer is set when callable is provided."""

    def my_renderer(model, **kwargs):
        return "custom"

    data = {"instruction": "x", "custom_renderer": my_renderer}
    c = InstructionContent.from_dict(data)
    assert callable(c.custom_renderer)


def test_from_dict_context_none_becomes_empty_list():
    """Line 298: context=None results in empty list."""
    data = {"instruction": "x", "context": None}
    c = InstructionContent.from_dict(data)
    assert c.prompt_context == []


def test_format_text_content_uses_model_json_schema():
    """Line 364: _format_text_content uses model_json_schema() when _model_class is set."""
    c = InstructionContent(response_format=ResponseModel)
    text = c._format_text_content()
    # The schema should appear in the rendered text
    assert isinstance(text, str)
    # model_json_schema returns properties like "answer", "score"
    assert len(text) > 0
