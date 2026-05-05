import pytest

from lionagi.ln.types import Undefined
from lionagi.models.note import Note


def test_content_key_round_trips_as_user_data():
    note = Note(content={"x": 1})

    assert note.to_dict() == {"content": {"x": 1}}
    assert Note.from_dict(note.to_dict()).to_dict() == {"content": {"x": 1}}
    assert Note.from_content({"x": 1}).to_dict() == {"x": 1}


def test_json_model_dump_strips_sentinels():
    note = Note(a=Undefined)

    assert note.to_dict(mode="json") == {}
    assert note.model_dump(mode="json") == {"content": {}}


def test_flatten_preserves_empty_containers_and_numeric_root_keys():
    note = Note(a={}, b=[])
    assert note.flatten() == {"a": {}, "b": []}
    assert Note.unflatten(note.flatten()).to_dict() == {"a": {}, "b": []}

    numeric = Note()
    numeric["0"] = "zero"
    numeric["1"] = "one"
    assert Note.unflatten(numeric.flatten()).to_dict() == {
        "0": "zero",
        "1": "one",
    }


def test_flatten_rebuilds_nested_lists():
    note = Note(items=["a", "b"])

    assert Note.unflatten(note.flatten()).to_dict() == {"items": ["a", "b"]}


def test_update_uses_set_semantics_for_missing_or_scalar_values():
    note = Note()

    note.update("x", 1)
    assert note.to_dict() == {"x": 1}

    note.update("x", 2)
    assert note.to_dict() == {"x": 2}

    note.update("items", [])
    note.update("items", 1)
    assert note.to_dict()["items"] == [1]


def test_top_level_note_keys_must_be_strings():
    with pytest.raises(TypeError):
        Note()[0] = "bad"
