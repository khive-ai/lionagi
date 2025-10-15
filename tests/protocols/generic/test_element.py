from datetime import datetime
from uuid import UUID, uuid4

import pytest

from lionagi.protocols.generic.element import ID, Element, validate_order


def test_UUID_creation():
    id_type = uuid4()
    assert isinstance(id_type, UUID)


def test_UUID_validation():
    uuid_str = "123e4567-e89b-42d3-a456-426614174000"
    id_type = ID.get_id(uuid_str)
    assert str(id_type) == uuid_str


def test_element_creation():
    element = Element()
    assert isinstance(element.id, UUID)
    assert isinstance(element.created_at, float)
    assert isinstance(element.metadata, dict)


def test_element_metadata_validation():
    element = Element(metadata={"key": "value"})
    assert element.metadata == {"key": "value"}


def test_element_timestamp():
    now = datetime.now().timestamp()
    element = Element()
    assert element.created_at >= now


def test_element_equality():
    element1 = Element()
    element2 = Element()
    assert element1 != element2
    assert element1 == element1


def test_element_to_dict():
    element = Element()
    element_dict = element.to_dict()
    assert isinstance(element_dict, dict)
    assert "id" in element_dict
    assert "created_at" in element_dict
    assert "metadata" in element_dict


def test_validate_order():
    element1 = Element()
    element2 = Element()
    order = validate_order([element1, element2])
    assert len(order) == 2
    assert order[0] == element1.id
    assert order[1] == element2.id


def test_id_class():
    element = Element()
    assert ID.get_id(element) == element.id
    assert ID.is_id(element.id) is True
    assert ID.is_id("invalid") is False


def test_element_from_dict():
    element = Element()
    element_dict = element.to_dict()
    new_element = Element.from_dict(element_dict)
    assert new_element.id == element.id
    assert new_element.created_at == element.created_at
    assert new_element.metadata == element.metadata


def test_invalid_UUID_validation():
    with pytest.raises(Exception):
        UUID.validate("invalid-uuid")


def test_element_metadata_class_mismatch():
    with pytest.raises(ValueError):
        Element(metadata={"lion_class": "invalid.class"})


def test_element_invalid_timestamp():
    with pytest.raises(ValueError):
        Element(created_at="invalid-timestamp")


def test_validate_order_invalid_input():
    with pytest.raises(ValueError):
        validate_order([1, 2, 3])


def test_element_hash():
    element = Element()
    assert hash(element) == hash(element.id)


def test_element_bool():
    element = Element()
    assert bool(element) is True


def test_element_class_name():
    assert Element.class_name() == "Element"
    assert (
        Element.class_name(full=True)
        == "lionagi.protocols.generic.element.Element"
    )
