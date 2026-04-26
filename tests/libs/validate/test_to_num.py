import pytest

from lionagi.libs.validate.to_num import to_num


def test_to_num_rejects_sequence():
    with pytest.raises(TypeError):
        to_num([1, 2])


@pytest.mark.parametrize(
    "value, num_type, expected",
    [
        (True, int, 1),
        (False, float, 0.0),
    ],
)
def test_to_num_converts_bool_with_requested_type(value, num_type, expected):
    result = to_num(value, num_type=num_type)
    assert result == expected
    assert isinstance(result, num_type)
