import pytest

from lionagi_v1.base.forms import Form, parse_assignment


def test_invalid_segment_raises():
    with pytest.raises(ValueError):
        parse_assignment("a,b c")  # missing '->'


def test_final_outputs_multiple():
    f = Form(assignment="a->b; b->c,d; d->e")
    f.parse()
    # produced: [b,c,d,e]; inputs used: [a,b,d]
    # final outputs: c,e (d is consumed; b consumed)
    assert f.output_fields == ["c", "e"]


def test_initial_inputs_calculated():
    f = Form(assignment="a->b; b,c->d")
    f.parse()
    # 'a' and 'c' are initial inputs (b produced before used; c never produced)
    assert f.input_fields == ["a", "c"]


def test_check_inputs_and_outputs_errors():
    f = Form(assignment="x,y->z")
    f.parse()
    with pytest.raises(ValueError):
        f.check_inputs()
    f.values.update({"x": 1, "y": 2})
    f.check_inputs()  # ok now
    with pytest.raises(ValueError):
        f.check_outputs()
    f.values["z"] = 3
    f.check_outputs()
