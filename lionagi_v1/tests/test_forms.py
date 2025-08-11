from lionagi_v1.base.forms import Form


def test_form_parse_and_checks():
    f = Form(assignment="a,b -> c; c -> d")
    f.parse()
    assert f.input_fields == ["a", "b"]
    assert f.output_fields == ["d"]  # final outputs only
    f.values.update({"a": 1, "b": 2})
    f.check_inputs()
    # simulate two steps: produce c, then d
    f.values["c"] = 3
    f.values["d"] = 4
    f.check_outputs()
    assert f.get_results() == {"d": 4}
