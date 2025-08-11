from lionagi_v1.base.registry import all_names, get


def test_registry_contains_priority_a_ops():
    names = set(all_names())
    for n in [
        "llm.generate",
        "http.get",
        "fs.read",
        "kv.get",
        "kv.set",
        "ctx.set",
        "subgraph.run",
        "with.retry",
        "with.timeout",
    ]:
        assert n in names, f"missing registry name: {n}"
    assert get("llm.generate") is not None
