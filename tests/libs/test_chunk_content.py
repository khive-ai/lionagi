# tests/libs/test_chunk_content.py
import importlib

import pytest


def test_chunk_content_chars_dicts(mod_paths, ensure_fake_lionagi):
    mod = importlib.import_module(mod_paths["chunk_mod"])

    content = "0123456789abcdefghij"  # 20 chars
    res = mod.chunk_content(
        content=content,
        chunk_by="chars",
        chunk_size=8,
        overlap=0.25,  # overlap_size = int(8*0.25/2)=1
        threshold=3,
        metadata={"doc_id": "D1"},
        as_node=False,
    )
    # Expect: chunks: [0..9], [7..17], [15..end] => 3 chunks
    assert isinstance(res, list) and res
    assert all(isinstance(x, dict) for x in res)
    total = res[0]["total_chunks"]
    assert all(x["total_chunks"] == total for x in res)
    assert {x["doc_id"] for x in res} == {"D1"}
    # chunk_size matches the length of the string chunk
    assert all(len(x["chunk_content"]) == x["chunk_size"] for x in res)
    # chunk_id increments
    assert [x["chunk_id"] for x in res] == list(range(1, len(res) + 1))


def test_chunk_content_tokens_with_custom_tokenizer_and_return_tokens(
    mod_paths, ensure_fake_lionagi
):
    mod = importlib.import_module(mod_paths["chunk_mod"])

    content = "a|b|c|d|e|f|g|h|i"

    # Custom tokenizer uses the kwargs passed through
    def tok(s: str, sep: str = "|"):
        return s.split(sep)

    res = mod.chunk_content(
        content=content,
        chunk_by="tokens",
        tokenizer=tok,
        chunk_size=4,
        overlap=0.5,  # overlap_size=int(4*0.5/2)=1
        threshold=1,
        metadata={"doc": 2},
        return_tokens=True,
        as_node=False,
        sep="|",
    )
    assert isinstance(res, list) and res
    assert all(isinstance(x, dict) for x in res)
    # chunk_content should be a list of tokens here
    assert all(isinstance(x["chunk_content"], list) for x in res)
    # chunk_size equals number of tokens in the chunk
    assert all(len(x["chunk_content"]) == x["chunk_size"] for x in res)
    assert {x["doc"] for x in res} == {2}


def test_chunk_content_as_node(mod_paths, ensure_fake_lionagi):
    mod = importlib.import_module(mod_paths["chunk_mod"])
    res = mod.chunk_content(
        content="a b c d e f g h i j",
        chunk_by="tokens",
        tokenizer=str.split,
        chunk_size=3,
        overlap=0.0,
        threshold=0,
        metadata={"src": "t"},
        return_tokens=False,
        as_node=True,
    )
    # Node objects (from our fake lionagi or real one if installed)
    assert res and hasattr(res[0], "content") and hasattr(res[0], "metadata")
    assert res[0].metadata["total_chunks"] == len(res)
