# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionagi.lndl.ast import (
    ASTNode,
    Expr,
    Identifier,
    Lact,
    Literal,
    Lvar,
    OutBlock,
    Program,
    RLvar,
    Stmt,
)


def test_literal_str():
    lit = Literal(value="hello")
    assert lit.value == "hello"


def test_literal_int():
    lit = Literal(value=42)
    assert lit.value == 42


def test_literal_float():
    lit = Literal(value=3.14)
    assert lit.value == 3.14


def test_literal_bool():
    lit = Literal(value=True)
    assert lit.value is True


def test_identifier():
    ident = Identifier(name="foo")
    assert ident.name == "foo"


def test_lvar_basic():
    lvar = Lvar(model="Report", field="title", alias="t", content="Hello")
    assert lvar.model == "Report"
    assert lvar.field == "title"
    assert lvar.alias == "t"
    assert lvar.content == "Hello"


def test_rlvar_basic():
    rlvar = RLvar(alias="raw", content="some text")
    assert rlvar.alias == "raw"
    assert rlvar.content == "some text"


def test_lact_namespaced():
    lact = Lact(model="Report", field="summary", alias="s", call="summarize(text='x')")
    assert lact.model == "Report"
    assert lact.field == "summary"
    assert lact.alias == "s"
    assert lact.call == "summarize(text='x')"


def test_lact_direct():
    lact = Lact(model=None, field=None, alias="data", call="fetch(url='http://x')")
    assert lact.model is None
    assert lact.field is None
    assert lact.alias == "data"


def test_out_block_empty():
    ob = OutBlock(fields={})
    assert ob.fields == {}


def test_out_block_with_fields():
    ob = OutBlock(fields={"report": ["t", "s"], "score": 0.9})
    assert ob.fields["report"] == ["t", "s"]
    assert ob.fields["score"] == 0.9


def test_program_minimal():
    prog = Program(lvars=[], lacts=[], out_block=None)
    assert prog.lvars == []
    assert prog.lacts == []
    assert prog.out_block is None


def test_program_with_out_block():
    ob = OutBlock(fields={"x": ["a"]})
    lvar = Lvar(model="M", field="f", alias="a", content="val")
    prog = Program(lvars=[lvar], lacts=[], out_block=ob)
    assert len(prog.lvars) == 1
    assert prog.out_block is not None


def test_inheritance_chain():
    assert issubclass(Literal, Expr)
    assert issubclass(Expr, ASTNode)
    assert issubclass(Identifier, Expr)
    assert issubclass(Lvar, Stmt)
    assert issubclass(RLvar, Stmt)
    assert issubclass(Lact, Stmt)
    assert issubclass(OutBlock, Stmt)
    assert issubclass(Stmt, ASTNode)


def test_dataclass_slots():
    # Dataclasses with slots=True don't allow arbitrary attribute assignment
    lvar = Lvar(model="M", field="f", alias="a", content="c")
    with pytest.raises(AttributeError):
        lvar.nonexistent = "x"


def test_all_exports():
    from lionagi.lndl.ast import __all__

    expected = {
        "ASTNode",
        "Expr",
        "Identifier",
        "Lact",
        "Literal",
        "Lvar",
        "OutBlock",
        "Program",
        "RLvar",
        "Stmt",
    }
    assert expected == set(__all__)
