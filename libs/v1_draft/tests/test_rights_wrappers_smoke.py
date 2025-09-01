import anyio
import pytest

from lionagi.base.graph import OpGraph, OpNode
from lionagi.base.ipu import StrictIPU, default_invariants
from lionagi.base.runner import Runner
from lionagi.base.types import create_branch
from lionagi.morph.binders import BoundOp
from lionagi.morph.wrappers import OpThenPatch
from lionagi.ops.core import HTTPGet, WithRetry, WithTimeout


class DummyHttpClient:
    async def get(self, url: str):
        return 200, f"ok:{url}"


def run(g, br):
    return anyio.run(lambda: Runner(StrictIPU(default_invariants())).run(br, g))


def single_graph(node: OpNode) -> OpGraph:
    return OpGraph(nodes={node.id: node}, roots={node.id})


def test_direct_httpget_dynamic_rights_ok():
    """
    Sanity check: raw HTTPGet computes host-scoped rights and passes with host-only cap.
    """
    inner = HTTPGet(client=DummyHttpClient())
    node = OpNode(m=inner)
    node.params["url"] = "https://example.com/a"

    # Branch only has host-scoped cap, NOT net.out:*
    br = create_branch(ctx={}, capabilities={"net.out:example.com"})

    results = run(single_graph(node), br)
    r = results[node.id]
    assert r["status"] == 200


def test_opthenpatch_forwards_required_rights():
    """
    Outer wrapper must forward dynamic rights from inner, or policy will require net.out:* and fail.
    """
    inner = HTTPGet(client=DummyHttpClient())
    wrapped = OpThenPatch(inner, patch=["status"])  # copy status -> ctx["status"]
    node = OpNode(m=wrapped)
    node.params["url"] = "https://example.com/b"

    br = create_branch(ctx={}, capabilities={"net.out:example.com"})
    results = run(single_graph(node), br)
    assert br.ctx["status"] == 200
    assert results[node.id]["status"] == 200


def test_withtimeout_and_withretry_chain_forwards_required_rights():
    """
    Stacked wrappers should still preserve/forward dynamic rights:
      WithTimeout(WithRetry(OpThenPatch(HTTPGet(...))))
    """
    inner = HTTPGet(client=DummyHttpClient())
    op = OpThenPatch(inner, patch=["status"])
    op = WithRetry(op, retries=1, backoff_ms=5, jitter=False)
    op = WithTimeout(op, timeout_ms=1000)

    node = OpNode(m=op)
    node.params["url"] = "https://example.com/c"

    br = create_branch(ctx={}, capabilities={"net.out:example.com"})
    results = run(single_graph(node), br)
    assert br.ctx["status"] == 200
    assert results[node.id]["status"] == 200


def test_boundop_binds_from_ctx_and_forwards_required_rights():
    """
    BoundOp must rebuild kwargs using bind/defaults/runtime **kw so that
    inner.required_rights(**call_kw) sees url derived from ctx.
    """
    inner = HTTPGet(client=DummyHttpClient())
    bound = BoundOp(inner, bind={"url": "fetch_url"})
    node = OpNode(m=bound)
    # url is NOT in node.params; it's in ctx and must be bound
    br = create_branch(
        ctx={"fetch_url": "https://example.com/d"}, capabilities={"net.out:example.com"}
    )

    results = run(single_graph(node), br)
    assert results[node.id]["status"] == 200
