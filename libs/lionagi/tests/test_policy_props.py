import os
import string

from hypothesis import given
from hypothesis import strategies as st

from lionagi.base.policy import _covers, _split  # internal but stable enough for tests

# ---------- Strategies ----------

SEG = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8)


@st.composite
def path_prefix(draw):
    segs = draw(st.lists(SEG, min_size=1, max_size=4))
    # produce an absolute prefix ending with '/'
    return "/" + "/".join(segs) + "/"


# ---------- Properties ----------


@given(prefix=path_prefix(), leaf=SEG)
def test_prefix_star_covers_concrete(prefix, leaf):
    """have: fs.read:{prefix}*  covers  req: fs.read:{prefix}{leaf}"""
    have = f"fs.read:{prefix}*"
    req = f"fs.read:{prefix}{leaf}"
    assert _covers(have, req)


@given(prefix=path_prefix(), leaf=SEG)
def test_req_wildcard_not_covered_by_concrete_have(prefix, leaf):
    """req with wildcard cannot be covered by concrete have (defense-in-depth)."""
    have = f"fs.read:{prefix}{leaf}"  # concrete
    req = f"fs.read:{prefix}{leaf}/*"  # wildcard in req
    assert not _covers(have, req)


@given(a=st.lists(SEG, min_size=1, max_size=3), b=st.lists(SEG, min_size=1, max_size=3))
def test_both_wildcard_prefix_star_rule(a, b):
    """
    have: X/* covers req: X/Y/*  iff  Y is under X (prefix rule).
    """
    base = "/" + "/".join(a) + "/"
    child = base + "/".join(b) + "/"  # may duplicate slashes; policy is robust to that

    have = f"fs.read:{base}*"
    req_under = f"fs.read:{child}*"

    # Always test that prefix coverage works
    assert _covers(have, req_under), (have, req_under)

    # Test that unrelated paths don't get covered (only if we can construct a truly different path)
    if len(a) > 1:  # Only test this when we have a multi-segment base path
        # Create an unrelated path that shares only the first segment
        unrelated_base = "/" + a[0] + "different/"
        req_unrelated = f"fs.read:{unrelated_base}*"
        assert not _covers(have, req_unrelated), (have, req_unrelated)


@given(prefix=path_prefix(), leaf=SEG)
def test_domain_mismatch_is_denied(prefix, leaf):
    have = f"net.out:{prefix}*"
    req = f"fs.read:{prefix}{leaf}"
    assert not _covers(have, req)


def test_split_normalizes_absolute_paths():
    s = "fs.read:/a/b/../c/./d"
    domain, res = _split(s)
    assert domain == "fs.read"
    assert res == os.path.normpath("/a/b/../c/./d")  # -> "/a/c/d"
