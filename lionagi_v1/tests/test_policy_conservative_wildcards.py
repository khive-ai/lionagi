# tests/test_policy_conservative_wildcards.py
from lionagi_v1.base.policy import policy_check
from lionagi_v1.base.types import Branch, Capability


class Needs:
    def __init__(self, reqs):
        self.name = "n"
        self.requires = set(reqs)

    async def pre(self, *a, **k):
        return True

    async def apply(self, *a, **k):
        return {}

    async def post(self, *a, **k):
        return True


def B(*rights):
    br = Branch(name="b")
    br.caps = (Capability(subject=br.id, rights=set(rights)),)
    return br


def test_complex_wildcard_not_covered():
    # /data/*alpha* should NOT be covered by /data/* (conservative rule)
    br = B("fs.read:/data/*")
    assert not policy_check(br, Needs({"fs.read:/data/*alpha*"}))


def test_prefix_star_coverage():
    # /data/* (have) does cover /data/subdir/* (req)
    br = B("fs.read:/data/*")
    assert policy_check(br, Needs({"fs.read:/data/subdir/*"}))
