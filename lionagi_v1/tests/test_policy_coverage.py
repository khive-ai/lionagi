from lionagi_v1.base.policy import policy_check
from lionagi_v1.base.types import Branch, Capability


# Helpers
class Needs:
    def __init__(self, reqs):
        self.name = "needs"
        self.requires = set(reqs)

    async def pre(self, *a, **k):
        return True

    async def apply(self, *a, **k):
        return {}

    async def post(self, *a, **k):
        return True


def branch_with(rights: set[str]) -> Branch:
    br = Branch(name="p")
    br.caps = (Capability(subject=br.id, rights=rights),)
    return br


def test_specific_does_not_cover_wildcard_req():
    br = branch_with({"fs.read:/data/alpha.txt"})
    assert not policy_check(br, Needs({"fs.read:/data/*"}))


def test_wildcard_have_covers_specific_req():
    br = branch_with({"fs.read:/data/*"})
    assert policy_check(br, Needs({"fs.read:/data/alpha.txt"}))


def test_broad_have_covers_any_resource_req():
    br = branch_with({"fs.read"})
    assert policy_check(br, Needs({"fs.read:/data/alpha.txt"}))
    assert policy_check(br, Needs({"fs.read:/"}))


def test_broad_requirement_not_covered_by_narrow_have():
    br = branch_with({"fs.read:/data/alpha.txt"})
    assert not policy_check(br, Needs({"fs.read"}))


def test_domain_mismatch_fails():
    br = branch_with({"net.out"})
    assert not policy_check(br, Needs({"fs.read:/x"}))


def test_multiple_requirements_all_must_hold():
    br = branch_with({"fs.read:/data/*"})
    assert not policy_check(br, Needs({"fs.read:/data/alpha.txt", "net.out"}))
    br2 = branch_with({"fs.read:/data/*", "net.out"})
    assert policy_check(br2, Needs({"fs.read:/data/alpha.txt", "net.out"}))


def test_wildcard_breadth_comparison():
    br = branch_with({"fs.read:/data/*"})
    # require a broader wildcard (/data/** not our grammar but /data/*alpha* is "broader" in match terms)
    # Our conservative rule: have must be at least as broad -> req wildcard broader than have wildcard => deny
    assert not policy_check(br, Needs({"fs.read:/data/*alpha*"}))
