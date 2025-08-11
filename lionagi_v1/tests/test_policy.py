from lionagi_v1.base.policy import policy_check
from lionagi_v1.base.types import Branch, Capability


class NeedsNet:
    name = "needs.net"
    requires = {"net.out"}

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {}

    async def post(self, br, res):
        return True


class NeedsFS:
    name = "needs.fs"
    requires = {"fs.read:/data/*"}

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {}

    async def post(self, br, res):
        return True


def test_policy_allows_and_denies():
    br = Branch(name="p")
    br.caps = (
        Capability(
            subject=br.id, rights={"net.out", "fs.read:/data/alpha.txt"}
        ),
    )

    assert policy_check(br, NeedsNet())
    assert not policy_check(
        br, NeedsFS()
    )  # needs fs.read:/data/*; we only have a specific file
