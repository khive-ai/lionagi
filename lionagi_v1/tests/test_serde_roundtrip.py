from lionagi_v1.base.serde import from_json, to_json
from lionagi_v1.base.types import Branch, Capability


def test_branch_roundtrip_json():
    br = Branch(name="serde")
    br.caps = (
        Capability(subject=br.id, rights={"net.out", "fs.read:/data/*"}),
    )
    js = to_json(br)
    br2 = from_json(js, type_=Branch)
    assert br2.name == br.name
    assert {frozenset(c.rights) for c in br2.caps} == {
        frozenset(c.rights) for c in br.caps
    }
    assert br2.id == br.id and br2.ts == br.ts
