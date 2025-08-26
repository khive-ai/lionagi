from __future__ import annotations

from collections.abc import Mapping, Iterable
from typing import Any

from lionagi.base.morphism import Morphism
from lionagi.base.types import Branch
from lionagi.ops.core import BaseOp


class OpThenPatch(BaseOp):
    """
    Execute inner morphism; then copy selected result keys into Branch.ctx.
    'patch' can be either:
      - Iterable[str]: copy result[k] -> ctx[k] for each k
      - Mapping[str, str]: copy result[src] -> ctx[dst]
    """

    name = "op.then_patch"

    def __init__(self, inner: Morphism, patch: Mapping[str, str] | Iterable[str]):
        self.inner = inner
        if isinstance(patch, dict):
            self.patch_map = dict(patch)
        else:
            self.patch_map = {k: k for k in patch}  # identity
        self.requires = getattr(inner, "requires", set())
        self.io = bool(getattr(inner, "io", False))

    async def pre(self, br: Branch, **kw) -> bool:
        return await self.inner.pre(br, **kw)

    async def apply(self, br: Branch, **kw) -> dict[str, Any]:
        res = await self.inner.apply(br, **kw)
        for src_key, dst_key in self.patch_map.items():
            if src_key in res:
                br.ctx[dst_key] = res[src_key]
        return res

    async def post(self, br: Branch, result: dict[str, Any]) -> bool:
        return await self.inner.post(br, result)
