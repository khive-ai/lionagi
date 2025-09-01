from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..base import Branch, Morphism
from ..ops import BaseOp


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

        # Propagate basic attributes
        self.requires = getattr(inner, "requires", set())
        self.io = bool(getattr(inner, "io", False))

        # Declare ctx_writes as patch targets union with inner's ctx_writes
        patch_targets = set(self.patch_map.values())
        inner_ctx_writes = getattr(inner, "ctx_writes", None)
        self.ctx_writes = (
            patch_targets if inner_ctx_writes is None else set(inner_ctx_writes) | patch_targets
        )

        # Propagate other policy surface attributes for IPU invariant checking
        self.result_schema = getattr(inner, "result_schema", None)
        self.result_keys = getattr(inner, "result_keys", None)
        self.result_bytes_limit = getattr(inner, "result_bytes_limit", None)
        self.latency_budget_ms = getattr(inner, "latency_budget_ms", None)

    def required_rights(self, **kw) -> set[str]:
        """Forward dynamic rights; use dynamic if available, static fallback otherwise."""
        fn = getattr(self.inner, "required_rights", None)
        if callable(fn):
            try:
                return set(fn(**kw))
            except Exception:
                pass
        return set(getattr(self.inner, "requires", set()) or self.requires)

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
