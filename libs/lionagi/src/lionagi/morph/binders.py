from __future__ import annotations

from typing import Any, Dict, Mapping

from lionagi.base.morphism import Morphism
from lionagi.base.types import Branch
from lionagi.ops.core import BaseOp


def _build_call_kwargs(
    br: Branch, runtime_kw: Dict[str, Any], bind: Mapping[str, str], defaults: Mapping[str, Any]
) -> Dict[str, Any]:
    # 1) from ctx via binding
    call_kw: Dict[str, Any] = {param: br.ctx[src] for param, src in bind.items() if src in br.ctx}
    # 2) default literals for any missing
    for k, v in defaults.items():
        call_kw.setdefault(k, v)
    # 3) let runtime kwargs (node.params/runner merge) override
    call_kw.update(runtime_kw)
    return call_kw


class BoundOp(BaseOp):
    """
    Wrap an inner morphism. Before calling it, build kwargs by pulling
    values from Branch.ctx using a binding map: param -> ctx_key.
    """

    name = "op.bound"

    def __init__(
        self,
        inner: Morphism,
        bind: Mapping[str, str] | None = None,
        defaults: Mapping[str, Any] | None = None,
    ):
        self.inner = inner
        self.bind = dict(bind or {})
        self.defaults = dict(defaults or {})
        self.requires = getattr(inner, "requires", set())
        self.io = bool(getattr(inner, "io", False))

    async def pre(self, br: Branch, **kw) -> bool:
        call_kw = _build_call_kwargs(br, kw, self.bind, self.defaults)
        return await self.inner.pre(br, **call_kw)

    async def apply(self, br: Branch, **kw) -> Dict[str, Any]:
        call_kw = _build_call_kwargs(br, kw, self.bind, self.defaults)
        return await self.inner.apply(br, **call_kw)

    async def post(self, br: Branch, result: Dict[str, Any]) -> bool:
        # post doesn't need new kwargs, pass result through
        return await self.inner.post(br, result)
