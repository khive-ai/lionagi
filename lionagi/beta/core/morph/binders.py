from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lionagi.beta.core.morphism import Morphism
from lionagi.beta.core.types import Principal
from .wrappers import BaseOp, inherit_contract, maybe_await, register


def _build_call_kwargs(
    br: Principal,
    runtime_kw: dict[str, Any],
    bind: Mapping[str, str],
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    # 1) from ctx via binding
    call_kw: dict[str, Any] = {
        param: br.ctx[src] for param, src in bind.items() if src in br.ctx
    }
    # 2) default literals for any missing
    for k, v in defaults.items():
        call_kw.setdefault(k, v)
    # 3) let runtime kwargs (node.params/runner merge) override
    call_kw.update(runtime_kw)
    return call_kw


@register
class BoundOp(BaseOp):
    """Wrap a morphism, injecting kwargs from Principal.ctx via a param→ctx_key binding map."""

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
        inherit_contract(self, inner)

    async def pre(self, br: Principal, **kw) -> bool:
        call_kw = _build_call_kwargs(br, kw, self.bind, self.defaults)
        return bool(await maybe_await(self.inner.pre(br, **call_kw)))

    async def apply(self, br: Principal, **kw) -> dict[str, Any]:
        call_kw = _build_call_kwargs(br, kw, self.bind, self.defaults)
        return await maybe_await(self.inner.apply(br, **call_kw))

    async def post(self, br: Principal, result: dict[str, Any]) -> bool:
        return bool(await maybe_await(self.inner.post(br, result)))
