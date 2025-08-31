from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import Branch, Morphism
from ..ops import BaseOp


def _build_call_kwargs(
    br: Branch,
    runtime_kw: dict[str, Any],
    bind: Mapping[str, str],
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    # 1) from ctx via binding
    call_kw: dict[str, Any] = {param: br.ctx[src] for param, src in bind.items() if src in br.ctx}
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
        # Propagate basic attributes
        self.requires = getattr(inner, "requires", set())
        self.io = bool(getattr(inner, "io", False))
        # Propagate policy surface attributes for IPU invariant checking
        self.ctx_writes = getattr(inner, "ctx_writes", None)
        self.result_schema = getattr(inner, "result_schema", None)
        self.result_keys = getattr(inner, "result_keys", None)
        self.result_bytes_limit = getattr(inner, "result_bytes_limit", None)
        self.latency_budget_ms = getattr(inner, "latency_budget_ms", None)

    def required_rights(self, **kw) -> set[str]:
        """
        Delegate dynamic rights to the inner morphism after reconstructing the
        inner-call kwargs from (bind, defaults) and runtime kwargs.

        Note: Runner already merges Branch.ctx into **kw, so we can rebuild
        bound parameters without direct access to br.ctx here.
        """
        inner_req = getattr(self.inner, "required_rights", None)
        # If inner has no dynamic calculator, fall back to static declaration
        if not callable(inner_req):
            return set(getattr(self.inner, "requires", set()) or self.requires)

        # 1) start with defaults
        call_kw: dict[str, Any] = dict(self.defaults)
        # 2) fill from binding map using values present in runtime kw (ctx merged by Runner)
        for param, ctx_key in self.bind.items():
            if ctx_key in kw:
                call_kw[param] = kw[ctx_key]
        # 3) allow direct param override if caller provided param-named entries
        for param in list(self.bind.keys()):
            if param in kw:
                call_kw[param] = kw[param]
        # 4) pass through any additional params inner may care about
        for k, v in kw.items():
            call_kw.setdefault(k, v)

        try:
            reqs = inner_req(**call_kw)
            return set(reqs) | set(getattr(self.inner, "requires", set()))
        except Exception:
            return set(getattr(self.inner, "requires", set()) or self.requires)

    async def pre(self, br: Branch, **kw) -> bool:
        call_kw = _build_call_kwargs(br, kw, self.bind, self.defaults)
        return await self.inner.pre(br, **call_kw)

    async def apply(self, br: Branch, **kw) -> dict[str, Any]:
        call_kw = _build_call_kwargs(br, kw, self.bind, self.defaults)
        return await self.inner.apply(br, **call_kw)

    async def post(self, br: Branch, result: dict[str, Any]) -> bool:
        return await self.inner.post(br, result)
