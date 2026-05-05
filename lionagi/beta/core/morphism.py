# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Morphism: the atomic executable unit in lionagi.beta."""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

import msgspec

__all__ = ("Morphism", "MorphismLike", "MorphismAdapter")


@runtime_checkable
class MorphismLike(Protocol):
    """Structural protocol for foreign executables that cannot inherit Morphism."""

    name: str
    requires: frozenset[str]

    async def apply(self, br: Any, **kw: Any) -> dict[str, Any]: ...


class Morphism(msgspec.Struct, kw_only=True):
    """Atomic executable unit with explicit capability declarations."""

    name: str
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    required_rights: Any = None

    async def pre(self, br: Any, **kw: Any) -> bool:
        return True

    async def apply(self, br: Any, **kw: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"Morphism subclass '{self.name}' must implement apply()"
        )

    async def post(self, br: Any, result: dict[str, Any]) -> bool:
        return True


class MorphismAdapter(Morphism, kw_only=True):
    """Wraps an arbitrary async callable as a Morphism for Runner compatibility."""

    name: str = "adapter"
    _fn: Any = None

    @classmethod
    def wrap(
        cls,
        *args: Any,
        name: str | None = None,
        fn: Any | None = None,
        requires: frozenset[str] | set[str] = frozenset(),
        provides: frozenset[str] | set[str] = frozenset(),
        required_rights: Any = None,
    ) -> MorphismAdapter:
        if args:
            if callable(args[0]):
                fn = args[0]
                if len(args) > 1:
                    raise TypeError(
                        "wrap(fn, ...) accepts only one positional argument"
                    )
                name = name or getattr(fn, "__name__", "adapter")
            else:
                if len(args) != 2:
                    raise TypeError(
                        "wrap(name, fn, ...) requires exactly two positional arguments"
                    )
                name = str(args[0])
                fn = args[1]

        if fn is None:
            raise TypeError("wrap() missing required callable")
        if name is None:
            name = getattr(fn, "__name__", "adapter")

        return cls(
            name=name,
            requires=frozenset(requires),
            provides=frozenset(provides),
            required_rights=required_rights,
            _fn=fn,
        )

    @classmethod
    def from_protocol(cls, obj: MorphismLike) -> MorphismAdapter:
        return cls(
            name=obj.name,
            requires=frozenset(obj.requires),
            provides=frozenset(getattr(obj, "provides", frozenset())),
            required_rights=getattr(obj, "required_rights", None),
            _fn=obj.apply,
        )

    @staticmethod
    def from_operation(
        decl: Any,
        params: Any,
        operation: Any | None = None,
    ) -> MorphismAdapter:

        def _required_rights(**kw: Any) -> set[str] | None:
            resolver = getattr(decl, "required_rights", None)
            if not callable(resolver):
                return None
            rights = resolver(
                params,
                operation=operation or kw.get("_lionagi_operation"),
                session=kw.get("_lionagi_session"),
                branch=kw.get("_lionagi_branch"),
                kwargs=kw,
            )
            return set(rights) if rights else None

        async def _apply(br: Any, **kw: Any) -> Any:
            from lionagi.beta.session.context import RequestContext

            session = kw.get("_lionagi_session")
            branch = kw.get("_lionagi_branch")
            op = operation or kw.get("_lionagi_operation")
            if op is not None and session is not None and branch is not None:
                ctx = op.make_context(
                    session,
                    branch,
                    verbose=kw.get("_lionagi_verbose"),
                    principal=br,
                )
            else:
                branch_ref = kw.get("_lionagi_branch_ref")
                if branch_ref is None and branch is not None:
                    branch_ref = getattr(branch, "name", None) or str(
                        getattr(branch, "id", "")
                    )

                metadata: dict[str, Any] = {"principal": br}
                if session is not None:
                    metadata["_bound_session"] = session
                if branch is not None:
                    metadata["_bound_branch"] = branch
                if "_lionagi_verbose" in kw:
                    metadata["_verbose"] = kw["_lionagi_verbose"]

                ctx = RequestContext(
                    name=kw.get("_lionagi_operation_type", decl.handler.__name__),
                    session_id=getattr(session, "id", None),
                    branch=branch_ref,
                    **metadata,
                )
            result_or_stream = decl.handler(params, ctx)
            if hasattr(result_or_stream, "__aiter__"):
                result = [item async for item in result_or_stream]
            elif inspect.isawaitable(result_or_stream):
                result = await result_or_stream
            else:
                result = result_or_stream
            if isinstance(result, dict) and "action" in result:
                return result
            return {"result": result}

        return MorphismAdapter.wrap(
            _apply,
            name=f"ops.{decl.handler.__name__}",
            requires=frozenset(getattr(decl, "requires", frozenset())),
            provides=frozenset(getattr(decl, "provides", frozenset())),
            required_rights=(
                _required_rights
                if callable(getattr(decl, "required_rights", None))
                else None
            ),
        )

    async def apply(self, br: Any, **kw: Any) -> dict[str, Any]:
        if self._fn is None:
            raise RuntimeError("MorphismAdapter has no wrapped function")
        result = self._fn(br, **kw)
        if inspect.isawaitable(result):
            return await result
        return result
