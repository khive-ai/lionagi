from __future__ import annotations

from typing import Any, Protocol

from .types import Branch


class Morphism(Protocol):
    """Smallest executable unit; composable; explicit rights."""

    name: str
    requires: set[str]  # e.g., {"net.out"}, {"fs.read:/x/*"}

    async def pre(self, br: Branch, **kw) -> bool: ...
    async def apply(self, br: Branch, **kw) -> dict[str, Any]: ...
    async def post(self, br: Branch, result: dict[str, Any]) -> bool: ...
