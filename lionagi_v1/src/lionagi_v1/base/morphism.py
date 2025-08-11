from __future__ import annotations

from typing import Any, Dict, Protocol, Set

from .types import Branch


class Morphism(Protocol):
    """Smallest executable unit; composable; explicit rights."""

    name: str
    requires: Set[str]  # e.g., {"net.out"}, {"fs.read:/x/*"}

    async def pre(self, br: Branch, **kw) -> bool: ...
    async def apply(self, br: Branch, **kw) -> Dict[str, Any]: ...
    async def post(self, br: Branch, result: Dict[str, Any]) -> bool: ...
