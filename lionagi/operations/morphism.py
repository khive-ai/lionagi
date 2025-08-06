from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, ClassVar

from anyio import sleep
from typing_extensions import TypedDict

from lionagi.protocols._concepts import Invariant
from lionagi.session.branch import Branch
from lionagi.utils import is_coro_func

@dataclass(slots=True, frozen=True, init=False)
class MorphismMeta:
    name: str
    """name of the morphism"""

    version: str
    """version control"""

    description: str
    """Human readable description"""

    can_stream: bool
    """whether the morphism can be streamed"""

    stream_handler: Callable
    """stream handler function, if any, when stream is True"""


@dataclass(slots=True, frozen=True, init=False)
class MorphismContext(Invariant):
    params: dict[str, Any] | None = None
    stream_morphism: bool = False
    morphism_timeout: int | None = None  # default no limit

    def to_dict(self) -> dict[str, Any]:
        return self.params if self.params else {}


@dataclass(slots=True, frozen=True, init=False)
class Morphism(Invariant):
    """A transformation that occurs in a space"""

    meta: ClassVar[MorphismMeta]
    ctx: MorphismContext

    @abstractmethod
    @classmethod
    async def _apply(cls, branch, **kw) -> Any:
        """Apply the morphism to the branch."""

    async def _stream(self, branch: Branch, **kw) -> Any:
        """Stream the morphism."""
        raise NotImplementedError(
            "Streaming not implemented for this morphism"
        )

    async def apply(self, branch: Branch):
        """Apply the morphism."""
        return await self._apply(branch, **self.ctx.to_dict())

    async def stream(self, branch: Branch) -> AsyncGenerator:
        """Stream the morphism."""
        if not self.meta.get("can_stream", False):
            raise ValueError("This morphism does not support streaming.")

        handler = _get_stream_handler(self.meta)
        async for item in self._stream(branch, **self.ctx.to_dict()):
            handled = await handler(item)
            yield handled


def _get_stream_handler(meta: MorphismMeta):
    # 1. if handler is not None, return as a coroutine function
    if handler := meta.get("stream_handler"):
        if not is_coro_func(handler):

            async def _func(x):
                sleep(0)
                return handler(x)

            return _func
        return handler

    # 2. return a default handler that just returns the item
    async def _func(x):
        sleep(0)
        return x

    return _func
