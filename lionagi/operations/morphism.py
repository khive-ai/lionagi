from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, ClassVar

from anyio import sleep
from lionagi.utils import is_coro_func
from lionagi.protocols._concepts import Invariant, Communicatable


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

    async def _stream(self, co_: Communicatable, **kw) -> Any:
        """Stream the morphism."""
        raise NotImplementedError("Streaming not implemented for this morphism")

    async def apply(self, co_: Communicatable):
        """Apply the morphism."""
        return await self._apply(co_, **self.ctx.to_dict())

    async def stream(self, co_: Communicatable) -> AsyncGenerator:
        """Stream the morphism."""
        if not self.meta.can_stream:
            raise ValueError("This morphism does not support streaming.")

        handler = _get_stream_handler(self.meta)
        async for item in self._stream(co_, **self.ctx.to_dict()):
            handled = await handler(item)
            yield handled


def _get_stream_handler(meta: MorphismMeta) -> Callable:

    if handler := meta.stream_handler:
        # handle non-coroutine functions
        if not is_coro_func(handler):

            async def _func(x):
                await sleep(0)
                return handler(x)
            return _func

        # directly return coroutine functions
        return handler

    # return a default handler that just returns the item
    async def _func(x):
        await sleep(0)
        return x

    return _func
