"""Core async primitives (thin wrappers over anyio)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import anyio

T = TypeVar("T")


class Lock:
    """Mutex for exclusive access (wraps anyio.Lock)."""

    __slots__ = ("_lock",)

    def __init__(self) -> None:
        self._lock = anyio.Lock()

    async def acquire(self) -> None:
        await self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    async def __aenter__(self) -> "Lock":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


class Semaphore:
    """Semaphore with the given initial value (wraps anyio.Semaphore)."""

    __slots__ = ("_sem",)

    def __init__(self, initial_value: int) -> None:
        if initial_value < 0:
            raise ValueError("initial_value must be >= 0")
        self._sem = anyio.Semaphore(initial_value)

    async def acquire(self) -> None:
        await self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> "Semaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


class CapacityLimiter:
    """Limiter restricting the number of concurrent acquires."""

    __slots__ = ("_lim",)

    def __init__(self, total_tokens: int) -> None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be >= 1")
        self._lim = anyio.CapacityLimiter(total_tokens)

    async def acquire(self) -> None:
        await self._lim.acquire()

    def release(self) -> None:
        self._lim.release()

    @property
    def remaining_tokens(self) -> float:
        return self._lim.remaining_tokens


@dataclass(slots=True)
class Queue(Generic[T]):
    """Typed queue over anyio MemoryObjectStream."""

    _send: anyio.abc.ObjectSendStream[T]
    _recv: anyio.abc.ObjectReceiveStream[T]

    @classmethod
    def with_maxsize(cls, maxsize: int) -> "Queue[T]":
        send, recv = anyio.create_memory_object_stream(maxsize)
        return cls(send, recv)

    async def put(self, item: T) -> None:
        await self._send.send(item)

    async def get(self) -> T:
        return await self._recv.receive()

    async def close(self) -> None:
        await self._send.aclose()
        await self._recv.aclose()

    @property
    def sender(self) -> anyio.abc.ObjectSendStream[T]:
        return self._send

    @property
    def receiver(self) -> anyio.abc.ObjectReceiveStream[T]:
        return self._recv
