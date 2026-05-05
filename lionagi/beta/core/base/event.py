# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, final

import orjson
from pydantic import Field, field_serializer, field_validator

from lionagi._errors import LionError, LionTimeoutError, ValidationError
from lionagi.beta.protocols import Invocable, Serializable, implements
from lionagi.ln.types._compat import ExceptionGroup
from lionagi.ln.types._sentinel import (
    MaybeSentinel,
    MaybeUnset,
    Unset,
    is_sentinel,
    is_unset,
)
from lionagi.ln.types import Enum
from lionagi.ln._json_dump import json_dumpb
from lionagi.ln._utils import async_synchronized
from lionagi.ln.concurrency import (
    Lock,
    current_time,
    fail_after,
    get_cancelled_exc_class,
)

from .element import LN_ELEMENT_FIELDS, Element

__all__ = (
    "Event",
    "EventStatus",
    "Execution",
)


class EventStatus(Enum):
    """Execution status states for events."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    ABORTED = "aborted"


@implements(Serializable)
@dataclass(slots=True)
class Execution:
    """Mutable execution state snapshot for an in-progress or completed event."""

    status: EventStatus = EventStatus.PENDING
    duration: MaybeUnset[float] = Unset
    response: MaybeSentinel[Any] = Unset
    error: MaybeUnset[BaseException] | None = Unset
    retryable: MaybeUnset[bool] = Unset

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        res_ = Unset
        if is_sentinel(self.response):
            res_ = None
        else:
            with contextlib.suppress(orjson.JSONDecodeError, TypeError):
                res_ = json_dumpb(self.response)
            if is_unset(res_):
                res_ = "<unserializable>"

        error_dict = None
        if not is_unset(self.error) and self.error is not None:
            if isinstance(self.error, Serializable):
                error_dict = self.error.to_dict()
            elif isinstance(self.error, ExceptionGroup):
                error_dict = self._serialize_exception_group(self.error)
            else:
                error_dict = {
                    "error": type(self.error).__name__,
                    "message": str(self.error),
                }

        duration_value = None if is_unset(self.duration) else self.duration
        retryable_value = None if is_unset(self.retryable) else self.retryable

        return {
            "status": self.status.value,
            "duration": duration_value,
            "response": res_,
            "error": error_dict,
            "retryable": retryable_value,
        }

    def _serialize_exception_group(
        self,
        eg: ExceptionGroup,
        depth: int = 0,
        _seen: set[int] | None = None,
    ) -> dict[str, Any]:
        max_depth = 100
        if depth > max_depth:
            return {
                "error": "ExceptionGroup",
                "message": f"Max nesting depth ({max_depth}) exceeded",
                "nested_count": len(eg.exceptions) if hasattr(eg, "exceptions") else 0,
            }

        if _seen is None:
            _seen = set()

        eg_id = id(eg)
        if eg_id in _seen:
            return {
                "error": "ExceptionGroup",
                "message": "Circular reference detected",
            }

        _seen.add(eg_id)

        try:
            exceptions = []
            for exc in eg.exceptions:
                if isinstance(exc, Serializable):
                    exceptions.append(exc.to_dict())
                elif isinstance(exc, ExceptionGroup):
                    exceptions.append(
                        self._serialize_exception_group(exc, depth + 1, _seen)
                    )
                else:
                    exceptions.append(
                        {
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    )

            return {
                "error": type(eg).__name__,
                "message": str(eg),
                "exceptions": exceptions,
            }
        finally:
            _seen.discard(eg_id)

    def add_error(self, exc: BaseException) -> None:
        if is_unset(self.error) or self.error is None:
            self.error = exc
        elif isinstance(self.error, ExceptionGroup):
            self.error = ExceptionGroup(  # type: ignore[type-var]
                "multiple errors",
                [*self.error.exceptions, exc],
            )
        else:
            self.error = ExceptionGroup(  # type: ignore[type-var]
                "multiple errors",
                [self.error, exc],
            )


@implements(Invocable)
class Event(Element):
    """Base event with lifecycle tracking; subclasses implement _invoke()."""

    execution: Execution = Field(default_factory=Execution)
    timeout: MaybeUnset[float] = Field(Unset, exclude=True)
    streaming: bool = Field(False, exclude=True)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._async_lock = Lock()

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, v: Any) -> MaybeUnset[float]:
        if is_sentinel(v, additions={"none", "empty"}):
            return Unset
        if not math.isfinite(v):
            raise ValueError(f"timeout must be finite, got {v}")
        if v <= 0:
            raise ValueError(f"timeout must be positive, got {v}")
        return v

    @field_serializer("execution")
    def _serialize_execution(self, val: Execution) -> dict:
        return val.to_dict()

    @property
    def request(self) -> dict:
        return {}

    async def _invoke(self) -> Any:
        """Subclasses must override to provide event execution logic."""
        raise NotImplementedError("Subclasses must implement _invoke()")

    async def _stream(self):
        """Subclasses override to yield incremental chunks; stream() manages lifecycle."""
        raise NotImplementedError("Subclasses must implement _stream()")

    @final
    @async_synchronized
    async def invoke(self) -> None:
        """Execute with lifecycle management. No-op if not PENDING; thread-safe."""
        if self.execution.status != EventStatus.PENDING:
            return

        start = current_time()

        try:
            self.execution.status = EventStatus.PROCESSING

            if not is_unset(self.timeout):
                with fail_after(self.timeout):
                    result = await self._invoke()
            else:
                result = await self._invoke()

            self.execution.response = result
            self.execution.error = None
            self.execution.status = EventStatus.COMPLETED
            self.execution.retryable = False

        except TimeoutError:
            lion_timeout = LionTimeoutError(
                f"Operation timed out after {self.timeout}s",
                retryable=True,
            )

            self.execution.response = Unset
            self.execution.error = lion_timeout
            self.execution.status = EventStatus.CANCELLED
            self.execution.retryable = lion_timeout.retryable

        except Exception as e:
            if isinstance(e, ExceptionGroup):
                retryable = all(
                    not isinstance(exc, LionError) or exc.retryable
                    for exc in e.exceptions
                )
                self.execution.retryable = retryable
            else:
                self.execution.retryable = (
                    e.retryable if isinstance(e, LionError) else True
                )

            self.execution.response = Unset
            self.execution.error = e
            self.execution.status = EventStatus.FAILED

        except BaseException as e:
            if isinstance(e, get_cancelled_exc_class()):
                self.execution.response = Unset
                self.execution.error = e
                self.execution.status = EventStatus.CANCELLED
                self.execution.retryable = True

            raise

        finally:
            self.execution.duration = current_time() - start

    @final
    @contextlib.asynccontextmanager
    async def stream(self):
        """Stream with lifecycle management; context manager guarantees cleanup (no aclosing() needed)."""
        async with self._async_lock:
            if self.execution.status != EventStatus.PENDING:

                async def _empty():
                    return
                    yield  # makes this an async generator (unreachable)

                yield _empty()
                return
            self.execution.status = EventStatus.PROCESSING

        start = current_time()
        chunks: list = []
        _done = False

        async def _iter():
            nonlocal _done
            async with contextlib.aclosing(self._stream()) as agen:
                if not is_unset(self.timeout):
                    with fail_after(self.timeout):
                        async for chunk in agen:
                            chunks.append(chunk)
                            yield chunk
                else:
                    async for chunk in agen:
                        chunks.append(chunk)
                        yield chunk
            _done = True

        inner = _iter()
        _exc: BaseException | None = None

        try:
            yield inner
        except Exception as e:
            _exc = e
            raise
        except BaseException as e:
            _exc = e
            raise
        finally:
            with contextlib.suppress(BaseException):
                await inner.aclose()

            self.execution.response = chunks
            self.execution.duration = current_time() - start

            if _exc is None:
                if _done:
                    self.execution.error = None
                    self.execution.status = EventStatus.COMPLETED
                    self.execution.retryable = False
                else:
                    self.execution.error = None
                    self.execution.status = EventStatus.CANCELLED
                    self.execution.retryable = False
            elif isinstance(_exc, TimeoutError):
                self.execution.error = LionTimeoutError(
                    f"Stream timed out after {self.timeout}s",
                    retryable=True,
                )
                self.execution.status = EventStatus.CANCELLED
                self.execution.retryable = True
            elif isinstance(_exc, Exception):
                if isinstance(_exc, ExceptionGroup):
                    self.execution.retryable = all(
                        not isinstance(e, LionError) or e.retryable
                        for e in _exc.exceptions
                    )
                else:
                    self.execution.retryable = (
                        _exc.retryable if isinstance(_exc, LionError) else True
                    )
                self.execution.error = _exc
                self.execution.status = EventStatus.FAILED
            else:
                self.execution.error = _exc
                if isinstance(_exc, get_cancelled_exc_class()):
                    self.execution.status = EventStatus.CANCELLED
                    self.execution.retryable = True
                else:
                    self.execution.status = EventStatus.FAILED
                    self.execution.retryable = False

    def as_fresh_event(self, copy_meta: bool = False) -> Event:
        """Clone with reset execution state; original id/created_at stored in metadata["original"]."""
        d_ = self.to_dict(exclude={"execution", *LN_ELEMENT_FIELDS})
        fresh = self.__class__(**d_)

        if not is_sentinel(self.timeout):
            fresh.timeout = self.timeout

        if copy_meta:
            fresh.metadata = self.metadata.copy()

        fresh.metadata["original"] = {
            "id": str(self.id),
            "created_at": self.created_at,
        }
        return fresh

    def assert_completed(self, *, retryable: MaybeUnset[bool] = Unset):
        if self.execution.status != EventStatus.COMPLETED:
            retryable_value = (
                self.execution.retryable if is_unset(retryable) else retryable
            )
            retryable_value = retryable_value is True
            exec_dict = self.execution.to_dict()
            exec_dict.pop("response", None)
            exec_dict.pop("retryable", None)

            raise ValidationError(
                "Event did not complete successfully.",
                details={
                    "event_id": str(self.id),
                    **exec_dict,
                },
                retryable=retryable_value,
            )
