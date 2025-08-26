# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import anyio
from pydantic import Field, PrivateAttr

from lionagi.libs.concurrency import fail_after, get_cancelled_exc_class
from lionagi.protocols.types import Event, EventStatus

from ._types import AssosiatedEventInfo, HookEventTypes
from .hook_registry import HookRegistry


class HookEvent(Event):
    registry: HookRegistry = Field(..., exclude=True)
    hook_type: HookEventTypes
    exit: bool = Field(False, exclude=True)
    timeout: int = Field(30, exclude=True)
    params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    event_like: Event | type[Event] = Field(..., exclude=True)
    _should_exit: bool = PrivateAttr(False)
    _exit_cause: BaseException | None = PrivateAttr(None)

    assosiated_event_info: AssosiatedEventInfo | None = None

    async def invoke(self):
        start = anyio.current_time()
        self.execution.status = EventStatus.PROCESSING
        try:
            with fail_after(self.timeout):
                (res, se, st), meta = await self.registry.call(
                    self.event_like,
                    hook_type=self.hook_type,
                    exit=self.exit,
                    **self.params,
                )

                self.assosiated_event_info = AssosiatedEventInfo(**meta)
                self._should_exit = se
                self.execution.status = st
                if isinstance(res, tuple) and len(res) == 2:
                    self.execution.response = None
                    self.execution.error = str(res[1])
                    self._exit_cause = res[1]
                    raise res[1]
                if isinstance(res, Exception):
                    self.execution.response = None
                    self.execution.error = str(res)
                    self._exit_cause = res
                else:
                    self.execution.response = res
                    self.execution.error = None
        except get_cancelled_exc_class():
            raise

        except Exception as e:
            self.execution.status = EventStatus.FAILED
            self.execution.response = None
            self.execution.error = str(e)
            self._should_exit = True

        finally:
            self.execution.duration = anyio.current_time() - start
