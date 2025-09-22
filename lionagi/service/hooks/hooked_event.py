# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import anyio
from pydantic import PrivateAttr

from lionagi.ln import get_cancelled_exc_class
from lionagi.protocols.types import DataLogger, Event, EventStatus, Log
from lionagi.service.hooks import HookEvent, HookEventTypes

global_hook_logger = DataLogger(
    persist_dir="./data/logs",
    subfolder="hooks",
    file_prefix="hook",
    capacity=100,
)


class HookedEvent(Event):
    """Handles asynchronous API calls with automatic token usage tracking.

    This class manages API calls through endpoints, handling both regular
    and streaming responses with optional token usage tracking.
    """

    _pre_invoke_hook_event: HookEvent = PrivateAttr(None)
    _post_invoke_hook_event: HookEvent = PrivateAttr(None)

    async def _stream(self):
        raise NotImplementedError

    async def _invoke(self):
        raise NotImplementedError

    async def invoke(self) -> None:
        """Execute the API call through the endpoint.

        Updates execution status and stores the response or error.
        """
        start = anyio.current_time()

        try:
            self.execution.status = EventStatus.PROCESSING
            if h_ev := self._pre_invoke_hook_event:
                await h_ev.invoke()
                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Pre-invocation hook requested exit without a cause"
                    )
                await global_hook_logger.alog(Log.create(h_ev))

            response = await self._invoke()

            if h_ev := self._post_invoke_hook_event:
                await h_ev.invoke()
                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Post-invocation hook requested exit without a cause"
                    )
                await global_hook_logger.alog(Log.create(h_ev))

            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except get_cancelled_exc_class():
            self.execution.error = "Invocation cancelled"
            self.execution.status = EventStatus.CANCELLED
            raise

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED

        finally:
            self.execution.duration = anyio.current_time() - start

    async def stream(self):
        """Stream the API response through the endpoint.

        Yields:
            Streaming chunks from the API.
        """
        start = anyio.current_time()

        response = []

        try:
            self.execution.status = EventStatus.PROCESSING

            async for chunk in self._stream():
                response.append(chunk)
                yield chunk

            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except get_cancelled_exc_class():
            self.execution.error = "Streaming cancelled"
            self.execution.status = EventStatus.CANCELLED
            raise

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED

        finally:
            self.execution.duration = anyio.current_time() - start

    def create_pre_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool = None,
        hook_timeout: float = 30.0,
        hook_params: dict = None,
    ):
        h_ev = HookEvent(
            hook_type=HookEventTypes.PreInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._pre_invoke_hook_event = h_ev

    def create_post_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool = None,
        hook_timeout: float = 30.0,
        hook_params: dict = None,
    ):
        h_ev = HookEvent(
            hook_type=HookEventTypes.PostInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._post_invoke_hook_event = h_ev
