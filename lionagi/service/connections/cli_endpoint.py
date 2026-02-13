# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from .endpoint import Endpoint
from .endpoint_config import EndpointConfig


class CLIEndpoint(Endpoint):
    """Base for CLI-based agentic endpoints (subprocess, not HTTP).

    CLI endpoints (Claude Code, Gemini CLI, Codex CLI) differ from HTTP API
    endpoints in fundamental ways:
    - They spawn heavy subprocesses, so concurrency must be low.
    - They may maintain sessions with resumption.
    - They manage their own context and actions.
    - They use subprocess + NDJSON streaming, not aiohttp + JSON.
    """

    is_cli: ClassVar[bool] = True

    # Conservative defaults for CLI subprocess concurrency
    DEFAULT_CONCURRENCY_LIMIT: ClassVar[int] = 3
    DEFAULT_QUEUE_CAPACITY: ClassVar[int] = 10

    def __init__(self, config: dict | EndpointConfig = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        """Current session ID for resume, if any."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None):
        self._session_id = value

    def _create_http_session(self):
        raise NotImplementedError("CLI endpoints do not use HTTP sessions")

    async def _call_aiohttp(self, *a, **kw):
        raise NotImplementedError("CLI endpoints do not use aiohttp")

    async def _stream_aiohttp(self, payload: dict, headers: dict, **kwargs):
        raise NotImplementedError("CLI endpoints do not use aiohttp streaming")
