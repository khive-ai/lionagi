# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import msgspec

from ..core import CallContext, Service
from ..endpoint import RequestModel
from ..providers.provider_registry import ProviderAdapter
from ..transport import HTTPXTransport


def _host_rights(url: str | None) -> set[str]:
    host = urlparse(url or "").netloc or "*"
    return {f"net.out:{host}" if host else "net.out:*"}


class HttpDescriptor(msgspec.Struct, kw_only=True):
    """Declarative HTTP shape for generic JSON calls."""

    method: str = "POST"
    path: str = "/"
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    query: dict[str, str] = msgspec.field(default_factory=dict)


class GenericRequestModel(RequestModel):
    """Msgspec request for generic JSON services."""

    payload: Any = None
    http: HttpDescriptor | None = None  # If not set, adapter must provide one


class GenericJSONService(Service[GenericRequestModel, dict, bytes]):
    """Minimal, fast generic JSON service using HTTPXTransport."""

    def __init__(self, *, name: str, base_url: str, http: HttpDescriptor | None = None):
        self.name = name
        self.base_url = base_url
        self._http_default = http  # optional; request may override

        # Set default conservative rights; adapter overrides in registry
        self.requires = _host_rights(base_url)

    async def call(self, req: GenericRequestModel, *, ctx: CallContext) -> dict:
        http = req.http or self._http_default
        if not http:
            raise ValueError(
                "GenericJSONService: HttpDescriptor must be provided (adapter or request)"
            )

        url = urljoin(self.base_url.rstrip("/") + "/", http.path.lstrip("/"))
        if http.query:
            url = f"{url}?{urlencode(http.query)}"

        timeout_s = ctx.remaining_time

        async with HTTPXTransport() as tx:
            return await tx.send_json(
                method=http.method,
                url=url,
                headers=http.headers,
                json=(req.payload if req.payload is not None else {}),
                timeout_s=timeout_s,
            )

    async def stream(self, req: GenericRequestModel, *, ctx: CallContext) -> AsyncIterator[bytes]:
        http = req.http or self._http_default
        if not http:
            raise ValueError(
                "GenericJSONService: HttpDescriptor must be provided (adapter or request)"
            )

        url = urljoin(self.base_url.rstrip("/") + "/", http.path.lstrip("/"))
        if http.query:
            url = f"{url}?{urlencode(http.query)}"

        timeout_s = ctx.remaining_time

        async with HTTPXTransport() as tx:
            async for chunk in tx.stream_json(
                method=http.method,
                url=url,
                headers=http.headers,
                json=(req.payload if req.payload is not None else {}),
                timeout_s=timeout_s,
            ):
                # pass-through; do not buffer
                yield chunk


class GenericJSONAdapter(ProviderAdapter):
    """Adapter for ANY HTTP/JSON provider. This is the universal escape hatch."""

    name = "generic"
    default_base_url = None
    request_model = GenericRequestModel
    requires: set[str] | None = None  # computed per base_url
    ConfigModel = None  # optional Pydantic config could be added by users

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        # If a base_url is supplied, we can always support it.
        return bool(base_url)

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        if not base_url:
            raise ValueError("generic adapter requires base_url")
        # Optionally accept an HttpDescriptor default at adapter level
        http = kwargs.get("http", None)
        return GenericJSONService(name="generic", base_url=base_url, http=http)

    def required_rights(self, *, base_url: str | None, **_: Any) -> set[str]:
        return _host_rights(base_url)
