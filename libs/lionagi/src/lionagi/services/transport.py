# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Transport protocol for HTTP IO boundary."""

from __future__ import annotations

from typing import Any, AsyncIterator, Mapping, Protocol

import httpx
import msgspec

from lionagi.services.core import NonRetryableError, RetryableError, TransportError


class Transport(Protocol):
    """IO boundary for HTTP requests.

    Thin and swappable (httpx/aiohttp/custom). All IO happens here.
    Transport implementations handle connection pooling, SSL, etc.
    """

    async def send_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict,
        timeout_s: float | None,
    ) -> dict:
        """Send JSON request and return parsed JSON response."""
        ...

    async def stream_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict,
        timeout_s: float | None,
    ) -> AsyncIterator[bytes]:
        """Send JSON request and yield response chunks as bytes."""
        ...


class HTTPXTransport:
    """HTTPX-based transport implementation.

    Provides connection pooling, SSL verification, and proper timeout handling.
    Maps HTTP errors to appropriate service exceptions.
    """

    def __init__(
        self,
        *,
        verify_ssl: bool = True,
        follow_redirects: bool = False,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ):
        self._client = httpx.AsyncClient(
            verify=verify_ssl,
            follow_redirects=follow_redirects,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def send_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict,
        timeout_s: float | None,
    ) -> dict:
        """Send JSON request and return parsed JSON response."""
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=timeout_s,
            )

            # Map HTTP status to service exceptions
            self._check_response_status(response)

            # Use msgspec for faster JSON parsing (v1 performance requirement)
            return msgspec.json.decode(response.content)

        except httpx.TimeoutException as e:
            raise TransportError(f"Request timed out: {e}")
        except httpx.NetworkError as e:
            raise RetryableError(f"Network error: {e}")
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except msgspec.DecodeError as e:
            raise TransportError(f"Invalid JSON response: {e}")

    async def stream_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict,
        timeout_s: float | None,
    ) -> AsyncIterator[bytes]:
        """Send JSON request and yield response chunks as bytes."""
        try:
            async with self._client.stream(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=timeout_s,
            ) as response:
                # Check status before starting to stream
                self._check_response_status(response)

                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if chunk:  # Skip empty chunks
                        yield chunk

        except httpx.TimeoutException as e:
            raise TransportError(f"Stream timed out: {e}")
        except httpx.NetworkError as e:
            raise RetryableError(f"Network error during streaming: {e}")
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)

    def _check_response_status(self, response: httpx.Response) -> None:
        """Check response status and raise appropriate exceptions."""
        if response.is_success:
            return

        if response.status_code == 429:
            # Rate limited - retryable
            raise RetryableError(f"Rate limited: {response.status_code} {response.text}")
        elif 500 <= response.status_code < 600:
            # Server error - retryable
            raise RetryableError(f"Server error: {response.status_code} {response.text}")
        elif 400 <= response.status_code < 500:
            # Client error - non-retryable (except 429 handled above)
            raise NonRetryableError(f"Client error: {response.status_code} {response.text}")
        else:
            # Other status codes
            raise TransportError(
                f"HTTP error: {response.status_code} {response.text}",
                status_code=response.status_code,
            )

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> None:
        """Handle HTTP status errors with proper classification."""
        response = error.response
        self._check_response_status(response)
