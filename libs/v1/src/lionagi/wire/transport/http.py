"""Pure HTTP transport - no SDK dependencies.

Ocean: "in v0, we deliberately avoided openai sdk, and opt-in for http only"

Pure HTTP gives complete control and superior debugging.
"""

from typing import Any, Optional, Dict
import httpx
import msgspec

from ..dto.base import Request, Response


class HTTPTransport:
    """Pure async HTTP with connection pooling.
    
    No SDK dependencies - just clean HTTP.
    Complete control over transport layer.
    """
    
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def request(self, 
                     method: str,
                     url: str, 
                     headers: Dict[str, str] = None,
                     json: Any = None,
                     **kwargs) -> httpx.Response:
        """Make HTTP request"""
        if self.base_url and not url.startswith('http'):
            url = f"{self.base_url}/{url}"
        
        return await self.client.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            **kwargs
        )
    
    async def post(self, url: str, data: Request) -> Response:
        """POST request with DTO (msgspec → JSON)"""
        builtins = msgspec.to_builtins(data)
        full_url = f"{self.base_url}/{url}" if self.base_url and not url.startswith("http") else url
        resp = await self.client.post(full_url, json=builtins)
        resp.raise_for_status()
        result = resp.json()
        # Best effort conversion back to DTO
        return Response(id=builtins.get("id"), result=result, error=None)
    
    async def stream(self, url: str, data: Request):
        """SSE streaming support (POST). Yields text chunks."""
        builtins = msgspec.to_builtins(data)
        full_url = f"{self.base_url}/{url}" if self.base_url and not url.startswith("http") else url
        async with self.client.stream("POST", full_url, json=builtins) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    yield chunk

    async def post_json(self, url: str, json: Dict[str, Any], headers: Dict[str, str] | None = None) -> Dict[str, Any]:
        """POST with raw JSON payload and return JSON response."""
        full_url = f"{self.base_url}/{url}" if self.base_url and not url.startswith("http") else url
        resp = await self.client.post(full_url, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()
    
    async def close(self):
        """Clean up client"""
        await self.client.aclose()
