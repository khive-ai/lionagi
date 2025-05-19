"""
Mock HTTP server for testing the HTTP transport.

This module provides a simple mock HTTP server for testing the HTTP transport
without making actual network requests.
"""

import asyncio
from typing import Any, Callable, Optional

from aiohttp import web


class MockHTTPServer:
    """Mock HTTP server for testing."""

    def __init__(self):
        """Initialize a new mock HTTP server."""
        self.app = web.Application()
        self.routes: dict[str, tuple[Any, int]] = {}
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.url: str = "http://localhost:8080"  # Default fallback URL
        self.base_url: str = "http://localhost:8080"  # Default fallback URL
        self.requests = []  # Store received requests for verification

    async def __aenter__(self) -> "MockHTTPServer":
        """Start the server.

        Returns:
            The server instance
        """
        # Create a new application for each context
        self.app = web.Application()

        # Set up routes
        for path, (handler, status_code) in self.routes.items():
            # Add a handler for both GET and POST methods
            self.app.router.add_get(path, self._create_handler(handler, status_code))
            self.app.router.add_post(path, self._create_handler(handler, status_code))

        # Set up streaming routes
        for path, handler in getattr(self, "streaming_routes", {}).items():
            self.app.router.add_get(path, handler)
            self.app.router.add_post(path, handler)

        # Start the server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "localhost", 0)
        await self.site.start()

        # Get the server URL
        server = self.site._server
        if server is not None and hasattr(server, "sockets") and server.sockets:
            socket = server.sockets[0]
            if socket is not None:
                # getsockname() might return more than 2 values, but we only need host and port
                sockname = socket.getsockname()
                # Always use localhost rather than 0.0.0.0 or other formats for better compatibility
                host = "localhost"
                port = sockname[1]  # Get port from the socket

                # Ensure port is a simple integer
                if isinstance(port, int) and port > 0:
                    self.url = f"http://{host}:{port}"
                    self.base_url = (
                        self.url
                    )  # Alias for compatibility with tests expecting base_url
                else:
                    # Fallback to a safe default if port is malformed
                    self.url = "http://localhost:8080"
                    self.base_url = self.url

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the server."""
        if self.runner is not None:
            await self.runner.cleanup()

    def add_route(self, path: str, handler: Any, status_code: int = 200) -> None:
        """Add a route to the server.

        Args:
            path: The URL path
            handler: The handler function or data
            status_code: The HTTP status code to return
        """
        self.routes[path] = (handler, status_code)

    def add_streaming_route(self, path: str, chunks: list[str]) -> None:
        """Add a streaming route to the server.

        Args:
            path: The URL path
            chunks: The chunks to stream
        """

        async def streaming_handler(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse()
            await response.prepare(request)
            for chunk in chunks:
                await response.write(chunk.encode("utf-8"))
                await asyncio.sleep(0.01)
            return response

        # Store streaming routes separately
        if not hasattr(self, "streaming_routes"):
            self.streaming_routes = {}
        self.streaming_routes[path] = streaming_handler

    def add_json_response(
        self, path: str, data: Any, status_code: int = 200, delay: float = 0
    ) -> None:
        """Add a JSON response to the server.

        Args:
            path: The URL path
            data: The JSON data to return
            status_code: The HTTP status code to return
            delay: Optional delay in seconds before responding
        """
        # Store the handler with delay if specified
        if delay > 0:

            async def delayed_handler():
                import asyncio

                await asyncio.sleep(delay)
                return data

            self.routes[path] = (delayed_handler, status_code)
        else:
            self.routes[path] = (data, status_code)

    def _create_handler(
        self, handler: Any, status_code: int
    ) -> Callable[[web.Request], web.Response]:
        """Create a request handler.

        Args:
            handler: The handler function or data
            status_code: The HTTP status code to return

        Returns:
            A request handler function
        """

        async def request_handler(request: web.Request) -> web.Response:
            # Log the request for debugging
            print(f"Mock server received request: {request.method} {request.path}")

            # Store request information for test verification
            self.requests.append(
                {
                    "method": request.method,
                    "path": request.path_qs,
                    "headers": dict(request.headers),
                }
            )

            # Get request body if any
            try:
                body = await request.json()
                self.requests[-1]["body"] = body
            except Exception:
                # Ignore errors in parsing the body
                pass

            # Process the request
            result = (
                await handler()
                if asyncio.iscoroutinefunction(handler)
                else handler()
                if callable(handler)
                else handler
            )

            # Create and return the response
            response = web.json_response(result, status=status_code)

            # Log the response
            print(f"Mock server sending response: {response.status}")

            return response

        return request_handler


# Alias for backward compatibility
MockHttpServer = MockHTTPServer
