# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from lionagi.ln.concurrency import run_sync
from lionagi.protocols.action.tool import Tool

from ..base import LionTool


class ReaderAction(str, Enum):
    read = "read"
    open = "open"
    list_dir = "list_dir"


class ReaderRequest(BaseModel):
    action: ReaderAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'read': Read a text file with line numbers (lightweight, no conversion).\n"
            "- 'open': Convert a document (PDF, PPTX, DOCX, HTML, URL) to text via docling. "
            "Result is cached by path — subsequent reads use offset/limit on the cached text.\n"
            "- 'list_dir': List files in a directory."
        ),
    )
    path: str | None = Field(
        None,
        description=(
            "File path, directory path, or URL. Required for all actions."
        ),
    )
    offset: int | None = Field(
        None,
        description=(
            "Zero-indexed line number to start reading from. "
            "Used for 'read' and for reading cached 'open' results. Defaults to 0."
        ),
    )
    limit: int | None = Field(
        None,
        description=(
            "Maximum number of lines to return. "
            "Used for 'read' and cached reads. Defaults to 2000."
        ),
    )
    recursive: bool | None = Field(
        None,
        description=(
            "Whether to list files recursively in subdirectories. "
            "Only used for 'list_dir'. Defaults to False."
        ),
    )
    file_types: list[str] | None = Field(
        None,
        description=(
            "Filter by file extensions (e.g. ['.py', '.txt']). "
            "Only used for 'list_dir'. If omitted, all files are listed."
        ),
    )


class ReaderResponse(BaseModel):
    success: bool = Field(
        ...,
        description="True if the action completed without error.",
    )
    content: str | None = Field(
        None,
        description="The file content (for 'read') or path listing (for 'list_dir').",
    )
    error: str | None = Field(
        None,
        description="Error message when success=False.",
    )


def _read_sync(
    path: str,
    offset: int | None,
    limit: int | None,
) -> ReaderResponse:
    p = Path(path)
    if not p.exists():
        return ReaderResponse(success=False, error=f"File not found: {path}")
    if not p.is_file():
        return ReaderResponse(success=False, error=f"Path is not a file: {path}")

    try:
        with open(p, "rb") as fbin:
            chunk = fbin.read(8192)
        if b"\x00" in chunk:
            return ReaderResponse(success=False, error=f"Binary file not supported: {path}")
    except OSError as e:
        return ReaderResponse(success=False, error=f"Cannot open file: {e}")

    start = max(0, offset or 0)
    max_lines = limit if (limit is not None and limit > 0) else 2000

    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return ReaderResponse(success=False, error=f"Read error: {e}")

    selected = lines[start : start + max_lines]
    numbered = "".join(
        f"{start + i + 1}\t{line}" for i, line in enumerate(selected)
    )
    return ReaderResponse(success=True, content=numbered)


def _list_dir_sync(
    path: str,
    recursive: bool | None,
    file_types: list[str] | None,
) -> ReaderResponse:
    from lionagi.libs.file.process import dir_to_files

    try:
        files = dir_to_files(
            path,
            recursive=bool(recursive),
            file_types=file_types,
        )
        content = "\n".join(str(f) for f in files)
    except Exception as e:
        return ReaderResponse(success=False, error=f"List error: {e}")

    return ReaderResponse(success=True, content=content)


import time

_CACHE_TTL_SECONDS = 300  # 5 minutes


def _open_sync(path: str, cache: dict[str, tuple[str, float]]) -> ReaderResponse:
    """Convert document via docling, cache result keyed by path."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return ReaderResponse(
            success=False,
            error="docling not installed. Run: pip install lionagi[reader]",
        )

    try:
        converter = DocumentConverter()
        result = converter.convert(path)
        text = result.document.export_to_markdown()
    except Exception as e:
        return ReaderResponse(success=False, error=f"Conversion error: {e}")

    cache[path] = (text, time.time())
    lines = text.split("\n")
    return ReaderResponse(
        success=True,
        content=f"Opened: {path} ({len(lines)} lines, {len(text)} chars). Use read with offset/limit to view.",
    )


def _read_cached(path: str, offset: int, limit: int, cache: dict[str, tuple[str, float]]) -> ReaderResponse | None:
    """Read from cache if path was previously opened and not expired."""
    if path not in cache:
        return None
    text, cached_at = cache[path]
    if time.time() - cached_at > _CACHE_TTL_SECONDS:
        del cache[path]
        return None
    lines = text.split("\n")
    selected = lines[offset : offset + limit]
    numbered = "".join(f"{offset + i + 1}\t{line}\n" for i, line in enumerate(selected))
    return ReaderResponse(success=True, content=numbered)


def _evict_expired(cache: dict[str, tuple[str, float]]) -> int:
    """Remove expired entries. Returns count evicted."""
    now = time.time()
    expired = [k for k, (_, t) in cache.items() if now - t > _CACHE_TTL_SECONDS]
    for k in expired:
        del cache[k]
    return len(expired)


class ReaderTool(LionTool):
    is_lion_system_tool = True
    system_tool_name = "reader_tool"

    def __init__(self, cache_ttl: int = _CACHE_TTL_SECONDS):
        self._tool = None
        self._cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = cache_ttl

    async def handle_request(self, request: ReaderRequest) -> ReaderResponse:
        if isinstance(request, dict):
            request = ReaderRequest(**request)
        if not request.path:
            return ReaderResponse(success=False, error="'path' is required")

        _evict_expired(self._cache)

        if request.action == ReaderAction.open:
            return await run_sync(_open_sync, request.path, self._cache)

        if request.action == ReaderAction.read:
            start = max(0, request.offset or 0)
            limit = request.limit if (request.limit and request.limit > 0) else 2000
            cached = _read_cached(request.path, start, limit, self._cache)
            if cached is not None:
                return cached
            return await run_sync(_read_sync, request.path, request.offset, request.limit)

        if request.action == ReaderAction.list_dir:
            return await run_sync(_list_dir_sync, request.path, request.recursive, request.file_types)

        return ReaderResponse(success=False, error="Unknown action")

    def to_tool(self) -> Tool:
        if self._tool is None:

            async def reader_tool(**kwargs):
                """Read files, convert documents (PDF/PPTX/DOCX/URL via docling), or list directories.

                Use action='read' for text files (lightweight, line numbers).
                Use action='open' for documents needing conversion (PDF, PPTX, HTML, URL) —
                result is cached by path for 5 minutes, then use 'read' with offset/limit.
                Use action='list_dir' for directory listings.
                """
                return (await self.handle_request(ReaderRequest(**kwargs))).model_dump()

            if self.system_tool_name != "reader_tool":
                reader_tool.__name__ = self.system_tool_name

            self._tool = Tool(
                func_callable=reader_tool,
                request_options=ReaderRequest,
            )
        return self._tool
