# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from lionagi.protocols.action.tool import Tool

from ..base import LionTool


class ReaderAction(str, Enum):
    read = "read"
    list_dir = "list_dir"


class ReaderRequest(BaseModel):
    action: ReaderAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'read': Read a file and return its contents with line numbers.\n"
            "- 'list_dir': List files in a directory."
        ),
    )
    path: str | None = Field(
        None,
        description=(
            "File or directory path. Required for both 'read' and 'list_dir'."
        ),
    )
    offset: int | None = Field(
        None,
        description=(
            "Zero-indexed line number to start reading from. "
            "Only used for 'read'. Defaults to 0."
        ),
    )
    limit: int | None = Field(
        None,
        description=(
            "Maximum number of lines to return. "
            "Only used for 'read'. Defaults to 2000."
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


class ReaderTool(LionTool):
    is_lion_system_tool = True
    system_tool_name = "reader_tool"

    def __init__(self):
        self._tool = None

    def handle_request(self, request: ReaderRequest) -> ReaderResponse:
        if isinstance(request, dict):
            request = ReaderRequest(**request)
        if request.action == ReaderAction.read:
            return self._read(request.path, request.offset, request.limit)
        if request.action == ReaderAction.list_dir:
            return self._list_dir(request.path, request.recursive, request.file_types)
        return ReaderResponse(success=False, error="Unknown action")

    def _read(
        self,
        path: str | None,
        offset: int | None,
        limit: int | None,
    ) -> ReaderResponse:
        if not path:
            return ReaderResponse(success=False, error="'path' is required for action='read'")

        p = Path(path)
        if not p.exists():
            return ReaderResponse(success=False, error=f"File not found: {path}")
        if not p.is_file():
            return ReaderResponse(success=False, error=f"Path is not a file: {path}")

        # Binary detection: try reading a small chunk as bytes
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

    def _list_dir(
        self,
        path: str | None,
        recursive: bool | None,
        file_types: list[str] | None,
    ) -> ReaderResponse:
        if not path:
            return ReaderResponse(success=False, error="'path' is required for action='list_dir'")

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

    def to_tool(self) -> Tool:
        if self._tool is None:

            def reader_tool(**kwargs):
                """
                Read files or list directory contents.

                Use action='read' to get file contents with line numbers (supports
                offset and limit for large files). Use action='list_dir' to enumerate
                files in a directory. Binary files are rejected gracefully.
                """
                return self.handle_request(ReaderRequest(**kwargs)).model_dump()

            if self.system_tool_name != "reader_tool":
                reader_tool.__name__ = self.system_tool_name

            self._tool = Tool(
                func_callable=reader_tool,
                request_options=ReaderRequest,
            )
        return self._tool
