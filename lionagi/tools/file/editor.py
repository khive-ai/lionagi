# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from lionagi.protocols.action.tool import Tool

from ..base import LionTool


class EditorAction(str, Enum):
    write = "write"
    edit = "edit"


class EditorRequest(BaseModel):
    action: EditorAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'write': Write (or overwrite) a file with the given content. "
            "Creates parent directories if they do not exist.\n"
            "- 'edit': Replace an exact string in a file. Fails if the old_string "
            "is not found, or if it appears multiple times and replace_all=False."
        ),
    )
    file_path: str = Field(
        ...,
        description="Absolute or relative path to the target file.",
    )
    content: str | None = Field(
        None,
        description=(
            "Full content to write to the file. Required when action='write'."
        ),
    )
    old_string: str | None = Field(
        None,
        description=(
            "Exact string to find and replace. Required when action='edit'. "
            "Must match the file contents byte-for-byte, including whitespace and indentation."
        ),
    )
    new_string: str | None = Field(
        None,
        description=(
            "Replacement string. Required when action='edit'. "
            "May be an empty string to delete the matched region."
        ),
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "When True, replace every occurrence of old_string. "
            "When False (default), the edit fails if old_string appears more than once."
        ),
    )


class EditorResponse(BaseModel):
    success: bool = Field(
        ...,
        description="True if the action completed without error.",
    )
    content: str | None = Field(
        None,
        description=(
            "For 'write': confirmation message with the path. "
            "For 'edit': a short snippet of the edited region for confirmation."
        ),
    )
    error: str | None = Field(
        None,
        description="Error message when success=False.",
    )


class EditorTool(LionTool):
    is_lion_system_tool = True
    system_tool_name = "editor_tool"

    def __init__(self):
        self._tool = None

    def handle_request(self, request: EditorRequest) -> EditorResponse:
        if isinstance(request, dict):
            request = EditorRequest(**request)
        if request.action == EditorAction.write:
            return self._write(request.file_path, request.content)
        if request.action == EditorAction.edit:
            return self._edit(
                request.file_path,
                request.old_string,
                request.new_string,
                request.replace_all,
            )
        return EditorResponse(success=False, error="Unknown action")

    def _write(self, file_path: str, content: str | None) -> EditorResponse:
        if content is None:
            return EditorResponse(success=False, error="'content' is required for action='write'")

        p = Path(file_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return EditorResponse(success=False, error=f"Write error: {e}")

        return EditorResponse(success=True, content=f"Written: {p}")

    def _edit(
        self,
        file_path: str,
        old_string: str | None,
        new_string: str | None,
        replace_all: bool,
    ) -> EditorResponse:
        if old_string is None:
            return EditorResponse(success=False, error="'old_string' is required for action='edit'")
        if new_string is None:
            return EditorResponse(success=False, error="'new_string' is required for action='edit'")

        p = Path(file_path)
        if not p.exists():
            return EditorResponse(success=False, error=f"File not found: {file_path}")
        if not p.is_file():
            return EditorResponse(success=False, error=f"Path is not a file: {file_path}")

        try:
            original = p.read_text(encoding="utf-8")
        except OSError as e:
            return EditorResponse(success=False, error=f"Read error: {e}")

        count = original.count(old_string)
        if count == 0:
            return EditorResponse(
                success=False,
                error=f"old_string not found in {file_path}",
            )
        if count > 1 and not replace_all:
            return EditorResponse(
                success=False,
                error=(
                    f"old_string appears {count} times in {file_path}. "
                    "Set replace_all=True to replace all occurrences."
                ),
            )

        updated = original.replace(old_string, new_string, -1 if replace_all else 1)

        try:
            p.write_text(updated, encoding="utf-8")
        except OSError as e:
            return EditorResponse(success=False, error=f"Write error: {e}")

        # Return a snippet around the first edited location
        idx = updated.find(new_string)
        if idx == -1:
            snippet = new_string[:200]
        else:
            snip_start = max(0, idx - 40)
            snip_end = min(len(updated), idx + len(new_string) + 40)
            snippet = updated[snip_start:snip_end]

        return EditorResponse(
            success=True,
            content=f"Replaced {count if replace_all else 1} occurrence(s). Snippet: ...{snippet}...",
        )

    def to_tool(self) -> Tool:
        if self._tool is None:

            def editor_tool(**kwargs):
                """
                Write or edit files on disk.

                Use action='write' to create or fully replace a file. Use action='edit'
                to perform an exact-string replacement — safer than full rewrites for
                targeted changes. Parent directories are created automatically on write.
                Edits fail fast if the old_string is ambiguous (multiple matches) unless
                replace_all=True is set.
                """
                return self.handle_request(EditorRequest(**kwargs)).model_dump()

            if self.system_tool_name != "editor_tool":
                editor_tool.__name__ = self.system_tool_name

            self._tool = Tool(
                func_callable=editor_tool,
                request_options=EditorRequest,
            )
        return self._tool
