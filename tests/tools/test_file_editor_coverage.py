# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage-targeted tests for tools/file/editor.py uncovered branches."""

from __future__ import annotations

from pathlib import Path

import pytest

import lionagi.tools.file.editor as editor_module
from lionagi.tools.file.editor import (
    EditorAction,
    EditorRequest,
    EditorResponse,
    EditorTool,
    _edit_sync,
    _resolve_existing_workspace_file,
    _write_sync,
)

# ---------------------------------------------------------------------------
# _resolve_existing_workspace_file — symlink path (line 44)
# ---------------------------------------------------------------------------


def test_resolve_existing_workspace_file_symlink_raises(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("content")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    with pytest.raises(PermissionError, match="symlink"):
        _resolve_existing_workspace_file(str(link), tmp_path)


def test_resolve_existing_workspace_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_existing_workspace_file("nonexistent.txt", tmp_path)


def test_resolve_existing_workspace_file_directory_raises(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        _resolve_existing_workspace_file("subdir", tmp_path)


def test_resolve_existing_workspace_file_valid(tmp_path):
    f = tmp_path / "ok.txt"
    f.write_text("hello")
    p = _resolve_existing_workspace_file("ok.txt", tmp_path)
    assert p == f.resolve()


# ---------------------------------------------------------------------------
# _write_sync — mtime staleness check (lines 134-139)
# ---------------------------------------------------------------------------


def test_write_sync_mtime_staleness_returns_error(tmp_path):
    f = tmp_path / "tracked.txt"
    f.write_text("original")
    key = str(f.resolve())
    # record a WRONG (outdated) mtime
    editor_module._file_states[key] = 0
    try:
        result = _write_sync(str(f), "new content", tmp_path)
        assert result.success is False
        assert "changed since last read" in result.error
    finally:
        editor_module._file_states.pop(key, None)


def test_write_sync_oserror_on_write(tmp_path, monkeypatch):
    f = tmp_path / "writelock.txt"
    original_write_text = Path.write_text

    def fake_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fake_write)
    result = _write_sync("writelock.txt", "content", tmp_path)
    assert result.success is False
    assert "Write error" in result.error


def test_write_sync_mtime_update_after_success(tmp_path):
    f = tmp_path / "newfile.txt"
    key = str(f.resolve())
    editor_module._file_states.pop(key, None)
    try:
        result = _write_sync("newfile.txt", "hello world", tmp_path)
        assert result.success is True
        assert key in editor_module._file_states
    finally:
        editor_module._file_states.pop(key, None)


# ---------------------------------------------------------------------------
# _edit_sync — mtime staleness on edit (lines 171-176)
# ---------------------------------------------------------------------------


def test_edit_sync_mtime_staleness_returns_error(tmp_path):
    f = tmp_path / "editable.txt"
    f.write_text("hello world")
    key = str(f.resolve())
    editor_module._file_states[key] = 0  # stale mtime
    try:
        result = _edit_sync("editable.txt", "hello", "goodbye", False, tmp_path)
        assert result.success is False
        assert "changed since last read" in result.error
    finally:
        editor_module._file_states.pop(key, None)


def test_edit_sync_oserror_on_read(tmp_path, monkeypatch):
    f = tmp_path / "target.txt"
    f.write_text("foo bar")
    original_read = Path.read_text

    def fake_read(self, *args, **kwargs):
        raise OSError("io error")

    monkeypatch.setattr(Path, "read_text", fake_read)
    result = _edit_sync("target.txt", "foo", "baz", False, tmp_path)
    assert result.success is False
    assert "Read error" in result.error


def test_edit_sync_snippet_when_new_string_empty(tmp_path):
    f = tmp_path / "snippet.txt"
    f.write_text("remove this word")
    # Replace with empty string — updated.find("") == 0, not -1
    result = _edit_sync("snippet.txt", "remove this word", "", False, tmp_path)
    assert result.success is True


def test_edit_sync_mtime_updated_after_success(tmp_path):
    f = tmp_path / "mtime_edit.txt"
    f.write_text("alpha beta")
    key = str(f.resolve())
    editor_module._file_states.pop(key, None)
    try:
        result = _edit_sync("mtime_edit.txt", "alpha", "gamma", False, tmp_path)
        assert result.success is True
        assert key in editor_module._file_states
    finally:
        editor_module._file_states.pop(key, None)


# ---------------------------------------------------------------------------
# EditorTool.handle_request — unknown action (line 263)
# ---------------------------------------------------------------------------


async def test_handle_request_unknown_action_returns_error(tmp_path):
    tool = EditorTool(workspace_root=tmp_path)
    # Inject a request with an unrecognized action by bypassing enum validation
    req = EditorRequest(action=EditorAction.write, file_path="x.txt", content="y")
    # Patch action to simulate unknown
    object.__setattr__(req, "action", "unknown_action")
    result = await tool.handle_request(req)
    assert result.success is False
    assert "Unknown action" in result.error


# ---------------------------------------------------------------------------
# EditorTool.to_tool — callable rename (line 282)
# ---------------------------------------------------------------------------


def test_to_tool_renamed_callable_when_custom_system_name(tmp_path):
    class CustomEditor(EditorTool):
        system_tool_name = "my_custom_editor"

    tool = CustomEditor(workspace_root=tmp_path)
    t = tool.to_tool()
    assert t is not None
    # The callable should be renamed
    assert t.func_callable.__name__ == "my_custom_editor"
