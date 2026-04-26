# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/utils.py: create_path, copy, and related utilities."""

import pytest

from lionagi.utils import copy, create_path


# ---------------------------------------------------------------------------
# A15: create_path rejects backslash and existing file without overwrite
# ---------------------------------------------------------------------------


def test_create_path_rejects_backslash_and_existing_file_without_overwrite(tmp_path):
    with pytest.raises(ValueError):
        create_path(tmp_path, "bad\\name.txt")

    existing = tmp_path / "report.txt"
    existing.write_text("content")

    with pytest.raises(FileExistsError):
        create_path(tmp_path, "report.txt", file_exist_ok=False)


def test_create_path_returns_correct_path(tmp_path):
    p = create_path(tmp_path, "output.txt")
    assert p.parent == tmp_path
    assert p.name == "output.txt"


def test_create_path_creates_parent_directories(tmp_path):
    p = create_path(tmp_path / "deep" / "subdir", "file.txt")
    assert p.parent.exists()


def test_create_path_file_exist_ok_true_allows_existing(tmp_path):
    existing = tmp_path / "exists.txt"
    existing.write_text("data")
    p = create_path(tmp_path, "exists.txt", file_exist_ok=True)
    assert p == existing


def test_create_path_subdirectory_in_filename(tmp_path):
    p = create_path(tmp_path, "sub/file.txt")
    assert p.parent == tmp_path / "sub"
    assert p.name == "file.txt"


def test_create_path_with_extension_arg(tmp_path):
    p = create_path(tmp_path, "report", extension="md")
    assert p.suffix == ".md"
    assert p.stem == "report"


# ---------------------------------------------------------------------------
# copy utility
# ---------------------------------------------------------------------------


def test_copy_deep_returns_independent_copy():
    original = {"x": [1, 2, 3]}
    clone = copy(original)
    clone["x"].append(99)
    assert original["x"] == [1, 2, 3]


def test_copy_shallow_shares_nested():
    original = {"x": [1, 2]}
    clone = copy(original, deep=False)
    clone["x"].append(99)
    assert original["x"][-1] == 99


def test_copy_num_returns_list():
    original = [1, 2]
    copies = copy(original, num=3)
    assert isinstance(copies, list)
    assert len(copies) == 3
    assert all(c == original for c in copies)
    assert all(c is not original for c in copies)


def test_copy_num_one_returns_single():
    original = {"a": 1}
    result = copy(original, num=1)
    assert isinstance(result, dict)
    assert result == original


def test_copy_rejects_non_positive_copy_count():
    with pytest.raises(ValueError):
        copy({"x": []}, num=0)

    with pytest.raises(ValueError):
        copy({"x": []}, num=-1)
