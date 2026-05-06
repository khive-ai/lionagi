# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/act.py — _resolve_scope and ActParams."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lionagi.operations.act import ActParams, _resolve_scope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_toolkit(name: str, actions: list[str]):
    tk = MagicMock()
    tk.name = name
    tk.allowed_actions = set(actions)
    return tk


# ---------------------------------------------------------------------------
# _resolve_scope
# ---------------------------------------------------------------------------


class TestResolveScope:
    def test_canonical_colon_form(self):
        tks = [make_toolkit("foo", ["wolf"])]
        scope, action, hint = _resolve_scope("foo:wolf", tks)
        assert scope == "foo:wolf"
        assert action == "wolf"
        assert hint is None

    def test_dotted_form(self):
        tks = [make_toolkit("foo", ["wolf"])]
        scope, action, hint = _resolve_scope("foo.wolf", tks)
        assert scope == "foo:wolf"
        assert action == "wolf"
        assert hint is None

    def test_bare_action_unambiguous(self):
        tks = [make_toolkit("foo", ["wolf"])]
        scope, action, hint = _resolve_scope("wolf", tks)
        assert scope == "foo:wolf"
        assert action == "wolf"
        assert hint is None

    def test_bare_action_ambiguous(self):
        tks = [
            make_toolkit("foo", ["wolf"]),
            make_toolkit("bar", ["wolf"]),
        ]
        scope, action, hint = _resolve_scope("wolf", tks)
        assert hint is not None
        assert "ambiguous" in hint

    def test_bare_toolkit_name(self):
        tks = [make_toolkit("foo", ["alpha", "beta"])]
        scope, action, hint = _resolve_scope("foo", tks)
        assert scope == "foo"
        assert action is None
        assert hint is None

    def test_unknown_name(self):
        tks = [make_toolkit("foo", ["alpha"])]
        scope, action, hint = _resolve_scope("nonexistent", tks)
        assert hint is not None
        assert "unknown" in hint

    def test_empty_toolkits_returns_hint(self):
        scope, action, hint = _resolve_scope("something", [])
        assert hint is not None


# ---------------------------------------------------------------------------
# ActParams
# ---------------------------------------------------------------------------


class TestActParams:
    def test_construction(self):
        params = ActParams(action_requests=[MagicMock()])
        assert params.delay_before_start == 0
        assert params.strategy == "concurrent"

    def test_sequential_strategy(self):
        params = ActParams(action_requests=[], strategy="sequential")
        assert params.strategy == "sequential"

    def test_max_concurrent(self):
        params = ActParams(action_requests=[], max_concurrent=5)
        assert params.max_concurrent == 5

    def test_throttle_period(self):
        params = ActParams(action_requests=[], throttle_period=0.5)
        assert params.throttle_period == 0.5

    def test_toolkits_default_none(self):
        params = ActParams(action_requests=[])
        assert params.toolkits is None

    def test_to_dict_includes_action_requests(self):
        msg = MagicMock()
        params = ActParams(action_requests=[msg])
        d = params.to_dict()
        assert "action_requests" in d
        assert len(d["action_requests"]) == 1
