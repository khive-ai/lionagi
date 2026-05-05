# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.beta.work.flow helper functions (no live Session required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lionagi.beta.core.runner import Runner
from lionagi.beta.core.types import Principal
from lionagi.beta.work.flow import (
    OperationResult,
    _default_branch,
    _operation_name,
    _principal_for_flow,
    _resolve_operation_branch,
    _runner_for_session,
    _unwrap_result,
)
from lionagi.beta.work.node import Operation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_op(name: str | None = None) -> Operation:
    op = Operation(operation_type="chat")
    if name is not None:
        op.metadata["name"] = name
    return op


# ---------------------------------------------------------------------------
# OperationResult
# ---------------------------------------------------------------------------


class TestOperationResult:
    def test_basic_construction(self):
        r = OperationResult(name="op1", result=42)
        assert r.name == "op1"
        assert r.result == 42
        assert r.error is None
        assert r.completed == 0
        assert r.total == 0

    def test_success_true_when_error_none(self):
        r = OperationResult(name="op1", result="x")
        assert r.success is True

    def test_success_false_when_error_set(self):
        r = OperationResult(name="op1", result=None, error=ValueError("boom"))
        assert r.success is False

    def test_success_false_when_error_is_exception_instance(self):
        r = OperationResult(name="x", result=None, error=RuntimeError("err"))
        assert r.success is False

    def test_completed_and_total_stored(self):
        r = OperationResult(name="op", result={}, completed=3, total=10)
        assert r.completed == 3
        assert r.total == 10

    def test_result_can_be_dict(self):
        r = OperationResult(name="op", result={"key": "val"})
        assert r.result == {"key": "val"}

    def test_result_can_be_none(self):
        r = OperationResult(name="op", result=None)
        assert r.result is None

    def test_name_stored_correctly(self):
        r = OperationResult(name="my_op_name", result=1)
        assert r.name == "my_op_name"


# ---------------------------------------------------------------------------
# _operation_name
# ---------------------------------------------------------------------------


class TestOperationName:
    def test_returns_metadata_name_when_set(self):
        op = _make_op("my_op")
        assert _operation_name(op) == "my_op"

    def test_returns_str_of_id_when_no_name_in_metadata(self):
        op = _make_op()  # no name set
        result = _operation_name(op)
        assert result == str(op.id)

    def test_returns_string_type(self):
        op = _make_op()
        assert isinstance(_operation_name(op), str)

    def test_name_with_spaces_preserved(self):
        op = _make_op("my operation name")
        assert _operation_name(op) == "my operation name"

    def test_name_overrides_id(self):
        op = _make_op("override")
        assert _operation_name(op) != str(op.id)


# ---------------------------------------------------------------------------
# _unwrap_result
# ---------------------------------------------------------------------------


class TestUnwrapResult:
    def test_single_result_key_unwraps(self):
        assert _unwrap_result({"result": 42}) == 42

    def test_single_result_key_none_unwraps(self):
        assert _unwrap_result({"result": None}) is None

    def test_single_result_key_dict_value_unwraps(self):
        inner = {"a": 1}
        assert _unwrap_result({"result": inner}) is inner

    def test_multi_key_dict_not_unwrapped(self):
        d = {"result": 1, "extra": 2}
        assert _unwrap_result(d) is d

    def test_empty_dict_not_unwrapped(self):
        d = {}
        assert _unwrap_result(d) is d

    def test_no_result_key_not_unwrapped(self):
        d = {"other": 99}
        assert _unwrap_result(d) is d

    def test_result_false_value_unwraps(self):
        # {"result": False} has exactly one key, should unwrap
        assert _unwrap_result({"result": False}) is False

    def test_result_zero_value_unwraps(self):
        assert _unwrap_result({"result": 0}) == 0


# ---------------------------------------------------------------------------
# _resolve_operation_branch
# ---------------------------------------------------------------------------


class TestResolveOperationBranch:
    def _make_session(self):
        session = MagicMock()
        return session

    def test_none_returns_default_branch(self):
        default = MagicMock()
        session = self._make_session()
        result = _resolve_operation_branch(session, None, default)
        assert result is default

    def test_branch_obj_with_id_and_order_returns_as_is(self):
        session = self._make_session()
        branch_obj = MagicMock()
        branch_obj.id = "some-id"
        branch_obj.order = 1
        default = MagicMock()
        result = _resolve_operation_branch(session, branch_obj, default)
        assert result is branch_obj

    def test_string_branch_calls_session_get_branch(self):
        session = self._make_session()
        mock_branch = MagicMock()
        session.get_branch.return_value = mock_branch
        default = MagicMock()
        result = _resolve_operation_branch(session, "my_branch", default)
        session.get_branch.assert_called_once_with("my_branch")
        assert result is mock_branch

    def test_session_get_branch_raises_returns_default(self):
        session = self._make_session()
        session.get_branch.side_effect = KeyError("not found")
        default = MagicMock()
        result = _resolve_operation_branch(session, "bad_branch", default)
        assert result is default

    def test_object_without_id_and_order_calls_get_branch(self):
        session = self._make_session()
        mock_branch = MagicMock()
        session.get_branch.return_value = mock_branch
        # An object with 'id' but without 'order'
        obj = MagicMock(spec=["id"])
        default = MagicMock()
        result = _resolve_operation_branch(session, obj, default)
        session.get_branch.assert_called_once_with(obj)
        assert result is mock_branch


# ---------------------------------------------------------------------------
# _default_branch
# ---------------------------------------------------------------------------


class TestDefaultBranch:
    def test_none_returns_session_default_branch(self):
        session = MagicMock()
        default = MagicMock()
        session.default_branch = default
        result = _default_branch(session, None)
        assert result is default

    def test_none_no_default_branch_attr_returns_none(self):
        session = MagicMock(spec=[])  # no default_branch attr
        result = _default_branch(session, None)
        assert result is None

    def test_branch_name_calls_session_get_branch(self):
        session = MagicMock()
        mock_branch = MagicMock()
        session.get_branch.return_value = mock_branch
        result = _default_branch(session, "my_branch")
        session.get_branch.assert_called_once_with("my_branch")
        assert result is mock_branch

    def test_branch_uuid_calls_session_get_branch(self):
        from uuid import uuid4

        session = MagicMock()
        uid = uuid4()
        mock_branch = MagicMock()
        session.get_branch.return_value = mock_branch
        result = _default_branch(session, uid)
        session.get_branch.assert_called_once_with(uid)
        assert result is mock_branch


# ---------------------------------------------------------------------------
# _principal_for_flow
# ---------------------------------------------------------------------------


class TestPrincipalForFlow:
    def test_no_branch_creates_flow_principal(self):
        p = _principal_for_flow(None, {}, None)
        assert isinstance(p, Principal)
        assert p.name == "flow"

    def test_no_branch_no_context_empty_ctx(self):
        p = _principal_for_flow(None, {}, None)
        assert p.ctx == {}

    def test_context_updates_principal_ctx(self):
        p = _principal_for_flow(None, {}, {"key": "val", "num": 42})
        assert p.ctx["key"] == "val"
        assert p.ctx["num"] == 42

    def test_with_default_branch_copies_principal(self):
        branch = MagicMock()
        source_principal = Principal(name="source_branch")
        branch.principal = source_principal
        p = _principal_for_flow(branch, {}, None)
        assert p.name == "source_branch"
        # It should be a copy, not the same instance
        assert p is not source_principal

    def test_with_default_branch_and_context_merged(self):
        branch = MagicMock()
        source_principal = Principal(name="branch")
        branch.principal = source_principal
        p = _principal_for_flow(branch, {}, {"injected": True})
        assert p.ctx.get("injected") is True

    def test_operation_branches_fallback_when_no_default(self):
        branch = MagicMock()
        source_principal = Principal(name="op_branch")
        branch.principal = source_principal
        op_branches = {"op1": branch}
        p = _principal_for_flow(None, op_branches, None)
        assert p.name == "op_branch"

    def test_operation_branches_all_none_creates_flow_principal(self):
        op_branches = {"op1": None, "op2": None}
        p = _principal_for_flow(None, op_branches, None)
        assert p.name == "flow"


# ---------------------------------------------------------------------------
# _runner_for_session
# ---------------------------------------------------------------------------


class TestRunnerForSession:
    def test_none_max_concurrent_with_get_runner_uses_session_runner(self):
        session = MagicMock()
        mock_runner = Runner()
        session._get_runner.return_value = mock_runner
        result = _runner_for_session(session, None)
        assert result is mock_runner
        session._get_runner.assert_called_once()

    def test_with_max_concurrent_creates_new_runner(self):
        session = MagicMock()
        result = _runner_for_session(session, 4)
        assert isinstance(result, Runner)
        assert result.max_concurrent == 4

    def test_none_max_concurrent_without_get_runner_creates_new_runner(self):
        session = MagicMock(spec=[])  # no _get_runner attr
        result = _runner_for_session(session, None)
        assert isinstance(result, Runner)
        assert result.max_concurrent is None

    def test_max_concurrent_1_creates_runner_with_limit_1(self):
        session = MagicMock(spec=[])
        result = _runner_for_session(session, 1)
        assert result.max_concurrent == 1

    def test_session_get_runner_not_called_when_max_concurrent_set(self):
        session = MagicMock()
        mock_runner = Runner()
        session._get_runner.return_value = mock_runner
        result = _runner_for_session(session, 4)
        # Should NOT use session._get_runner when max_concurrent is given
        session._get_runner.assert_not_called()
        assert result is not mock_runner
