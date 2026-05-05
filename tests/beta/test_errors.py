# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/errors.py — LionError hierarchy."""

from __future__ import annotations

import pytest

from lionagi.beta.errors import (
    AccessError,
    ConfigurationError,
    ExecutionError,
    ExistsError,
    LionConnectionError,
    LionError,
    LionTimeoutError,
    NotAllowedError,
    NotFoundError,
    OperationError,
    QueueFullError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# LionError base
# ---------------------------------------------------------------------------


class TestLionError:
    def test_default_message(self):
        e = LionError()
        assert e.message == "lion error"
        assert str(e) == "lion error"

    def test_custom_message(self):
        e = LionError("custom msg")
        assert e.message == "custom msg"

    def test_default_retryable(self):
        e = LionError()
        assert e.retryable is True

    def test_explicit_retryable(self):
        e = LionError(retryable=False)
        assert e.retryable is False

    def test_details_stored(self):
        e = LionError("oops", extra_key="val")
        assert e.details["extra_key"] == "val"

    def test_empty_details(self):
        e = LionError()
        assert e.details == {}

    def test_cause_chained(self):
        cause = ValueError("original")
        e = LionError("wrapper", cause=cause)
        assert e.__cause__ is cause

    def test_to_dict_basic(self):
        e = LionError("test error")
        d = e.to_dict()
        assert d["error"] == "LionError"
        assert d["message"] == "test error"
        assert d["retryable"] is True

    def test_to_dict_no_details_when_empty(self):
        e = LionError()
        d = e.to_dict()
        assert "details" not in d

    def test_to_dict_with_details(self):
        e = LionError("msg", key="val")
        d = e.to_dict()
        assert "details" in d
        assert d["details"]["key"] == "val"

    def test_is_exception(self):
        e = LionError("test")
        assert isinstance(e, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(LionError) as exc_info:
            raise LionError("caught")
        assert exc_info.value.message == "caught"


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_default_message(self):
        e = ValidationError()
        assert e.message == "Validation failed"

    def test_not_retryable(self):
        assert ValidationError().retryable is False

    def test_to_dict_class_name(self):
        d = ValidationError().to_dict()
        assert d["error"] == "ValidationError"

    def test_is_lion_error(self):
        assert isinstance(ValidationError(), LionError)


class TestAccessError:
    def test_default_message(self):
        assert AccessError().message == "Access denied"

    def test_not_retryable(self):
        assert AccessError().retryable is False


class TestConfigurationError:
    def test_default_message(self):
        assert ConfigurationError().message == "Configuration error"

    def test_not_retryable(self):
        assert ConfigurationError().retryable is False


class TestExecutionError:
    def test_default_message(self):
        assert ExecutionError().message == "Execution failed"

    def test_retryable(self):
        assert ExecutionError().retryable is True


class TestLionConnectionError:
    def test_default_message(self):
        assert LionConnectionError().message == "Connection error"

    def test_retryable(self):
        assert LionConnectionError().retryable is True


class TestLionTimeoutError:
    def test_default_message(self):
        assert LionTimeoutError().message == "Operation timed out"

    def test_retryable(self):
        assert LionTimeoutError().retryable is True


class TestNotFoundError:
    def test_default_message(self):
        assert NotFoundError().message == "Item not found"

    def test_not_retryable(self):
        assert NotFoundError().retryable is False


class TestNotAllowedError:
    def test_default_message(self):
        assert NotAllowedError().message == "Item not allowed"

    def test_not_retryable(self):
        assert NotAllowedError().retryable is False


class TestExistsError:
    def test_default_message(self):
        assert ExistsError().message == "Item already exists"

    def test_not_retryable(self):
        assert ExistsError().retryable is False


class TestQueueFullError:
    def test_default_message(self):
        assert QueueFullError().message == "Queue is full"

    def test_retryable(self):
        assert QueueFullError().retryable is True

    def test_with_details(self):
        e = QueueFullError("too full", queue_size=100, max_size=100)
        d = e.to_dict()
        assert d["details"]["queue_size"] == 100


class TestOperationError:
    def test_default_message(self):
        assert OperationError().message == "Operation failed"

    def test_not_retryable(self):
        assert OperationError().retryable is False
