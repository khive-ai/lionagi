# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi/beta/session/context.py — RequestContext."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from lionagi.beta.session.context import RequestContext

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self):
        ctx = RequestContext("chat")
        assert ctx.name == "chat"

    def test_id_is_uuid(self):
        ctx = RequestContext("chat")
        assert isinstance(ctx.id, UUID)

    def test_id_is_unique_per_instance(self):
        ctx1 = RequestContext("chat")
        ctx2 = RequestContext("chat")
        assert ctx1.id != ctx2.id

    def test_explicit_id(self):
        uid = uuid4()
        ctx = RequestContext("chat", id=uid)
        assert ctx.id == uid

    def test_session_id_default_none(self):
        ctx = RequestContext("chat")
        assert ctx.session_id is None

    def test_branch_default_none(self):
        ctx = RequestContext("chat")
        assert ctx.branch is None

    def test_conn_default_none(self):
        ctx = RequestContext("chat")
        assert ctx.conn is None

    def test_service_default_none(self):
        ctx = RequestContext("chat")
        assert ctx.service is None

    def test_session_id_stored(self):
        sid = uuid4()
        ctx = RequestContext("chat", session_id=sid)
        assert ctx.session_id == sid

    def test_branch_stored_as_string(self):
        ctx = RequestContext("chat", branch="main")
        assert ctx.branch == "main"

    def test_branch_stored_as_uuid(self):
        bid = uuid4()
        ctx = RequestContext("chat", branch=bid)
        assert ctx.branch == bid

    def test_conn_stored(self):
        fake_conn = object()
        ctx = RequestContext("chat", conn=fake_conn)
        assert ctx.conn is fake_conn

    def test_extra_kwargs_go_to_metadata(self):
        ctx = RequestContext("chat", foo="bar", baz=42)
        assert ctx.metadata == {"foo": "bar", "baz": 42}

    def test_metadata_empty_when_no_extra_kwargs(self):
        ctx = RequestContext("chat")
        assert ctx.metadata == {}


# ---------------------------------------------------------------------------
# __getattr__
# ---------------------------------------------------------------------------


class TestGetAttr:
    def test_extra_kwarg_accessible_via_attribute(self):
        ctx = RequestContext("chat", foo="bar")
        assert ctx.foo == "bar"

    def test_multiple_extra_kwargs_accessible(self):
        ctx = RequestContext("chat", x=1, y=2)
        assert ctx.x == 1
        assert ctx.y == 2

    def test_underscore_prefix_raises_attribute_error(self):
        ctx = RequestContext("chat")
        with pytest.raises(AttributeError):
            _ = ctx._private

    def test_double_underscore_prefix_raises_attribute_error(self):
        ctx = RequestContext("chat")
        with pytest.raises(AttributeError):
            _ = ctx.__secret

    def test_missing_key_raises_attribute_error(self):
        ctx = RequestContext("chat")
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = ctx.nonexistent

    def test_direct_field_name_does_not_go_through_metadata(self):
        ctx = RequestContext("my_name")
        # 'name' is a direct slot, not metadata
        assert ctx.name == "my_name"

    def test_metadata_key_accessible(self):
        ctx = RequestContext("chat", my_key="my_value")
        assert ctx.my_key == "my_value"


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_bound_session_returned_directly(self):
        fake_session = MagicMock()
        ctx = RequestContext("chat", _bound_session=fake_session)

        result = asyncio.run(ctx.get_session())
        assert result is fake_session

    def test_no_session_id_returns_none(self):
        ctx = RequestContext("chat")
        result = asyncio.run(ctx.get_session())
        assert result is None

    def test_session_id_none_returns_none(self):
        ctx = RequestContext("chat", session_id=None)
        result = asyncio.run(ctx.get_session())
        assert result is None

    def test_session_id_set_calls_registry(self, monkeypatch):
        fake_session = MagicMock()
        sid = uuid4()

        async def mock_get_session(session_id):
            assert session_id == sid
            return fake_session

        monkeypatch.setattr(
            "lionagi.beta.session.registry.get_session", mock_get_session
        )

        ctx = RequestContext("chat", session_id=sid)
        result = asyncio.run(ctx.get_session())
        assert result is fake_session


# ---------------------------------------------------------------------------
# get_branch
# ---------------------------------------------------------------------------


class TestGetBranch:
    def test_bound_branch_returned_directly(self):
        fake_branch = MagicMock()
        ctx = RequestContext("chat", _bound_branch=fake_branch)

        result = asyncio.run(ctx.get_branch())
        assert result is fake_branch

    def test_no_session_returns_none(self):
        ctx = RequestContext("chat")  # no session_id, no _bound_session
        result = asyncio.run(ctx.get_branch())
        assert result is None

    def test_session_but_no_branch_returns_none(self):
        fake_session = MagicMock()
        ctx = RequestContext("chat", _bound_session=fake_session)
        # branch is None by default
        result = asyncio.run(ctx.get_branch())
        assert result is None

    def test_session_and_branch_calls_get_branch(self):
        fake_branch = MagicMock()
        fake_session = MagicMock()
        fake_session.get_branch = MagicMock(return_value=fake_branch)

        ctx = RequestContext("chat", branch="main", _bound_session=fake_session)
        result = asyncio.run(ctx.get_branch())
        fake_session.get_branch.assert_called_once_with("main")
        assert result is fake_branch


# ---------------------------------------------------------------------------
# get_service
# ---------------------------------------------------------------------------


class TestGetService:
    def test_bound_service_returned_directly(self):
        fake_service = MagicMock()
        ctx = RequestContext("chat", _bound_service=fake_service)

        result = asyncio.run(ctx.get_service())
        assert result is fake_service

    def test_simple_name_no_separator_returns_none(self):
        ctx = RequestContext("simple")
        result = asyncio.run(ctx.get_service())
        assert result is None

    def test_dot_separator_extracts_service_name(self, monkeypatch):
        fake_service = MagicMock()

        async def mock_get_service(name):
            assert name == "chat"
            return fake_service

        monkeypatch.setattr(
            "lionagi.beta.resource.service.get_service", mock_get_service
        )

        ctx = RequestContext("chat.generate")
        result = asyncio.run(ctx.get_service())
        assert result is fake_service

    def test_colon_separator_extracts_service_name(self, monkeypatch):
        fake_service = MagicMock()

        async def mock_get_service(name):
            assert name == "chat"
            return fake_service

        monkeypatch.setattr(
            "lionagi.beta.resource.service.get_service", mock_get_service
        )

        ctx = RequestContext("chat:generate")
        result = asyncio.run(ctx.get_service())
        assert result is fake_service

    def test_explicit_service_used_over_name_parsing(self, monkeypatch):
        fake_service = MagicMock()

        async def mock_get_service(name):
            # service is explicitly set to "myservice"
            assert name == "myservice"
            return fake_service

        monkeypatch.setattr(
            "lionagi.beta.resource.service.get_service", mock_get_service
        )

        ctx = RequestContext("chat.generate", service="myservice")
        result = asyncio.run(ctx.get_service())
        assert result is fake_service

    def test_no_service_no_separator_returns_none(self):
        ctx = RequestContext("plainname")
        result = asyncio.run(ctx.get_service())
        assert result is None

    def test_colon_takes_prefix_only(self, monkeypatch):
        """chat:sub:sub2 should yield 'chat' as service."""
        fake_service = MagicMock()

        async def mock_get_service(name):
            return fake_service if name == "chat" else None

        monkeypatch.setattr(
            "lionagi.beta.resource.service.get_service", mock_get_service
        )

        ctx = RequestContext("chat:sub:sub2")
        result = asyncio.run(ctx.get_service())
        assert result is fake_service

    def test_dot_takes_prefix_only(self, monkeypatch):
        """a.b.c should yield 'a' as service."""
        fake_service = MagicMock()

        async def mock_get_service(name):
            return fake_service if name == "a" else None

        monkeypatch.setattr(
            "lionagi.beta.resource.service.get_service", mock_get_service
        )

        ctx = RequestContext("a.b.c")
        result = asyncio.run(ctx.get_service())
        assert result is fake_service


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


class TestMisc:
    def test_query_fn_stored(self):
        fn = lambda: None
        ctx = RequestContext("chat", query_fn=fn)
        assert ctx.query_fn is fn

    def test_now_stored(self):
        from datetime import datetime

        now = datetime.now()
        ctx = RequestContext("chat", now=now)
        assert ctx.now == now

    def test_service_stored_as_uuid(self):
        uid = uuid4()
        ctx = RequestContext("chat", service=uid)
        assert ctx.service == uid

    def test_service_stored_as_string(self):
        ctx = RequestContext("chat", service="my-svc")
        assert ctx.service == "my-svc"
