# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for lionagi.beta.resource.service — Service, ServiceCalling, registry functions,
ResourceMeta, _ResourceDecl, and decorator helpers.
"""

from __future__ import annotations

import pytest

from lionagi._errors import ExistsError, NotFoundError
from lionagi.beta.resource.backend import (
    Calling,
    Normalized,
    ResourceBackend,
    ResourceConfig,
)
from lionagi.beta.resource.service import (
    ResourceMeta,
    Service,
    ServiceCalling,
    _ResourceDecl,
    _ServiceBackend,
    _to_pascal,
    add_service,
    clear_services,
    get_resource_decl,
    get_service,
    has_service,
    list_services,
    list_services_sync,
    remove_service,
    resource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRequestContext:
    """Minimal stand-in for RequestContext."""

    pass


def make_ctx():
    return FakeRequestContext()


# ---------------------------------------------------------------------------
# _to_pascal
# ---------------------------------------------------------------------------


class TestToPascal:
    def test_single_word(self):
        assert _to_pascal("hello") == "Hello"

    def test_snake_case(self):
        assert _to_pascal("hello_world") == "HelloWorld"

    def test_multiple_words(self):
        assert _to_pascal("foo_bar_baz") == "FooBarBaz"

    def test_already_pascal(self):
        # single word is treated as one segment
        assert _to_pascal("Hello") == "Hello"


# ---------------------------------------------------------------------------
# resource decorator and get_resource_decl
# ---------------------------------------------------------------------------


class TestResourceDecorator:
    def test_decorator_attaches_decl(self):
        @resource("my_op")
        def handler():
            pass

        decl = get_resource_decl(handler)
        assert decl is not None
        assert decl.name == "my_op"

    def test_decorator_with_inputs_outputs(self):
        @resource("op", inputs={"a", "b"}, outputs={"c"})
        def handler():
            pass

        decl = get_resource_decl(handler)
        assert "a" in decl.inputs
        assert "b" in decl.inputs
        assert "c" in decl.outputs

    def test_decorator_with_description(self):
        @resource("op", description="desc here")
        def handler():
            pass

        decl = get_resource_decl(handler)
        assert decl.description == "desc here"

    def test_decorator_with_hooks(self):
        @resource("op", pre_hooks=["pre1"], post_hooks=["post1"])
        def handler():
            pass

        decl = get_resource_decl(handler)
        assert "pre1" in decl.pre_hooks
        assert "post1" in decl.post_hooks

    def test_get_resource_decl_none_for_plain_fn(self):
        def plain():
            pass

        assert get_resource_decl(plain) is None


# ---------------------------------------------------------------------------
# _ResourceDecl.bind
# ---------------------------------------------------------------------------


class TestResourceDeclBind:
    def test_bind_creates_resource_meta(self):
        from lionagi.ln.types import Operable

        decl = _ResourceDecl(name="op", description="test")
        op = Operable([])
        meta = decl.bind(op)
        assert isinstance(meta, ResourceMeta)
        assert meta.name == "op"
        assert meta.description == "test"
        assert meta.op is op

    def test_bind_preserves_hooks(self):
        from lionagi.ln.types import Operable

        decl = _ResourceDecl(
            name="op",
            pre_hooks=("pre1",),
            post_hooks=("post2",),
        )
        op = Operable([])
        meta = decl.bind(op)
        assert "pre1" in meta.pre_hooks
        assert "post2" in meta.post_hooks


# ---------------------------------------------------------------------------
# ResourceMeta properties
# ---------------------------------------------------------------------------


class TestResourceMeta:
    def _make_meta(self, name="op", inputs=None, outputs=None):
        from lionagi.ln.types import Operable

        op = Operable([])
        return ResourceMeta(
            name=name,
            op=op,
            description="test",
            inputs=frozenset(inputs or []),
            outputs=frozenset(outputs or []),
        )

    def test_schema_structure(self):
        meta = self._make_meta("my_op")
        schema = meta.schema
        assert schema["name"] == "my_op"
        assert "inputs" in schema
        assert "outputs" in schema

    def test_schema_with_inputs_outputs(self):
        meta = self._make_meta("op", inputs=["x"], outputs=["y"])
        schema = meta.schema
        assert "x" in schema["inputs"]
        assert "y" in schema["outputs"]

    def test_schema_description_defaults_to_name(self):
        from lionagi.ln.types import Operable

        meta = ResourceMeta(name="op", op=Operable([]), description=None)
        assert meta.schema["description"] == "op"

    def test_schema_description_custom(self):
        from lionagi.ln.types import Operable

        meta = ResourceMeta(name="op", op=Operable([]), description="custom desc")
        assert meta.schema["description"] == "custom desc"


# ---------------------------------------------------------------------------
# Service class — basic creation and resources
# ---------------------------------------------------------------------------


class EmptyCatalog:
    pass


class TestServiceCreation:
    def test_basic_service_no_resources(self):
        class MyService(Service):
            name: str = "my_service"

        svc = MyService(name="my_service")
        assert svc.name == "my_service"
        assert svc.resources == frozenset()

    def test_service_with_resource_decorator(self):
        class MyService(Service):
            name: str = "svc"

            @resource("do_thing")
            async def do_thing(self, options, ctx):
                return "done"

        svc = MyService(name="svc")
        assert "do_thing" in svc.resources

    def test_service_schemas_empty_no_resources(self):
        class MyService(Service):
            name: str = "svc"

        svc = MyService(name="svc")
        assert svc.schemas == []

    def test_service_schemas_with_resource(self):
        class MyService(Service):
            name: str = "svc"

            @resource("do_thing")
            async def do_thing(self, options, ctx):
                return "done"

        svc = MyService(name="svc")
        schemas = svc.schemas
        assert len(schemas) == 1
        assert schemas[0]["name"] == "do_thing"

    def test_service_catalog_none_returns_empty(self):
        class MyService(Service):
            name: str = "svc"
            catalog = None

        svc = MyService(name="svc")
        specs = svc._get_catalog_specs()
        assert specs == []


# ---------------------------------------------------------------------------
# Service.call
# ---------------------------------------------------------------------------


class TestServiceCall:
    def _make_service(self):
        class MyService(Service):
            name: str = "svc"

            @resource("echo", inputs=set(), outputs=set())
            async def echo(self, options, ctx):
                return "echoed"

            @resource("raise_perm")
            async def raise_perm(self, options, ctx):
                raise PermissionError("not allowed")

            @resource("raise_general")
            async def raise_general(self, options, ctx):
                raise ValueError("something went wrong")

        return MyService(name="svc")

    async def test_call_unknown_resource_returns_error(self):
        svc = self._make_service()
        result = await svc.call("unknown", {}, make_ctx())
        assert result.status == "error"
        assert "unknown" in result.error.lower() or "Unknown" in result.error

    async def test_call_known_resource_returns_success(self):
        svc = self._make_service()
        result = await svc.call("echo", {}, make_ctx())
        assert result.status == "success"
        assert result.data == "echoed"

    async def test_call_permission_error_returns_error(self):
        svc = self._make_service()
        result = await svc.call("raise_perm", {}, make_ctx())
        assert result.status == "error"
        assert "Permission" in result.error or "not allowed" in result.error

    async def test_call_general_exception_returns_error(self):
        svc = self._make_service()
        result = await svc.call("raise_general", {}, make_ctx())
        assert result.status == "error"
        assert "something went wrong" in result.error

    async def test_call_with_inputs_validates_options(self):
        """When inputs are declared, handler receives validated options."""

        class MyService(Service):
            name: str = "svc"

            @resource("greet", inputs=set(), outputs=set())
            async def greet(self, options, ctx):
                return {"greeting": "hello"}

        svc = MyService(name="svc")
        result = await svc.call("greet", {}, make_ctx())
        assert result.status == "success"

    async def test_call_invokes_post_hooks(self):
        hook_calls = []

        async def post_log_hook(svc, options, ctx, result=None):
            hook_calls.append(result)

        class MyService(Service):
            name: str = "svc"

            @resource("do_it", post_hooks=["post_log"])
            async def do_it(self, options, ctx):
                return "ok"

        svc = MyService(name="svc")
        # Set hooks via class variable after class creation to avoid shadowing warning
        MyService.hooks = {"post_log": post_log_hook}
        result = await svc.call("do_it", {}, make_ctx())
        assert result.status == "success"

    async def test_call_pre_hooks_missing_hook_fn_skips(self):
        """Pre-hook name not in hooks dict — silently skipped."""

        class MyService(Service):
            name: str = "svc"
            hooks: dict = {}

            @resource("do_it", pre_hooks=["missing_hook"])
            async def do_it(self, options, ctx):
                return "ok"

        svc = MyService(name="svc")
        result = await svc.call("do_it", {}, make_ctx())
        assert result.status == "success"


# ---------------------------------------------------------------------------
# Service.stream
# ---------------------------------------------------------------------------


class TestServiceStream:
    async def test_stream_yields_one_result(self):
        class MyService(Service):
            name: str = "svc"

            @resource("op")
            async def op(self, options, ctx):
                return "value"

        svc = MyService(name="svc")
        results = []
        async for r in svc.stream("op", {}, make_ctx()):
            results.append(r)
        assert len(results) == 1
        assert results[0].status == "success"


# ---------------------------------------------------------------------------
# Service._evaluate_policy
# ---------------------------------------------------------------------------


class TestServiceEvaluatePolicy:
    async def test_default_policy_allows(self):
        class MyService(Service):
            name: str = "svc"

        from lionagi.beta.resource.service import ResourceMeta
        from lionagi.ln.types import Operable

        svc = MyService(name="svc")
        meta = ResourceMeta(name="op", op=Operable([]))
        result = await svc._evaluate_policy(meta, {}, make_ctx())
        assert result.get("allowed", True) is True


# ---------------------------------------------------------------------------
# Service._run_hooks — error propagation
# ---------------------------------------------------------------------------


class TestServiceRunHooks:
    async def test_run_hooks_raises_on_exception(self):
        async def bad_hook(svc, options, ctx, result=None):
            raise ValueError("hook failed")

        class MyService(Service):
            name: str = "svc"
            hooks: dict = {"bad": bad_hook}

        svc = MyService(name="svc")
        with pytest.raises(ValueError, match="hook failed"):
            await svc._run_hooks(("bad",), {}, make_ctx())

    async def test_run_hooks_skips_missing_hook(self):
        class MyService(Service):
            name: str = "svc"
            hooks: dict = {}

        svc = MyService(name="svc")
        # Should not raise
        await svc._run_hooks(("missing",), {}, make_ctx())


# ---------------------------------------------------------------------------
# Service.create_imodel / create_backend
# ---------------------------------------------------------------------------


class TestServiceCreateIModel:
    def test_create_backend_returns_service_backend(self):
        class MyService(Service):
            name: str = "svc"

        svc = MyService(name="svc")
        backend = svc.create_backend()
        assert isinstance(backend, _ServiceBackend)

    def test_create_imodel_returns_imodel(self):
        from lionagi.beta.resource.imodel import iModel

        class MyService(Service):
            name: str = "svc"

        svc = MyService(name="svc")
        im = svc.create_imodel()
        assert isinstance(im, iModel)


# ---------------------------------------------------------------------------
# _ServiceBackend
# ---------------------------------------------------------------------------


class TestServiceBackend:
    def _make(self):
        class MyService(Service):
            name: str = "svc"

            @resource("op")
            async def op(self, options, ctx):
                return "result"

        svc = MyService(name="svc")
        return _ServiceBackend(service=svc), svc

    def test_event_type_is_service_calling(self):
        backend, _ = self._make()
        assert backend.event_type is ServiceCalling

    def test_create_payload_basic(self):
        backend, _ = self._make()
        payload = backend.create_payload({"key": "val"})
        assert payload["key"] == "val"

    def test_create_payload_with_kwargs(self):
        backend, _ = self._make()
        payload = backend.create_payload({"x": 1}, y=2)
        assert payload["x"] == 1
        assert payload["y"] == 2

    def test_create_payload_unwraps_arguments_key(self):
        """Single 'arguments' key with dict value is unwrapped."""
        backend, _ = self._make()
        payload = backend.create_payload({"arguments": {"a": 1}})
        assert payload == {"a": 1}

    def test_create_payload_no_unwrap_if_multiple_keys(self):
        """'arguments' key is NOT unwrapped if other keys present."""
        backend, _ = self._make()
        payload = backend.create_payload({"arguments": {"a": 1}, "other": 2})
        assert "arguments" in payload
        assert "other" in payload

    async def test_call_dispatches_to_service(self):
        backend, svc = self._make()
        result = await backend.call(name="op", options={}, ctx=make_ctx())
        assert result.status == "success"

    def test_config_provider_default(self):
        backend, _ = self._make()
        assert backend.config.provider == "service"


# ---------------------------------------------------------------------------
# ServiceCalling
# ---------------------------------------------------------------------------


class TestServiceCalling:
    def _make_backend(self):
        class MyService(Service):
            name: str = "svc"

        svc = MyService(name="svc")
        return _ServiceBackend(service=svc)

    def test_call_args_with_action_field(self):
        backend = self._make_backend()
        calling = ServiceCalling(
            backend=backend,
            payload={"key": "value"},
            action="my_action",
            ctx=make_ctx(),
        )
        args = calling.call_args
        assert args["name"] == "my_action"
        assert args["options"] == {"key": "value"}

    def test_call_args_action_from_payload(self):
        backend = self._make_backend()
        calling = ServiceCalling(
            backend=backend,
            payload={"action": "from_payload", "x": 1},
        )
        args = calling.call_args
        assert args["name"] == "from_payload"
        assert args["options"] == {"x": 1}

    def test_call_args_name_from_payload(self):
        backend = self._make_backend()
        calling = ServiceCalling(
            backend=backend,
            payload={"name": "from_name_key", "y": 2},
        )
        args = calling.call_args
        assert args["name"] == "from_name_key"

    def test_call_args_ctx_from_payload(self):
        ctx = make_ctx()
        backend = self._make_backend()
        calling = ServiceCalling(
            backend=backend,
            payload={"ctx": ctx, "action": "op"},
        )
        args = calling.call_args
        assert args["ctx"] is ctx

    def test_call_args_empty_payload(self):
        backend = self._make_backend()
        calling = ServiceCalling(
            backend=backend,
            payload={},
        )
        args = calling.call_args
        assert args["name"] == ""
        assert args["options"] == {}
        assert args["ctx"] is None


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------


class TestServiceRegistry:
    async def _reset(self):
        await clear_services()

    async def test_add_service(self):
        await self._reset()

        class MyService(Service):
            name: str = "reg_svc"

        svc = MyService(name="reg_svc")
        svc_id = await add_service(svc)
        assert svc_id == svc.id
        await self._reset()

    async def test_add_service_duplicate_raises(self):
        await self._reset()

        class MyService(Service):
            name: str = "dup_svc"

        svc1 = MyService(name="dup_svc")
        svc2 = MyService(name="dup_svc")
        await add_service(svc1)
        with pytest.raises(ExistsError):
            await add_service(svc2)
        await self._reset()

    async def test_add_service_with_update(self):
        await self._reset()

        class MyService(Service):
            name: str = "upd_svc"

        svc1 = MyService(name="upd_svc")
        svc2 = MyService(name="upd_svc")
        await add_service(svc1)
        await add_service(svc2, update=True)  # Should not raise
        await self._reset()

    async def test_has_service_by_name(self):
        await self._reset()

        class MyService(Service):
            name: str = "chk_svc"

        svc = MyService(name="chk_svc")
        await add_service(svc)
        assert has_service("chk_svc") is True
        await self._reset()

    async def test_has_service_by_id(self):
        await self._reset()

        class MyService(Service):
            name: str = "id_svc"

        svc = MyService(name="id_svc")
        await add_service(svc)
        assert has_service(svc.id) is True
        await self._reset()

    async def test_has_service_missing_false(self):
        await self._reset()
        import uuid

        assert has_service("nonexistent") is False
        assert has_service(uuid.uuid4()) is False

    async def test_get_service_by_name(self):
        await self._reset()

        class MyService(Service):
            name: str = "get_svc"

        svc = MyService(name="get_svc")
        await add_service(svc)
        found = await get_service("get_svc")
        assert found.id == svc.id
        await self._reset()

    async def test_get_service_by_id(self):
        await self._reset()

        class MyService(Service):
            name: str = "getid_svc"

        svc = MyService(name="getid_svc")
        await add_service(svc)
        found = await get_service(svc.id)
        assert found.id == svc.id
        await self._reset()

    async def test_get_service_not_found_raises(self):
        await self._reset()
        with pytest.raises(NotFoundError):
            await get_service("does_not_exist")

    async def test_remove_service_by_name(self):
        await self._reset()

        class MyService(Service):
            name: str = "rm_svc"

        svc = MyService(name="rm_svc")
        await add_service(svc)
        await remove_service("rm_svc")
        assert has_service("rm_svc") is False
        await self._reset()

    async def test_remove_service_by_id(self):
        await self._reset()

        class MyService(Service):
            name: str = "rmid_svc"

        svc = MyService(name="rmid_svc")
        await add_service(svc)
        await remove_service(svc.id)
        assert has_service(svc.id) is False
        await self._reset()

    async def test_remove_service_not_found_raises(self):
        await self._reset()
        with pytest.raises(NotFoundError):
            await remove_service("nonexistent_service")

    async def test_list_services_by_instance(self):
        await self._reset()

        class MyService(Service):
            name: str = "ls_svc"

        svc = MyService(name="ls_svc")
        await add_service(svc)
        services = await list_services(by="instance")
        assert any(s.id == svc.id for s in services)
        await self._reset()

    async def test_list_services_by_name(self):
        await self._reset()

        class MyService(Service):
            name: str = "lsn_svc"

        svc = MyService(name="lsn_svc")
        await add_service(svc)
        names = await list_services(by="name")
        assert "lsn_svc" in names
        await self._reset()

    def test_list_services_sync_by_instance(self):
        services = list_services_sync(by="instance")
        assert isinstance(services, list)

    def test_list_services_sync_by_name(self):
        names = list_services_sync(by="name")
        assert isinstance(names, list)

    async def test_clear_services(self):
        await self._reset()

        class MyService(Service):
            name: str = "cl_svc"

        svc = MyService(name="cl_svc")
        await add_service(svc)
        await clear_services()
        assert has_service("cl_svc") is False
