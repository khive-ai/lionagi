# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import UUID

from pydantic import PrivateAttr, model_validator

from lionagi.protocols.generic.element import Element
from lionagi.beta.core.base.pile import Pile
from lionagi._errors import ExistsError, NotFoundError
from lionagi.ln.types import Operable
from .backend import Calling, Normalized, ResourceBackend, ResourceConfig
from .imodel import iModel

if TYPE_CHECKING:
    from lionagi.beta.session.context import RequestContext

logger = logging.getLogger(__name__)

__all__ = (
    "Normalized",
    "ResourceMeta",
    "Service",
    "ServiceCalling",
    "add_service",
    "clear_services",
    "get_resource_decl",
    "get_service",
    "has_service",
    "list_services",
    "list_services_sync",
    "remove_service",
    "resource",
)

_RESOURCE_ATTR = "_SERVICE_RESOURCE_META"


def _to_pascal(snake_name: str) -> str:
    return "".join(word.capitalize() for word in snake_name.split("_"))


@dataclass(frozen=True, slots=True)
class ResourceMeta:
    name: str
    op: Operable
    description: str | None = None
    inputs: frozenset[str] = frozenset()
    outputs: frozenset[str] = frozenset()
    pre_hooks: tuple[str, ...] = ()
    post_hooks: tuple[str, ...] = ()

    @property
    def options_type(self) -> type:
        return self.op.compose_structure(
            _to_pascal(self.name) + "Options", include=set(self.inputs)
        )

    @property
    def result_type(self) -> type:
        return self.op.compose_structure(
            _to_pascal(self.name) + "Result", include=set(self.outputs)
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description or self.name,
            "inputs": sorted(self.inputs) if self.inputs else [],
            "outputs": sorted(self.outputs) if self.outputs else [],
        }


@dataclass(frozen=True, slots=True)
class _ResourceDecl:
    """Intermediate declaration before Operable is known."""

    name: str
    description: str | None = None
    inputs: frozenset[str] = frozenset()
    outputs: frozenset[str] = frozenset()
    pre_hooks: tuple[str, ...] = ()
    post_hooks: tuple[str, ...] = ()

    def bind(self, op: Operable) -> ResourceMeta:
        return ResourceMeta(
            name=self.name,
            op=op,
            description=self.description,
            inputs=self.inputs,
            outputs=self.outputs,
            pre_hooks=self.pre_hooks,
            post_hooks=self.post_hooks,
        )


def resource(
    name: str,
    inputs: set[str] | None = None,
    outputs: set[str] | None = None,
    description: str | None = None,
    pre_hooks: list[str] | None = None,
    post_hooks: list[str] | None = None,
):
    def decorator(func: Callable) -> Callable:
        decl = _ResourceDecl(
            name=name,
            description=description,
            inputs=frozenset(inputs or set()),
            outputs=frozenset(outputs or set()),
            pre_hooks=tuple(pre_hooks or []),
            post_hooks=tuple(post_hooks or []),
        )
        setattr(func, _RESOURCE_ATTR, decl)
        return func

    return decorator


def get_resource_decl(handler: Callable) -> _ResourceDecl | None:
    return getattr(handler, _RESOURCE_ATTR, None)


class Service(Element):
    catalog: ClassVar[type | None] = None
    hooks: ClassVar[dict[str, Callable]] = {}
    name: str
    _registry: dict[str, tuple[Callable, ResourceMeta]] = PrivateAttr(default_factory=dict)

    @property
    def resources(self) -> frozenset[str]:
        return frozenset(sorted(self._registry.keys()))

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [meta.schema for _, meta in self._registry.values()]

    @model_validator(mode="after")
    def _populate_resources(self):
        all_specs = self._get_catalog_specs()

        for cls in reversed(type(self).mro()):
            for attr_name, attr in vars(cls).items():
                if attr_name.startswith("__"):
                    continue
                decl = get_resource_decl(attr)
                if decl is None:
                    continue
                method = getattr(self, attr_name)
                fields = decl.inputs | decl.outputs
                op = Operable([s for s in all_specs if s.name in fields])
                meta = decl.bind(op)
                self._registry[meta.name] = (method, meta)
        return self

    def _get_catalog_specs(self) -> list:
        if self.catalog is None:
            return []
        cat_op = Operable.from_structure(self.catalog)
        return list(cat_op.get_specs())

    async def _run_hooks(
        self,
        hook_names: tuple[str, ...],
        options: Any,
        ctx: RequestContext,
        result: Any = None,
    ) -> None:
        for hook_name in hook_names:
            hook_fn = self.hooks.get(hook_name)
            if hook_fn is None:
                continue
            try:
                await hook_fn(self, options, ctx, result)
            except Exception:
                logger.exception("Hook %r failed", hook_name)
                raise

    def create_imodel(self) -> iModel:
        return iModel(backend=self.create_backend())

    def create_backend(self, **kwargs: Any) -> ResourceBackend:
        return _ServiceBackend(service=self, **kwargs)

    async def call(
        self,
        name: str,
        options: Any,
        ctx: RequestContext,
    ) -> Normalized:
        if name not in self._registry:
            return Normalized(
                status="error",
                data=None,
                error=f"Unknown resource '{name}'. Available: {sorted(self._registry)}",
            )

        handler, meta = self._registry[name]

        if meta.pre_hooks:
            await self._run_hooks(meta.pre_hooks, options, ctx)

        policy_result = await self._evaluate_policy(meta, options, ctx)
        if not policy_result.get("allowed", True):
            return Normalized(
                status="error",
                data=None,
                error=f"Policy blocked {name}: {policy_result.get('reason', '')}",
            )

        if meta.inputs:
            try:
                data = options if isinstance(options, dict) else meta.op.dump_instance(options)
                options = meta.op.validate_instance(meta.options_type, data)
            except Exception as e:
                return Normalized(
                    status="error",
                    data=None,
                    error=f"Invalid options for '{name}': {e}",
                )

        try:
            output = await handler(options, ctx)
            normalized = Normalized(
                status="success",
                data=output,
                serialized=output if isinstance(output, dict) else {"result": output},
            )
        except PermissionError as e:
            normalized = Normalized(status="error", data=None, error=f"Permission: {e}")
        except Exception as e:
            logger.exception("Service %s.%s failed: %s", self.name, name, e)
            normalized = Normalized(status="error", data=None, error=str(e))

        if meta.post_hooks:
            await self._run_hooks(meta.post_hooks, options, ctx, result=normalized.data)

        return normalized

    async def stream(
        self,
        name: str,
        options: Any,
        ctx: RequestContext,
    ) -> AsyncIterable[Normalized]:
        result = await self.call(name, options, ctx)
        yield result  # type: ignore[misc]

    async def _evaluate_policy(
        self, meta: ResourceMeta, options: Any, ctx: RequestContext
    ) -> dict[str, Any]:
        return {"allowed": True}


class ServiceCalling(Calling):
    """Calling event for Service dispatch."""

    action: str = ""
    ctx: Any = None

    @property
    def call_args(self) -> dict[str, Any]:
        payload = dict(self.payload or {})
        action = self.action or payload.pop("action", payload.pop("name", ""))
        ctx = self.ctx or payload.pop("ctx", None)
        return {
            "name": action,
            "options": payload,
            "ctx": ctx,
        }


class _ServiceBackend(ResourceBackend):
    """Thin ResourceBackend wrapper around a Service for iModel compatibility."""

    config: ResourceConfig

    def __init__(self, service: Service, **kwargs: Any) -> None:
        kwargs.setdefault("provider", "service")
        config = ResourceConfig(name=service.name, **kwargs)
        super().__init__(config=config)
        self._service = service

    @property
    def event_type(self) -> type[Calling]:
        return ServiceCalling

    def create_payload(self, request: dict | None = None, **kwargs: Any) -> dict:
        payload = dict(request) if request else {}
        payload.update(kwargs)
        if "arguments" in payload and len(payload) == 1 and isinstance(payload["arguments"], dict):
            return payload["arguments"]
        return payload

    async def call(self, **kw: Any) -> Normalized:
        name = kw.get("name")
        options = kw.get("options")
        ctx = kw.get("ctx")
        return await self._service.call(name, options, ctx)


SERVICE_REGISTRY: Pile[Service] = Pile(item_type=Service)
SERVICE_NAME_MAP: dict[str, UUID] = {}

async def add_service(service: Service, update: bool = False) -> UUID:
    if service.name in SERVICE_NAME_MAP:
        if not update:
            raise ExistsError(f"Service with name '{service.name}' already exists in registry.")
        await remove_service(service.name)

    async with SERVICE_REGISTRY:
        SERVICE_REGISTRY.add(service)
        SERVICE_NAME_MAP[service.name] = service.id
    return service.id


def has_service(service: UUID | str) -> bool:
    if service in SERVICE_REGISTRY:
        return True
    return isinstance(service, str) and service in SERVICE_NAME_MAP

async def get_service(service: UUID | str) -> Service:
    async with SERVICE_REGISTRY:
        if service in SERVICE_REGISTRY:
            return SERVICE_REGISTRY[service]
        if service in SERVICE_NAME_MAP:
            service_id = SERVICE_NAME_MAP[service]
            return SERVICE_REGISTRY[service_id]
    raise NotFoundError(f"Service with id or name '{service}' not found in registry.")

async def remove_service(service: UUID | str) -> None:
    async with SERVICE_REGISTRY:
        if service in SERVICE_REGISTRY:
            removed = SERVICE_REGISTRY.remove(service)
            SERVICE_NAME_MAP.pop(removed.name, None)
            return
        if service in SERVICE_NAME_MAP:
            service_id = SERVICE_NAME_MAP.pop(service)
            SERVICE_REGISTRY.remove(service_id)
            return
    raise NotFoundError(f"Service with id or name '{service}' not found in registry.")

async def list_services(by: Literal["name", "instance"] = "instance") -> list[str | Service]:
    async with SERVICE_REGISTRY:
        if by == "name":
            return list(SERVICE_NAME_MAP.keys())
        return list(SERVICE_REGISTRY)
    return []


def list_services_sync(by: Literal["name", "instance"] = "instance") -> list[str | Service]:
    if by == "name":
        return list(SERVICE_NAME_MAP.keys())
    return list(SERVICE_REGISTRY)

async def clear_services():
    async with SERVICE_REGISTRY:
        SERVICE_REGISTRY.clear()
        SERVICE_NAME_MAP.clear()
