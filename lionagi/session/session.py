# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Session and Branch: multi-branch orchestration hub for messages, resources, and operations."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import inspect
import time
from collections.abc import AsyncGenerator, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr, field_serializer, model_validator

from lionagi._errors import NotFoundError
from lionagi.core.types import Capability, Principal
from lionagi.service.backend import Calling
from lionagi.service.imodel_v2 import iModel
from lionagi.service.resource_registry import ResourceRegistry
from lionagi.work.node import Operation
from lionagi.ln.types._sentinel import Unset, UnsetType, not_sentinel
from lionagi.models import HashableModel
from lionagi.protocols.types import Element, Flow, Message, Pile, Progression

__all__ = (
    "Branch",
    "OperationDecl",
    "OperationHandler",
    "OperationRegistry",
    "Session",
    "SessionConfig",
)

OperationHandler = Callable[..., Any]


def _status(message: str, style: str | None = None) -> None:
    print(f"[{style or 'info'}] {message}")


@contextlib.contextmanager
def _timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _status(f"{label} ({time.perf_counter() - start:.2f}s)", style="success")


@dataclass(frozen=True, slots=True)
class OperationDecl:
    """Registered operation handler plus Runner capability declaration."""

    handler: OperationHandler | None = None
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    required_rights: Callable[..., Iterable[str] | None] | None = None
    morphism: Any | None = None
    morphism_factory: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if (
            self.handler is None
            and self.morphism is None
            and self.morphism_factory is None
        ):
            raise ValueError(
                "OperationDecl requires a handler, morphism, or morphism_factory"
            )
        if self.morphism is not None and self.morphism_factory is not None:
            raise ValueError(
                "OperationDecl cannot define both morphism and morphism_factory"
            )
        object.__setattr__(self, "requires", frozenset(self.requires))
        object.__setattr__(self, "provides", frozenset(self.provides))

    def to_morphism(
        self,
        params: Any = None,
        *,
        operation: Any | None = None,
        name: str | None = None,
    ) -> Any:
        from lionagi.core.morphism import MorphismAdapter

        if self.handler is not None:
            return MorphismAdapter.from_operation(self, params, operation=operation)

        source = (
            self.morphism_factory(params) if self.morphism_factory else self.morphism
        )
        return _coerce_morphism(
            source,
            name=name,
            requires=self.requires,
            provides=self.provides,
        )


def _coerce_morphism(
    source: Any,
    *,
    name: str | None = None,
    requires: frozenset[str] | set[str] = frozenset(),
    provides: frozenset[str] | set[str] = frozenset(),
) -> Any:
    from lionagi.core.morphism import MorphismAdapter

    if source is None:
        raise TypeError("morphism source cannot be None")

    reqs = frozenset(requires or getattr(source, "requires", frozenset()))
    provs = frozenset(provides or getattr(source, "provides", frozenset()))
    morph_name = (
        name or getattr(source, "name", None) or getattr(source, "__name__", "morphism")
    )

    apply = getattr(source, "apply", None)
    if callable(apply):
        return _MorphismProxy(source, morph_name, reqs, provs)

    if callable(source):
        return MorphismAdapter.wrap(
            source,
            name=morph_name,
            requires=reqs,
            provides=provs,
        )

    raise TypeError(
        "morphism source must be a Morphism, Morphism-like object, or callable"
    )


class _MorphismProxy:

    def __init__(
        self,
        source: Any,
        name: str,
        requires: frozenset[str],
        provides: frozenset[str],
    ) -> None:
        self._source = source
        self.name = name
        self.requires = requires
        self.provides = provides

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    async def pre(self, br: Principal, **kw: Any) -> bool:
        fn = getattr(self._source, "pre", None)
        if not callable(fn):
            return True
        result = fn(br, **kw)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def apply(self, br: Principal, **kw: Any) -> dict[str, Any]:
        result = self._source.apply(br, **kw)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def post(self, br: Principal, result: dict[str, Any]) -> bool:
        fn = getattr(self._source, "post", None)
        if not callable(fn):
            return True
        ok = fn(br, result)
        if inspect.isawaitable(ok):
            ok = await ok
        return bool(ok)


class OperationRegistry:

    def __init__(self) -> None:
        self._decls: dict[str, OperationDecl] = {}

    def register(
        self,
        operation_name: str,
        handler: OperationHandler | OperationDecl,
        *,
        requires: frozenset[str] | set[str] = frozenset(),
        provides: frozenset[str] | set[str] = frozenset(),
        required_rights: Callable[..., Iterable[str] | None] | None = None,
        override: bool = False,
        update: bool | None = None,
    ) -> None:
        if update is not None:
            override = update
        if operation_name in self._decls and not override:
            raise ValueError(
                f"Operation '{operation_name}' already registered. Use override=True to replace."
            )

        if isinstance(handler, OperationDecl):
            decl = handler
        else:
            decl = OperationDecl(
                handler=handler,
                requires=frozenset(requires),
                provides=frozenset(provides),
                required_rights=required_rights,
            )
        self._decls[operation_name] = decl

    def register_morphism(
        self,
        operation_name: str,
        morphism: Any,
        *,
        requires: frozenset[str] | set[str] | None = None,
        provides: frozenset[str] | set[str] | None = None,
        factory: bool = False,
        override: bool = False,
        update: bool | None = None,
    ) -> None:
        if update is not None:
            override = update
        if operation_name in self._decls and not override:
            raise ValueError(
                f"Operation '{operation_name}' already registered. Use override=True to replace."
            )
        reqs = frozenset(
            requires
            if requires is not None
            else getattr(morphism, "requires", frozenset())
        )
        provs = frozenset(
            provides
            if provides is not None
            else getattr(morphism, "provides", frozenset())
        )
        self._decls[operation_name] = OperationDecl(
            requires=reqs,
            provides=provs,
            morphism_factory=morphism if factory else None,
            morphism=None if factory else morphism,
        )

    def get_decl(self, operation_name: str) -> OperationDecl:
        if operation_name not in self._decls:
            raise KeyError(
                f"Operation '{operation_name}' not registered. Available: {self.list_names()}"
            )
        return self._decls[operation_name]

    def get(self, operation_name: str) -> OperationHandler:
        handler = self.get_decl(operation_name).handler
        if handler is None:
            raise TypeError(f"Operation '{operation_name}' is registered as a morphism")
        return handler

    def has(self, operation_name: str) -> bool:
        return operation_name in self._decls

    def unregister(
        self,
        operation_name: str,
        *,
        func: OperationHandler | None = None,
    ) -> bool:
        if operation_name not in self._decls:
            return False
        if func is not None and self._decls[operation_name].handler is not func:
            return False
        del self._decls[operation_name]
        return True

    def list_names(self) -> list[str]:
        return list(self._decls.keys())

    def __contains__(self, operation_name: str) -> bool:
        return operation_name in self._decls

    def __len__(self) -> int:
        return len(self._decls)

    def __repr__(self) -> str:
        return f"OperationRegistry(operations={self.list_names()})"


class Branch(Progression):
    """Message progression with session binding and capability/resource access control."""

    session_id: UUID = Field(..., frozen=True)
    principal: Principal = Field(default_factory=Principal)

    _scratchpad: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _sync_principal_id(self) -> Branch:
        """Capability subjects are keyed by principal.id; it must equal branch.id."""
        if self.principal.id != self.id:
            self.principal = self._principal_with_rights(
                self.principal,
                set(self.principal.rights()),
                principal_id=self.id,
                name=self.name,
            )
        return self

    @staticmethod
    def _resource_to_right(resource: str) -> str:
        if resource in {"*", "*:*"}:
            return "service.call"
        return (
            resource
            if resource.startswith("service.call")
            else f"service.call:{resource}"
        )

    @staticmethod
    def _coerce_principal(value: Any, name: str | None = None) -> Principal:
        if isinstance(value, Principal):
            return value
        if value is None:
            return Principal(name=name or "default")
        return Principal.model_validate(value)

    @staticmethod
    def _principal_with_rights(
        principal: Principal,
        rights: set[str],
        *,
        principal_id: UUID | None = None,
        name: str | None = None,
    ) -> Principal:
        subject = principal_id or principal.id
        data = principal.model_dump()
        data["id"] = subject
        data["name"] = name or principal.name
        data["caps"] = (
            [Capability(subject=subject, rights=frozenset(rights))] if rights else []
        )
        return Principal(**data)

    @property
    def capabilities(self) -> set[str]:
        return {
            r
            for r in self.principal.rights()
            if r != "service.call" and not r.startswith("service.call:")
        }

    @property
    def resources(self) -> set[str]:
        resources = {
            r.split(":", 1)[1]
            for r in self.principal.rights()
            if r.startswith("service.call:")
        }
        if "service.call" in self.principal.rights():
            resources.add("*:*")
        return resources

    @property
    def scratchpad(self) -> Any:
        """Lazy-initialized Note store for cross-round lvar state; supports nested key paths."""
        if self._scratchpad is None:
            from lionagi.models.note import Note

            self._scratchpad = Note()
        return self._scratchpad

    def scratchpad_summary(self) -> dict[str, str] | None:
        """Top-level scratchpad values stringified for prompt injection; None when empty."""
        from lionagi.libs.schema import minimal_yaml

        if self._scratchpad is None or len(self._scratchpad) == 0:
            return None

        content = (
            self._scratchpad.content
            if hasattr(self._scratchpad, "content")
            else self._scratchpad
        )
        rendered: dict[str, str] = {}
        for k, v in content.items():
            if isinstance(v, str):
                rendered[k] = v
            elif isinstance(v, (dict, list)):
                rendered[k] = minimal_yaml(v)
            else:
                rendered[k] = str(v)
        return rendered

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Branch(messages={len(self)}, session={str(self.session_id)[:8]}{name_str})"


class SessionConfig(HashableModel):
    default_branch_name: str | None = None
    shared_capabilities: set[str] = Field(default_factory=set)
    shared_resources: set[str] = Field(default_factory=set)
    default_gen_model: str | None = None
    default_parse_model: str | None = None
    auto_create_default_branch: bool = True

    system_prefix: str | None = Field(
        default=None,
        description="System prompt prefix prepended to every branch's system message. "
        "Use for framework-level instructions (e.g., LNDL output format).",
    )

    aggregate_actions: bool = Field(
        default=False,
        description="When True, action request/response pairs are aggregated into "
        "compact round-level summaries instead of individual full renders.",
    )
    round_notifications: bool = Field(
        default=False,
        description="When True, inject system notification blocks between rounds "
        "with tools, context size, and scratchpad state for agent grounding.",
    )

    # Logging configuration
    log_persist_dir: str | Path | None = Field(
        default=None,
        description="Directory for session dumps. None disables logging.",
    )
    log_auto_save_on_exit: bool = Field(
        default=True,
        description="Register atexit handler on Session creation.",
    )

    @property
    def logging_enabled(self) -> bool:
        """True if logging is configured (log_persist_dir is set)."""
        return self.log_persist_dir is not None


class Session(Element):
    user: str | None = None
    communications: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message)
    )
    resources: ResourceRegistry = Field(default_factory=ResourceRegistry, exclude=True)
    operations: OperationRegistry = Field(
        default_factory=OperationRegistry, exclude=True
    )
    config: SessionConfig = Field(default_factory=SessionConfig)
    default_branch_id: UUID | None = None

    _registered_atexit: bool = PrivateAttr(default=False)
    _dump_count: int = PrivateAttr(default=0)
    _runner: Any = PrivateAttr(default=None)

    @field_serializer("communications")
    def _serialize_communications(self, flow: Flow) -> dict:
        return flow.to_dict(mode="json")

    @model_validator(mode="after")
    def _validate_default_branch(self) -> Session:
        if self.config.auto_create_default_branch and self.default_branch is None:
            default_branch_name = self.config.default_branch_name or "main"
            self.create_branch(
                name=default_branch_name,
                capabilities=self.config.shared_capabilities,
                resources=self.config.shared_resources,
            )
            self.set_default_branch(default_branch_name)

        if (
            self.config.logging_enabled
            and self.config.log_auto_save_on_exit
            and not self._registered_atexit
        ):
            atexit.register(self._save_at_exit)
            self._registered_atexit = True

        # Lazy import avoids circular dependency with operations module.
        try:
            from lionagi.operations import builtin_operation_declarations

            for name, decl in builtin_operation_declarations().items():
                if not self.operations.has(name):
                    self.operations.register(name, decl)
        except ImportError:
            pass

        return self

    @property
    def default_gen_model(self) -> iModel | None:
        if self.config.default_gen_model is None:
            return None
        return self.resources.get(self.config.default_gen_model)

    @property
    def default_parse_model(self) -> iModel | None:
        if self.config.default_parse_model is None:
            return None
        return self.resources.get(self.config.default_parse_model)

    @property
    def messages(self) -> Pile[Message]:
        return self.communications.items

    @property
    def branches(self) -> Pile[Branch]:
        return self.communications.progressions

    @property
    def default_branch(self) -> Branch | None:
        if self.default_branch_id is None:
            return None
        with contextlib.suppress(KeyError, NotFoundError):
            return self.communications.get_progression(self.default_branch_id)
        return None

    def create_branch(
        self,
        *,
        name: str | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
        principal: Principal | None = None,
        messages: Iterable[UUID | Message] | None = None,
    ) -> Branch:
        if name:
            from .constraints import branch_name_must_be_unique

            branch_name_must_be_unique(self, name)

        order: list[UUID] = []
        if messages:
            order.extend([self._coerce_id(msg) for msg in messages])

        branch_name = name or f"branch_{len(self.branches)}"

        if principal is None:
            rights: set[str] = set()
            for cap in capabilities or set():
                rights.add(str(cap))
            for res in resources or set():
                rights.add(Branch._resource_to_right(str(res)))
            principal = Branch._principal_with_rights(
                Principal(name=branch_name),
                rights,
            )

        branch = Branch(
            session_id=self.id,
            name=branch_name,
            principal=principal,
            order=order,
        )

        self.communications.add_progression(branch)
        return branch

    def get_branch(
        self, branch: UUID | str | Branch, default: Branch | UnsetType = Unset, /
    ) -> Branch:
        if isinstance(branch, Branch) and branch in self.branches:
            return branch
        with contextlib.suppress(KeyError):
            return self.communications.get_progression(branch)
        if not_sentinel(default):
            return default
        raise NotFoundError("Branch not found")

    def set_default_branch(self, branch: Branch | UUID | str) -> None:
        resolved = self.get_branch(branch)
        self.default_branch_id = resolved.id

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        capabilities: set[str] | Literal[True] | None = None,
        resources: set[str] | Literal[True] | None = None,
    ) -> Branch:
        """Fork a branch for divergent exploration; pass True to inherit access control."""
        source = self.get_branch(branch)

        forked = self.create_branch(
            name=name or f"{source.name}_fork",
            messages=source.order,
            capabilities=(
                {*source.capabilities}
                if capabilities is True
                else (capabilities or set())
            ),
            resources=(
                {*source.resources} if resources is True else (resources or set())
            ),
        )

        forked.metadata["forked_from"] = {
            "branch_id": str(source.id),
            "branch_name": source.name,
            "created_at": source.created_at,
            "message_count": len(source),
        }
        return forked

    def add_message(
        self,
        message: Message,
        branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
    ) -> None:
        self.communications.add_item(message, progressions=branches)

    async def request(
        self,
        name: str,
        /,
        branch: Branch | UUID | str | None = None,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **options,
    ) -> Calling:
        if branch is not None:
            resolved_branch = self.get_branch(branch)

            from .constraints import resource_must_be_accessible

            resource_must_be_accessible(resolved_branch, name)

        resource = self.resources.get(name)
        return await resource.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **options,
        )

    def _get_runner(self):
        if self._runner is None:
            from lionagi.core.runner import Runner

            self._runner = Runner()
        return self._runner

    @staticmethod
    def _unwrap_runner_result(result: dict[str, Any]) -> Any:
        if set(result.keys()) == {"result"}:
            return result["result"]
        return result

    @staticmethod
    def _principal_for_branch(branch: Branch):
        return branch.principal.model_copy(deep=True)

    @staticmethod
    def _operation_node_params(
        op: Operation,
        session: Session,
        branch: Branch | None,
        *,
        verbose: bool,
    ) -> dict[str, Any]:
        branch_ref = None if branch is None else branch.name or str(branch.id)
        return {
            "_lionagi_operation": op,
            "_lionagi_session": session,
            "_lionagi_branch": branch,
            "_lionagi_branch_ref": branch_ref,
            "_lionagi_operation_type": op.operation_type,
            "_lionagi_operation_name": str(op.metadata.get("name", op.id)),
            "_lionagi_verbose": verbose,
        }

    @staticmethod
    def _morphism_node_params(
        params: Any | None,
        session: Session,
        branch: Branch | None,
        *,
        name: str,
        verbose: bool,
    ) -> dict[str, Any]:
        if params is None:
            node_params: dict[str, Any] = {}
        elif isinstance(params, dict):
            node_params = dict(params)
        else:
            node_params = {"params": params}
        branch_ref = None if branch is None else branch.name or str(branch.id)
        node_params.update(
            {
                "_lionagi_session": session,
                "_lionagi_branch": branch,
                "_lionagi_branch_ref": branch_ref,
                "_lionagi_operation_type": name,
                "_lionagi_operation_name": name,
                "_lionagi_verbose": verbose,
            }
        )
        return node_params

    def _morphism_for_operation(self, op: Operation) -> Any:
        if getattr(op, "morphism", None) is not None:
            return _coerce_morphism(
                op.morphism,
                name=op.operation_type,
            )
        decl = self.operations.get_decl(op.operation_type)
        return decl.to_morphism(op.parameters, operation=op, name=op.operation_type)

    def _operation_uses_direct_morphism(self, op: Operation) -> bool:
        if getattr(op, "morphism", None) is not None:
            return True
        with contextlib.suppress(KeyError):
            return self.operations.get_decl(op.operation_type).handler is None
        return False

    async def _execute_operation_handler(
        self,
        op: Operation,
        branch: Branch,
        *,
        verbose: bool | None = None,
    ) -> Any:
        from lionagi.core.graph import OpGraph, OpNode

        use_verbose = op._verbose if verbose is None else verbose
        node_params = (
            self._morphism_node_params(
                op.parameters,
                self,
                branch,
                name=op.operation_type,
                verbose=use_verbose,
            )
            if self._operation_uses_direct_morphism(op)
            else self._operation_node_params(
                op,
                self,
                branch,
                verbose=use_verbose,
            )
        )
        node = OpNode(
            id=op.id,
            m=self._morphism_for_operation(op),
            params=node_params,
            control=op.is_control,
        )
        graph = OpGraph(nodes={node.id: node}, roots={node.id})
        results = await self._get_runner().run(
            self._principal_for_branch(branch),
            graph,
        )
        return self._unwrap_runner_result(results[node.id])

    async def _stream_operation_handler(
        self,
        op: Operation,
        branch: Branch,
        *,
        verbose: bool,
    ) -> AsyncGenerator[Any, None]:
        from lionagi.core.graph import OpGraph, OpNode
        from lionagi.core.morphism import MorphismAdapter

        decl = self.operations.get_decl(op.operation_type)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()

        async def _apply(br: Any, **_kw: Any) -> dict[str, Any]:
            ctx = op.make_context(self, branch, verbose=verbose, principal=br)
            chunks: list[Any] = []
            result_or_stream = decl.handler(op.parameters, ctx)
            if hasattr(result_or_stream, "__aiter__"):
                async for chunk in result_or_stream:
                    chunks.append(chunk)
                    await queue.put(chunk)
                return {"result": chunks}
            if inspect.isawaitable(result_or_stream):
                result = await result_or_stream
                chunks.append(result)
                await queue.put(result)
                return {"result": chunks}
            raise TypeError(
                f"Streaming operation '{op.operation_type}' returned "
                f"{type(result_or_stream).__name__}, expected async iterator or awaitable"
            )

        node = OpNode(
            id=op.id,
            m=MorphismAdapter.wrap(
                _apply,
                name=f"ops.{decl.handler.__name__}.stream",
                requires=decl.requires,
                provides=decl.provides,
            ),
            params=self._operation_node_params(op, self, branch, verbose=verbose),
            control=op.is_control,
        )
        graph = OpGraph(nodes={node.id: node}, roots={node.id})

        async def _run_graph() -> dict[UUID, dict[str, Any]]:
            try:
                return await self._get_runner().run(
                    self._principal_for_branch(branch),
                    graph,
                )
            finally:
                await queue.put(done)

        task = asyncio.create_task(_run_graph())
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    async def conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        params: Any | None = None,
        verbose: bool = False,
    ) -> Operation:
        resolved = self._resolve_branch(branch)

        if verbose:
            branch_name = resolved.name or str(resolved.id)[:8]
            _status(
                f"conduct({operation_type}) on branch={branch_name}",
                style="info",
            )

        op = Operation(
            operation_type=operation_type,
            parameters=params,
        )
        op._verbose = verbose
        op.bind(self, resolved)

        if verbose:
            with _timer(f"{operation_type} completed"):
                with contextlib.suppress(Exception):
                    await op.invoke()

            resp = op.execution.response
            if op.execution.error:
                _status(f"ERROR: {op.execution.error}", style="error")
            elif isinstance(resp, str):
                _status(f"response: {len(resp)} chars", style="success")
            else:
                _status(f"response: {type(resp).__name__}", style="success")
        else:
            with contextlib.suppress(Exception):
                await op.invoke()

        return op

    async def stream_conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        params: Any | None = None,
        verbose: bool = False,
    ) -> AsyncGenerator[Any, None]:
        resolved = self._resolve_branch(branch)
        op = Operation(
            operation_type=operation_type,
            parameters=params,
        )
        op._verbose = verbose
        op.bind(self, resolved)

        if verbose:
            branch_name = resolved.name or str(resolved.id)[:8]
            _status(
                f"stream_conduct({operation_type}) on branch={branch_name}",
                style="info",
            )

        async for result in self._stream_operation_handler(
            op,
            resolved,
            verbose=verbose,
        ):
            yield result

    def register_operation(
        self,
        operation_name: str,
        handler: OperationHandler,
        *,
        requires: frozenset[str] | set[str] = frozenset(),
        provides: frozenset[str] | set[str] = frozenset(),
        required_rights: Callable[..., Iterable[str] | None] | None = None,
        update: bool = True,
        override: bool | None = None,
    ) -> None:
        self.operations.register(
            operation_name,
            handler,
            requires=frozenset(requires),
            provides=frozenset(provides),
            required_rights=required_rights,
            override=update if override is None else override,
        )

    def register_morphism(
        self,
        operation_name: str,
        morphism: Any,
        *,
        requires: frozenset[str] | set[str] | None = None,
        provides: frozenset[str] | set[str] | None = None,
        factory: bool = False,
        update: bool = True,
        override: bool | None = None,
    ) -> None:
        """Register a Runner-native Morphism; it executes as OpNode.m without a RequestContext."""
        self.operations.register_morphism(
            operation_name,
            morphism,
            requires=requires,
            provides=provides,
            factory=factory,
            override=update if override is None else override,
        )

    async def run_morphism(
        self,
        morphism: Any,
        *,
        branch: Branch | UUID | str | None = None,
        params: dict[str, Any] | Any | None = None,
        context: dict[str, Any] | None = None,
        name: str | None = None,
        requires: frozenset[str] | set[str] = frozenset(),
        provides: frozenset[str] | set[str] = frozenset(),
        control: bool = False,
        max_concurrent: int | None = None,
        verbose: bool = False,
    ) -> Any:
        from lionagi.core.graph import OpGraph, OpNode
        from lionagi.core.runner import Runner

        resolved = self._resolve_branch(branch)
        if isinstance(morphism, str):
            decl = self.operations.get_decl(morphism)
            m = decl.to_morphism(params, name=name or morphism)
            morph_name = name or morphism
        else:
            morph_name = name or getattr(morphism, "name", None) or "morphism"
            m = _coerce_morphism(
                morphism,
                name=morph_name,
                requires=requires,
                provides=provides,
            )

        principal = self._principal_for_branch(resolved)
        if context:
            principal.ctx.update(context)

        node = OpNode(
            m=m,
            params=self._morphism_node_params(
                params,
                self,
                resolved,
                name=morph_name,
                verbose=verbose,
            ),
            control=control,
        )
        graph = OpGraph(nodes={node.id: node}, roots={node.id})
        runner = (
            self._get_runner()
            if max_concurrent is None
            else Runner(max_concurrent=max_concurrent)
        )
        results = await runner.run(principal, graph)
        return self._unwrap_runner_result(results[node.id])

    def unregister_operation(
        self,
        operation_name: str,
        *,
        func: OperationHandler | None = None,
    ) -> bool:
        return self.operations.unregister(operation_name, func=func)

    async def flow(
        self,
        graph: Any,
        *,
        branch: Branch | UUID | str | None = None,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from lionagi.work.flow import flow

        return await flow(
            self,
            graph,
            branch=branch,
            max_concurrent=max_concurrent,
            stop_on_error=stop_on_error,
            verbose=verbose,
            context=context,
        )

    async def flow_stream(
        self,
        graph: Any,
        *,
        branch: Branch | UUID | str | None = None,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Any, None]:
        from lionagi.work.flow import flow_stream

        async for result in flow_stream(
            self,
            graph,
            branch=branch,
            max_concurrent=max_concurrent,
            stop_on_error=stop_on_error,
            context=context,
        ):
            yield result

    def dump(self, clear: bool = False) -> Path | None:
        """Serialize session to JSON; resources and operations are excluded and must be re-registered on restore."""
        from lionagi.ln._json_dump import json_dumpb
        from lionagi.ln._utils import create_path

        if not self.config.logging_enabled or len(self.messages) == 0:
            return None

        self._dump_count += 1

        filepath = create_path(
            directory=self.config.log_persist_dir,
            filename=str(self.id)[:8],
            extension=".json",
            timestamp=True,
            time_prefix=True,
            timestamp_format="%Y%m%d_%H%M%S",
            random_hash_digits=4,
            file_exist_ok=True,
        )

        data = json_dumpb(self.to_dict(mode="json"), safe_fallback=True)
        std_path = Path(filepath)
        std_path.write_bytes(data)

        if clear:
            self.communications.clear()

        return std_path

    async def adump(self, clear: bool = False) -> Path | None:
        """Async variant of dump; resources and operations are excluded and must be re-registered on restore."""
        from lionagi.ln._json_dump import json_dumpb
        from lionagi.ln._utils import acreate_path

        if not self.config.logging_enabled or len(self.messages) == 0:
            return None

        async with self.messages:
            self._dump_count += 1

            filepath = await acreate_path(
                directory=self.config.log_persist_dir,
                filename=str(self.id)[:8],
                extension=".json",
                timestamp=True,
                time_prefix=True,
                timestamp_format="%Y%m%d_%H%M%S",
                random_hash_digits=4,
                file_exist_ok=True,
            )

            data = json_dumpb(self.to_dict(mode="json"), safe_fallback=True)
            await filepath.write_bytes(data)

            if clear:
                self.communications.clear()

        return Path(filepath)

    def _save_at_exit(self) -> None:
        """atexit callback. Dumps session synchronously. Errors are suppressed."""
        if len(self.messages) > 0:
            with contextlib.suppress(Exception):
                self.dump(clear=False)

    def _resolve_branch(self, branch: Branch | UUID | str | None) -> Branch:
        """Resolve to Branch, falling back to default. Raises if neither available."""
        if branch is not None:
            return self.get_branch(branch)
        if self.default_branch is not None:
            return self.default_branch
        raise RuntimeError("No branch provided and no default branch set")

    def __repr__(self) -> str:
        return (
            f"Session(messages={len(self.messages)}, "
            f"branches={len(self.branches)}, "
            f"resources={len(self.resources)})"
        )
