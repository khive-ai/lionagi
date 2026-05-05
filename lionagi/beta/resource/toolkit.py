# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""ToolKit — Service for agent-runtime tool execution.

ToolKit is the single class for defining, enforcing, and exposing tools
to agents. It combines:
- @tool_action decorated methods (handler discovery)
- Policy enforcement (path, process, capability checks)
- Service registry integration (register globally, invoke with RequestContext)

Usage — subclass::

    class FileKit(ToolKit):
        @tool_action(name="read", requires={"fs.read:/workspace/*"})
        async def read(self, args, ctx):
            return {"content": Path(args["path"]).read_text()}

    kit = FileKit(config=ToolKitConfig(
        name="file", provider="local",
        path_policy=PathGuard(root=Path.cwd()),
    ))
    await add_service(kit)

Usage — from handlers::

    kit = ToolKit.from_handlers({
        "search": search_fn,
        "count": count_fn,
    }, name="utils")

ToolKit vs Endpoint:
    Endpoint is a ResourceBackend transport adapter.
    ToolKit is a Service for agent-runtime tools with policy enforcement.
    Use ``create_backend()`` or ``create_imodel()`` when a ToolKit must be
    invoked through the ResourceBackend/iModel path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from lionagi.beta.resource.backend import Normalized, ResourceConfig
from lionagi.beta.resource.service import Service, add_service, list_services_sync
from lionagi.ln.types._compat import StrEnum

from lionagi.tools.sandbox import PathGuard, ProcessGuard

if TYPE_CHECKING:
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class ToolPolicyEvaluator(Protocol):
        async def evaluate(self, action: str, args: dict, ctx: Any) -> ToolPolicyResult: ...


__all__ = (
    "ToolActionMeta",
    "ToolEnforcement",
    "ToolKit",
    "ToolKitConfig",
    "ToolPolicyResult",
    "list_toolkits",
    "register_toolkit",
    "get_tool_action_meta",
    "tool_action",
)


# ─── Action metadata + decorator (was tools/enforcement.py) ──────────

_TOOL_ACTION_ATTR = "_tool_action"


class ToolEnforcement(StrEnum):
    """Policy enforcement level for a tool action."""

    HARD = "hard"
    SOFT = "soft"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class ToolActionMeta:
    """Metadata for a tool action handler.

    Action-level invocation gating is enforced through the bound branch
    Principal using ``lionagi.beta.core.policy.policy_check``. ``requires``
    declares additional capabilities beyond the service-call right.

    Attributes:
        name: Action identifier (e.g., "read", "write", "grep").
        input_schema: Pydantic model for input validation.
        output_schema: Pydantic model for output validation + LLM schema rendering.
        pre_hooks: Hook names to run before action.
        post_hooks: Hook names to run after action.
    """

    name: str
    input_schema: type | None = None
    output_schema: type | None = None
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    pre_hooks: tuple[str, ...] = ()
    post_hooks: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolPolicyResult:
    """Result of a toolkit-level policy check (hooks-based)."""

    allowed: bool
    enforcement: ToolEnforcement = ToolEnforcement.HARD
    reason: str = ""
    action: str = ""


def tool_action(
    name: str,
    input_schema: type | None = None,
    output_schema: type | None = None,
    requires: set[str] | frozenset[str] | None = None,
    provides: set[str] | frozenset[str] | None = None,
    capabilities: set[str] | frozenset[str] | None = None,
    pre_hooks: list[str] | None = None,
    post_hooks: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator declaring tool action metadata.

    Usage::

        @tool_action(name="read", input_schema=ReadInput, output_schema=ReadOutput)
        async def read(self, args, ctx):
            ...
    """

    def decorator(func: Callable) -> Callable:
        meta = ToolActionMeta(
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            requires=frozenset(requires or capabilities or frozenset()),
            provides=frozenset(provides or frozenset()),
            pre_hooks=tuple(pre_hooks or []),
            post_hooks=tuple(post_hooks or []),
        )
        setattr(func, _TOOL_ACTION_ATTR, meta)
        return func

    return decorator


def get_tool_action_meta(handler: Callable) -> ToolActionMeta | None:
    """Get action metadata from a handler method (or None if undecorated)."""
    return getattr(handler, _TOOL_ACTION_ATTR, None)


logger = logging.getLogger(__name__)


class ToolKitConfig(ResourceConfig):
    """Configuration for a ToolKit.

    Carries local effect policies, hooks, and an optional contextual policy
    evaluator alongside standard ResourceConfig fields. Invocation authority
    is always checked through ``lionagi.beta.core.policy.policy_check`` when
    a RequestContext is bound to a branch.
    """

    path_policy: PathGuard | None = Field(default=None, exclude=True)
    process_policy: ProcessGuard | None = Field(default=None, exclude=True)
    hooks: dict[str, Callable] = Field(default_factory=dict, exclude=True)
    fail_open: bool = False
    policy_evaluator: Any = Field(default=None, exclude=True)
    use_policies: bool = True


class ToolKit(Service):
    """Agent-runtime tool service with policy enforcement.

    Subclass with @tool_action methods, or use from_handlers() for dynamic
    construction. Register globally via ``register_toolkit()`` or
    ``lionagi.beta.resource.service.add_service()``.
    """

    config: ToolKitConfig = Field(..., description="ToolKit configuration")
    _action_registry: dict[str, tuple[Callable, ToolActionMeta]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _derive_service_name(cls, data: Any) -> Any:
        """Mirror config.name onto Service.name for global registration."""
        if not isinstance(data, dict) or data.get("name") is not None:
            return data
        config = data.get("config")
        if isinstance(config, ToolKitConfig):
            data = dict(data)
            data["name"] = config.name
        elif isinstance(config, dict) and config.get("name") is not None:
            data = dict(data)
            data["name"] = config["name"]
        return data

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._discover_actions()

    def _discover_actions(self) -> None:
        """Scan for @tool_action decorated methods and register them."""
        for cls in reversed(type(self).mro()):
            for attr_name, attr in vars(cls).items():
                if attr_name.startswith("_"):
                    continue
                meta = get_tool_action_meta(attr)
                if meta is None:
                    continue
                method = getattr(self, attr_name)
                self._action_registry[meta.name] = (method, meta)

    @property
    def actions(self) -> list[str]:
        return sorted(self._action_registry.keys())

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def version(self) -> str | None:
        return self.config.version

    @property
    def tags(self) -> set[str]:
        return set(self.config.tags) if self.config.tags else set()

    @property
    def request_options(self) -> type[BaseModel] | None:
        return self.config.request_options

    @property
    def resources(self) -> frozenset[str]:
        return frozenset(self.actions)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self.tool_schemas(fmt="dict")  # type: ignore[return-value]

    def create_payload(self, request: dict | BaseModel | None = None, **kwargs: Any) -> dict:
        """Build payload for iModel pipeline.

        Unwraps the iModel invoke convention where kwargs arrive as
        {"arguments": {...}}. The payload becomes the inner dict directly,
        so ToolCalling.call_args passes it cleanly to call().
        """
        if request is None:
            payload = kwargs
        elif isinstance(request, BaseModel):
            payload = request.model_dump(exclude_none=True)
        else:
            payload = dict(request)
            payload.update(kwargs)

        if "arguments" in payload and len(payload) == 1 and isinstance(payload["arguments"], dict):
            return payload["arguments"]
        return payload

    async def _principal_allows(
        self,
        action: str,
        meta: ToolActionMeta,
        ctx: Any,
    ) -> ToolPolicyResult:
        branch = None
        get_branch = getattr(ctx, "get_branch", None)
        if callable(get_branch):
            branch = await get_branch()
        if branch is None:
            branch = getattr(ctx, "_bound_branch", None)
        if branch is None:
            return ToolPolicyResult(allowed=True, action=action)

        principal = getattr(branch, "principal", None)
        if principal is None:
            return ToolPolicyResult(
                allowed=False,
                action=action,
                reason="bound branch has no Principal",
            )

        from lionagi.beta.core.policy import policy_check

        scope_req = f"service.call:{self.name}:{action}"
        bare_req = f"service.call:{self.name}"
        scope_allowed = policy_check(
            principal,
            None,
            override_reqs={scope_req},
        ) or (
            len(self.actions) == 1
            and policy_check(
                principal,
                None,
                override_reqs={bare_req},
            )
        )
        if not scope_allowed:
            return ToolPolicyResult(
                allowed=False,
                action=action,
                reason=f"missing capability: {scope_req}",
            )

        if meta.requires and not policy_check(
            principal,
            None,
            override_reqs=set(meta.requires),
        ):
            missing = [
                req
                for req in sorted(meta.requires)
                if not policy_check(principal, None, override_reqs={req})
            ]
            return ToolPolicyResult(
                allowed=False,
                action=action,
                reason=f"missing capabilities: {missing}",
            )

        return ToolPolicyResult(allowed=True, action=action)

    async def call(
        self,
        name: str | None = None,
        options: dict[str, Any] | BaseModel | None = None,
        ctx: Any = None,
        arguments: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Normalized:
        """Execute a tool action with policy enforcement.

        Routes to the correct handler by service action name or by an
        ``action`` key in arguments/options.
        Single-action toolkits auto-select the only available action.
        """
        if arguments is not None:
            payload = dict(arguments)
        elif isinstance(options, BaseModel):
            payload = options.model_dump(exclude_none=True)
        elif options is None:
            payload = {}
        else:
            payload = dict(options)
        payload.update(kw)

        action = name or payload.pop("action", None)

        if action is None:
            if len(self._action_registry) == 1:
                action = next(iter(self._action_registry))
            else:
                return Normalized(
                    status="error",
                    data=None,
                    error=f"No action specified. Available: {self.actions}",
                )

        if action not in self._action_registry:
            return Normalized(
                status="error",
                data=None,
                error=f"Unknown action '{action}'. Available: {self.actions}",
            )

        handler, meta = self._action_registry[action]

        from lionagi.beta.session.context import RequestContext

        ctx = ctx or RequestContext(name=f"{self.name}:{action}", service=self.name)

        # Pre-hooks
        payload = await self._run_hooks(meta.pre_hooks, payload, ctx)

        # Policy check
        result = await self._evaluate_policy(meta, payload, ctx)
        if not result.allowed:
            if result.enforcement in (ToolEnforcement.HARD, ToolEnforcement.SOFT):
                return Normalized(
                    status="error",
                    data=None,
                    error=f"Policy blocked {action}: {result.reason}",
                )
            logger.warning("Policy advisory for %s: %s", action, result.reason)

        # Execute
        try:
            output = await handler(payload, ctx)
            normalized = Normalized(
                status="success",
                data=output,
                serialized=output if isinstance(output, dict) else {"result": output},
            )
        except PermissionError as e:
            normalized = Normalized(status="error", data=None, error=f"Permission: {e}")
        except Exception as e:
            logger.exception("ToolKit %s.%s failed: %s", self.name, action, e)
            normalized = Normalized(status="error", data=None, error=str(e))

        # Post-hooks
        await self._run_hooks(meta.post_hooks, payload, ctx, output=normalized.data)

        return normalized

    async def _evaluate_policy(
        self,
        meta: ToolActionMeta,
        args: dict,
        ctx: Any,
    ) -> ToolPolicyResult:
        """Toolkit-level policy gate.

        The branch Principal is checked first through the core capability
        policy substrate. ``policy_evaluator`` remains only for contextual
        policy that depends on runtime arguments.
        """
        principal_result = await self._principal_allows(meta.name, meta, ctx)
        if not principal_result.allowed:
            return principal_result

        evaluator = self.config.policy_evaluator
        if not self.config.use_policies or evaluator is None:
            return ToolPolicyResult(allowed=True, action=meta.name)

        try:
            result = await evaluator.evaluate(
                action=f"{self.name}.{meta.name}",
                args=args,
                ctx=ctx,
            )
            if isinstance(result, ToolPolicyResult):
                return result
            return ToolPolicyResult(
                allowed=bool(getattr(result, "allowed", True)),
                enforcement=ToolEnforcement(getattr(result, "enforcement", "hard")),
                reason=str(getattr(result, "reason", "")),
                action=meta.name,
            )
        except Exception as e:
            logger.exception("Policy evaluator failed for %s.%s", self.name, meta.name)
            if self.config.fail_open:
                return ToolPolicyResult(
                    allowed=True,
                    action=meta.name,
                    reason=f"Evaluator error (fail-open): {e}",
                )
            return ToolPolicyResult(
                allowed=False,
                enforcement=ToolEnforcement.HARD,
                reason=f"Evaluator error (fail-closed): {e}",
                action=meta.name,
            )

    async def _run_hooks(
        self,
        hook_names: tuple[str, ...],
        args: dict,
        ctx: Any,
        output: Any = None,
    ) -> dict:
        for hook_name in hook_names:
            hook_fn = self.config.hooks.get(hook_name)
            if hook_fn is None:
                continue
            try:
                if output is not None:
                    await hook_fn(self, args, ctx, output)
                else:
                    result = await hook_fn(self, args, ctx)
                    if isinstance(result, dict):
                        args = result
            except Exception:
                if not self.config.fail_open:
                    raise
                logger.exception("Hook %r failed (fail-open)", hook_name)
        return args

    @property
    def allowed_actions(self) -> set[str]:
        """Return set of registered action names."""
        return set(self._action_registry.keys())

    def tool_schemas(
        self,
        *,
        fmt: str = "yaml",
        branch: Any = None,
    ) -> list[str] | list[dict[str, Any]]:
        """Generate tool schemas for LLM consumption.

        Renders input + output schemas as TypeScript-style notation
        following the khive-mcp @operation pattern. When output_schema
        is present, the LLM sees what fields the tool returns — enabling
        proper LNDL <lact> → OUT{} field mapping.

        When ``branch`` is provided, actions are filtered against
        ``branch.resources`` using :func:`scope_in_resources` semantics.
        Schemas for denied actions are omitted entirely so the LLM never
        sees them — defense in depth, plus token savings, plus cleaner
        prompts. The same matcher is used at dispatch (``act.py``), so
        what the LLM sees == what the system permits.

        Args:
            fmt: "yaml" (default, compact minimal_yaml) or "dict" (raw dict).
            branch: Optional Branch — filters actions by ``branch.resources``.

        Returns:
            List of schema strings (yaml) or dicts (dict format).
        """
        from lionagi.libs.schema import typescript_schema

        # Lazy import to avoid circular dep
        if branch is not None:
            from lionagi.beta.session.constraints import scope_in_resources

        schemas: list[dict[str, Any]] = []
        for action_name in self.actions:
            if branch is not None:
                scope = f"{self.name}:{action_name}"
                if not scope_in_resources(scope, branch.resources) and not (
                    len(self.actions) == 1
                    and scope_in_resources(self.name, branch.resources)
                ):
                    continue

            handler, meta = self._action_registry[action_name]
            full_name = f"{self.name}.{action_name}" if len(self.actions) > 1 else self.name

            desc = handler.__doc__ or full_name

            if meta.input_schema is not None:
                ts_in = typescript_schema(meta.input_schema.model_json_schema())
                if ts_in:
                    desc += f"\n\nInput:\n{ts_in}"

            if meta.output_schema is not None:
                ts_out = typescript_schema(meta.output_schema.model_json_schema())
                if ts_out:
                    desc += f"\n\nReturns:\n{ts_out}"

            schema: dict[str, Any] = {"name": full_name, "description": desc}
            schemas.append(schema)

        if fmt == "dict":
            return schemas

        from lionagi.libs.schema import minimal_yaml

        return [minimal_yaml(s) for s in schemas]

    @classmethod
    def from_handlers(
        cls,
        handlers: dict[str, Callable],
        *,
        name: str = "toolkit",
        provider: str = "local",
        path_policy: PathGuard | None = None,
        process_policy: ProcessGuard | None = None,
        input_schema: type | None = None,
        output_schema: type | None = None,
    ) -> ToolKit:
        """Create ToolKit from a dict of {action_name: async handler_fn}.

        Each handler should have signature: async def handler(args: dict, ctx) -> Any

        Args:
            handlers: {action_name: handler_fn} mapping.
            name: ToolKit name (used as resource name in session).
            provider: Provider identifier.
            path_policy: Workspace containment policy.
            process_policy: Subprocess security policy.
            input_schema: Pydantic BaseModel for input validation.
            output_schema: Pydantic BaseModel for output validation + schema rendering.
                When provided, the tool schema shows return field names so the LLM
                can map tool outputs to LNDL variables.
        """
        config = ToolKitConfig(
            name=name,
            provider=provider,
            path_policy=path_policy,
            process_policy=process_policy,
        )
        tk = cls(config=config)

        for action_name, handler_fn in handlers.items():
            meta = get_tool_action_meta(handler_fn)
            if meta is None:
                decorated = tool_action(
                    name=action_name,
                    input_schema=input_schema,
                    output_schema=output_schema,
                )(handler_fn)
                meta = get_tool_action_meta(decorated)
                handler_fn = decorated
            tk._action_registry[action_name] = (handler_fn, meta)

        return tk


async def register_toolkit(toolkit: ToolKit, update: bool = False) -> ToolKit:
    """Register a ToolKit in the global service registry."""
    await add_service(toolkit, update=update)
    return toolkit


def list_toolkits() -> list[ToolKit]:
    """Return globally registered ToolKit services."""
    return [
        service
        for service in list_services_sync()
        if isinstance(service, ToolKit)
    ]
