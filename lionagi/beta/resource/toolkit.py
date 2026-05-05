# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""ToolKit: Service subclass for agent-runtime tool execution with policy enforcement."""

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
        async def evaluate(
            self, action: str, args: dict, ctx: Any
        ) -> ToolPolicyResult: ...


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


_TOOL_ACTION_ATTR = "_tool_action"


class ToolEnforcement(StrEnum):

    HARD = "hard"
    SOFT = "soft"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class ToolActionMeta:
    """Metadata attached to a @tool_action handler; requires declares extra capabilities."""

    name: str
    input_schema: type | None = None
    output_schema: type | None = None
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    pre_hooks: tuple[str, ...] = ()
    post_hooks: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolPolicyResult:

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
    """Decorator that attaches ToolActionMeta to a handler method."""

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
    return getattr(handler, _TOOL_ACTION_ATTR, None)


logger = logging.getLogger(__name__)


class ToolKitConfig(ResourceConfig):
    """ResourceConfig extended with path/process policies and a contextual policy evaluator."""

    path_policy: PathGuard | None = Field(default=None, exclude=True)
    process_policy: ProcessGuard | None = Field(default=None, exclude=True)
    hooks: dict[str, Callable] = Field(default_factory=dict, exclude=True)
    fail_open: bool = False
    policy_evaluator: Any = Field(default=None, exclude=True)
    use_policies: bool = True


class ToolKit(Service):
    """Service with @tool_action handler discovery and capability-gated invocation."""

    config: ToolKitConfig = Field(..., description="ToolKit configuration")
    _action_registry: dict[str, tuple[Callable, ToolActionMeta]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_service_name(cls, data: Any) -> Any:
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

    def create_payload(
        self, request: dict | BaseModel | None = None, **kwargs: Any
    ) -> dict:
        """Unwrap iModel's {"arguments": {...}} envelope so call() receives the inner dict."""
        if request is None:
            payload = kwargs
        elif isinstance(request, BaseModel):
            payload = request.model_dump(exclude_none=True)
        else:
            payload = dict(request)
            payload.update(kwargs)

        if (
            "arguments" in payload
            and len(payload) == 1
            and isinstance(payload["arguments"], dict)
        ):
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

        payload = await self._run_hooks(meta.pre_hooks, payload, ctx)

        result = await self._evaluate_policy(meta, payload, ctx)
        if not result.allowed:
            if result.enforcement in (ToolEnforcement.HARD, ToolEnforcement.SOFT):
                return Normalized(
                    status="error",
                    data=None,
                    error=f"Policy blocked {action}: {result.reason}",
                )
            logger.warning("Policy advisory for %s: %s", action, result.reason)

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

        await self._run_hooks(meta.post_hooks, payload, ctx, output=normalized.data)

        return normalized

    async def _evaluate_policy(
        self,
        meta: ToolActionMeta,
        args: dict,
        ctx: Any,
    ) -> ToolPolicyResult:
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
        return set(self._action_registry.keys())

    def tool_schemas(
        self,
        *,
        fmt: str = "yaml",
        branch: Any = None,
    ) -> list[str] | list[dict[str, Any]]:
        """Render action schemas for LLM; branch filters by branch.resources (same matcher as dispatch)."""
        from lionagi.libs.schema import typescript_schema

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
            full_name = (
                f"{self.name}.{action_name}" if len(self.actions) > 1 else self.name
            )

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
    await add_service(toolkit, update=update)
    return toolkit


def list_toolkits() -> list[ToolKit]:
    return [service for service in list_services_sync() if isinstance(service, ToolKit)]
