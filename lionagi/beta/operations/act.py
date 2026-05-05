# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Act operation: execute tool calls from LLM action requests.

Handler signature: act(params, ctx) → list[ActionResponse]

Supports sequential and concurrent execution strategies with
rate-limiting via alcall (delay, throttle, max_concurrent).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from lionagi.protocols.messages.action_request import ActionRequestContent as ActionRequest
from lionagi.protocols.messages.action_response import ActionResponseContent as ActionResponse
from lionagi.beta.session.constraints import scope_must_be_accessible
from lionagi.ln import alcall
from lionagi.ln.types import Params

if TYPE_CHECKING:

    from lionagi.beta.session.context import RequestContext
    from lionagi.beta.session.session import Branch, Session
    from lionagi.protocols.messages import Message
    from lionagi.beta.resource.toolkit import ToolKit


@dataclass(frozen=True, slots=True, init=False)
class ActParams(Params):
    """Tool execution parameters; strategy controls sequential vs concurrent dispatch."""

    action_requests: list[Message]
    delay_before_start: float = 0
    throttle_period: float | None = None
    max_concurrent: int | None = None
    strategy: Literal["sequential", "concurrent"] = "concurrent"
    toolkits: list[ToolKit] | None = None


async def act(params: ActParams, ctx: RequestContext) -> list[ActionResponse]:
    session = await ctx.get_session()
    branch = session.get_branch(ctx.branch)
    return await _act(session=session, branch=branch, ctx=ctx, **params.to_dict())


def _resolve_scope(
    name: str,
    toolkits: list[ToolKit],
) -> tuple[str, str | None, str | None]:
    """Resolve an LLM-emitted function name to canonical ``service:operation`` form.

    Handles the four shapes the LLM may emit:
      - canonical:    ``"foo:wolf"``  →  ("foo:wolf", "wolf", None)
      - dotted:       ``"foo.wolf"``  →  ("foo:wolf", "wolf", None)
      - bare action:  ``"wolf"``      →  ("foo:wolf", "wolf", None) if unambiguous
      - bare toolkit: ``"foo"``       →  ("foo", None, None) — toolkit dispatches its action

    Ambiguous bare actions (action name appears in multiple toolkits) hard-error
    with a hint to qualify, surfaced as the third tuple slot. Unknown names also
    return a hint so the dispatcher produces a clean error.

    Returns:
        (canonical_scope, action_name_or_None, error_hint_or_None)
    """
    # Canonical "service:op"
    if ":" in name:
        service, action_name = name.split(":", 1)
        return name, action_name, None

    # Dotted "service.op"
    if "." in name:
        service, action_name = name.split(".", 1)
        return f"{service}:{action_name}", action_name, None

    # Bare name — could be an action or a toolkit
    matching_toolkits = [tk for tk in toolkits if name in tk.allowed_actions]
    if len(matching_toolkits) == 1:
        tk = matching_toolkits[0]
        return f"{tk.name}:{name}", name, None
    if len(matching_toolkits) > 1:
        candidates = sorted(f"{tk.name}:{name}" for tk in matching_toolkits)
        hint = f"ambiguous action '{name}'; qualify as one of {candidates}"
        return name, None, hint

    # Maybe it's a bare toolkit name (single-action or multi-action)
    for tk in toolkits:
        if tk.name == name:
            # Single-action toolkit: ToolKit.call() will auto-select.
            # Multi-action toolkit: ToolKit.call() will return a helpful error.
            return name, None, None

    # Unknown — let downstream raise NotFoundError, but include a hint
    return name, None, f"unknown action '{name}'"


async def _act(
    action_requests: list[Message],
    session: Session,
    branch: Branch,
    ctx: RequestContext | None = None,
    delay_before_start: float = 0,
    throttle_period: float | None = None,
    max_concurrent: int | None = None,
    strategy: Literal["sequential", "concurrent"] = "concurrent",
    toolkits: list[ToolKit] | None = None,
) -> list[ActionResponse]:
    if not action_requests:
        return []

    if toolkits is None:
        from lionagi.tools import list_toolkits

        toolkits = list_toolkits()
    toolkit_by_name = {tk.name: tk for tk in toolkits}

    # Resolve LLM-emitted names to canonical "service:operation" scopes.
    # Persist canonical form to req.content so _execute_one and downstream
    # gating both see the resolved scope.
    for req in action_requests:
        content: ActionRequest = req.content
        canonical, action_name, error_hint = _resolve_scope(content.function, toolkits)
        if action_name is not None:
            args = dict(content.arguments) if content.arguments else {}
            args["action"] = action_name
            req.content = content.with_updates(function=canonical, arguments=args)
        elif canonical != content.function:
            req.content = content.with_updates(function=canonical)
        # Stash hint for downstream error reporting via metadata.
        if error_hint:
            req._scope_resolution_hint = error_hint  # type: ignore[attr-defined]

    async def _execute_one(req_msg: Message) -> ActionResponse:
        action_request: ActionRequest = req_msg.content
        scope = action_request.function
        hint = getattr(req_msg, "_scope_resolution_hint", None)
        if hint:
            return ActionResponse.create(request_id=str(req_msg.id), error=str(hint))

        try:
            scope_must_be_accessible(branch, scope)
            toolkit_name = scope.split(":", 1)[0] if ":" in scope else scope
            from lionagi.beta.resource.service import get_service

            try:
                service = await get_service(toolkit_name)
            except Exception:
                service = toolkit_by_name[toolkit_name]
        except Exception as e:
            return ActionResponse.create(request_id=str(req_msg.id), error=str(e))
        try:
            from lionagi.beta.session.context import RequestContext

            service_ctx = RequestContext(
                name=scope,
                session_id=session.id,
                branch=branch.name or branch.id,
                service=toolkit_name,
                conn=getattr(ctx, "conn", None) if ctx is not None else None,
                query_fn=getattr(ctx, "query_fn", None) if ctx is not None else None,
                now=getattr(ctx, "now", None) if ctx is not None else None,
                **{
                    **(ctx.metadata if ctx is not None else {}),
                    "_bound_session": session,
                    "_bound_branch": branch,
                    "_bound_service": service,
                    **({"parent_context_id": ctx.id} if ctx is not None else {}),
                },
            )
            action_name = scope.split(":", 1)[1] if ":" in scope else None
            normalized = await service.call(
                action_name,
                dict(action_request.arguments or {}),
                service_ctx,
            )
        except Exception as e:
            return ActionResponse.create(request_id=str(req_msg.id), error=f"ExecutionError: {e}")
        if normalized.status == "error":
            return ActionResponse.create(request_id=str(req_msg.id), error=normalized.error or "Tool error")
        return ActionResponse.create(request_id=str(req_msg.id), result=normalized.data)

    if strategy == "sequential":
        results: list[ActionResponse] = []
        for req_msg in action_requests:
            results.append(await _execute_one(req_msg))
        return results

    return await partial(
        alcall,
        delay_before_start=delay_before_start,
        throttle_period=throttle_period,
        max_concurrent=max_concurrent,
    )(action_requests, _execute_one)
