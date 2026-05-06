# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operate: top-level agent operation chain.

Handler signature: operate(params, ctx) -> validated model instance

Pipeline:
  1. Compose request structure from operable (inject action spec when needed)
  2. Structure: generate -> parse -> validate (produces typed model)
  3. Act: extract action_requests, execute, persist messages
  4. Compose response structure with action_results, validate

Action spec injection is narrow and explicit: only the `action_requests` spec
is auto-injected, only when `invoke_actions=True` AND `tool_schemas` is set
AND the branch carries the "action" capability AND the structure format is
JSON. LNDL routes through `lionagi.lndl.orchestrator` instead because
it carries `<lact>` tool calls in the response itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from lionagi.lndl.orchestrator import run_continuation, run_with_tools
from lionagi.service.toolkit import ToolKit
from lionagi.rules import Validator
from lionagi.ln.types import ModelConfig, Params
from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.protocols.messages import Message
from lionagi.protocols.messages.action_request import (
    ActionRequestContent as ActionRequest,
)
from lionagi.protocols.messages.action_response import (
    ActionResponseContent as ActionResponse,
)

from .act import ActParams
from .generate import GenerateParams
from .specs import Action, ActionResult, get_action_result_spec, get_action_spec
from .structure import StructureParams

if TYPE_CHECKING:
    from lionagi.service.imodel_v2 import iModel
    from lionagi.session.context import RequestContext
    from lionagi.ln.types import Operable
    from lionagi.protocols.messages.rendering import CustomParser

__all__ = ("OperateParams", "operate")


@dataclass(frozen=True, slots=True)
class OperateParams(Params):
    """Parameters for the operate pipeline; StructureParams is built internally after runtime spec composition."""

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    generate_params: GenerateParams
    operable: Operable
    validator: Validator = field(default_factory=Validator)

    capabilities: set[str] | None = None
    persist: bool = True

    invoke_actions: bool = False
    action_strategy: Literal["sequential", "concurrent"] = "concurrent"
    max_concurrent: int | None = None
    throttle_period: float | None = None

    auto_fix: bool = True
    strict: bool = True

    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    similarity_threshold: float = 0.85
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset

    max_lndl_rounds: int = 3

    toolkits: list[ToolKit] | None = None


async def operate(params: OperateParams, ctx: RequestContext) -> Any:
    """Operate handler: compose -> structure -> act -> merge.

    Returns a validated model instance. If actions were invoked, the response
    structure includes action_results. LNDL responses route through the LNDL
    orchestrator; JSON responses go through the structure pipeline plus the
    classic act/merge step.
    """
    session = await ctx.get_session()
    branch = await ctx.get_branch()

    operable = params.operable
    gen_params = params.generate_params

    is_lndl = gen_params.structure_format == "lndl"
    has_tools = not gen_params.is_sentinel_field("tool_schemas")
    branch_caps = getattr(branch, "capabilities", set())
    # LNDL uses <lact> tags inline; never inject the JSON action_requests spec.
    inject_actions = (
        params.invoke_actions and has_tools and "action" in branch_caps and not is_lndl
    )

    request_operable = (
        operable.extend([get_action_spec()]) if inject_actions else operable
    )
    request_structure = request_operable.compose_structure()
    use_gen_params = gen_params.with_updates(
        copy_containers="deep", request_model=request_structure
    )

    if is_lndl and params.invoke_actions and has_tools:
        return await run_with_tools(
            operable=request_operable,
            structure=request_structure,
            gen_params=use_gen_params,
            validator=params.validator,
            session=session,
            branch=branch,
            ctx=ctx,
            max_rounds=params.max_lndl_rounds,
            persist=params.persist,
            toolkits=params.toolkits,
        )

    structure_params = StructureParams(
        generate_params=use_gen_params,
        validator=params.validator,
        operable=request_operable,
        structure=request_structure,
        persist=params.persist,
        capabilities=params.capabilities,
        auto_fix=params.auto_fix,
        strict=params.strict,
        parse_imodel=params.parse_imodel,
        parse_imodel_kwargs=params.parse_imodel_kwargs,
        custom_parser=params.custom_parser,
        similarity_threshold=params.similarity_threshold,
        max_retries=params.max_retries,
        fill_mapping=params.fill_mapping,
        fill_value=params.fill_value,
    )

    structured = await run_continuation(
        structure_params=structure_params,
        ctx=ctx,
        session=session,
        branch=branch,
        use_gen_params=use_gen_params,
        max_rounds=params.max_lndl_rounds,
    )

    if not inject_actions:
        return structured

    return await _execute_and_merge_actions(
        structured=structured,
        request_operable=request_operable,
        params=params,
        session=session,
        branch=branch,
        ctx=ctx,
    )


async def _execute_and_merge_actions(
    *,
    structured: Any,
    request_operable: Operable,
    params: OperateParams,
    session: Any,
    branch: Any,
    ctx: RequestContext,
) -> Any:
    """JSON action path: pull action_requests off the validated model, execute
    them via `act`, persist responses, then re-emit the model with action_results
    appended. No re-validation — the structure was validated upstream.
    """
    act_requests = getattr(structured, "action_requests", None)
    if not act_requests:
        return structured

    action_messages = _to_action_messages(act_requests)
    if not action_messages:
        return structured

    for msg in action_messages:
        session.add_message(msg, branches=branch)

    responses = await ctx.conduct(
        "act",
        ActParams(
            action_requests=action_messages,
            strategy=params.action_strategy,
            max_concurrent=params.max_concurrent,
            throttle_period=params.throttle_period,
            toolkits=params.toolkits,
        ),
    )
    for resp in responses:
        session.add_message(Message(content=resp), branches=branch)

    action_results = _to_action_results(responses, action_messages)
    response_operable = request_operable.extend([get_action_result_spec()])
    response_structure = response_operable.compose_structure()

    data = request_operable.dump_instance(structured)
    data["action_results"] = action_results
    return response_structure(**data)


def _to_action_messages(act_requests: list[Action | dict]) -> list[Message]:
    messages: list[Message] = []
    for req in act_requests:
        if isinstance(req, Action):
            content = ActionRequest.create(
                function=req.function, arguments=req.arguments
            )
        elif isinstance(req, dict):
            content = ActionRequest.create(
                function=req.get("function", ""),
                arguments=req.get("arguments", {}),
            )
        else:
            continue
        messages.append(Message(content=content))
    return messages


def _to_action_results(
    responses: list[ActionResponse | dict],
    action_messages: list[Message],
) -> list[ActionResult]:
    id_to_func: dict[str, str] = {}
    for msg in action_messages:
        content = msg.content
        if hasattr(content, "function"):
            id_to_func[str(msg.id)] = content.function

    results: list[ActionResult] = []
    for resp in responses:
        if isinstance(resp, ActionResponse):
            request_id = (
                resp.request_id
                if not resp._is_sentinel(resp.request_id)
                else ""
            )
            results.append(
                ActionResult(
                    function=id_to_func.get(request_id, ""),
                    result=resp.result if resp.success else None,
                    error=resp.error if not resp.success else None,
                )
            )
        elif isinstance(resp, dict):
            results.append(ActionResult.model_validate(resp))
    return results
