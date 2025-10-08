# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from lionagi.fields.action import ActionResponseModel
from lionagi.ln._async_call import AlcallParams
from lionagi.protocols.types import ActionRequest, ActionResponse, Log

from ..types import ActionContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch

_DEFAULT_ALCALL_PARAMS = None


async def _act(
    branch: "Branch",
    action_request: BaseModel | dict,
    suppress_errors: bool = False,
    verbose_action: bool = False,
) -> ActionResponseModel | None:
    _request = {}

    if isinstance(action_request, BaseModel):
        # Check if it's an ActionRequest with content
        if hasattr(action_request, "content"):
            if hasattr(action_request.content, "function") and hasattr(
                action_request.content, "arguments"
            ):
                _request["function"] = action_request.content.function
                _request["arguments"] = action_request.content.arguments
        # Fallback for direct attributes (backward compatibility)
        elif hasattr(action_request, "function") and hasattr(
            action_request, "arguments"
        ):
            _request["function"] = action_request.function
            _request["arguments"] = action_request.arguments
    elif isinstance(action_request, dict):
        if {"function", "arguments"} <= set(action_request.keys()):
            _request["function"] = action_request["function"]
            _request["arguments"] = action_request["arguments"]

    try:
        if verbose_action:
            args_ = str(_request["arguments"])
            args_ = args_[:50] + "..." if len(args_) > 50 else args_
            print(f"Invoking action {_request['function']} with {args_}.")

        func_call = await branch._action_manager.invoke(_request)
        if verbose_action:
            print(
                f"Action {_request['function']} invoked, status: {func_call.status}."
            )

    except Exception as e:
        content = {
            "error": str(e),
            "function": _request.get("function"),
            "arguments": _request.get("arguments"),
            "branch": str(branch.id),
        }
        branch._log_manager.log(Log(content=content))
        if verbose_action:
            print(f"Action {_request['function']} failed, error: {str(e)}.")
        if suppress_errors:
            logging.error(
                f"Error invoking action '{_request['function']}': {e}"
            )
            return None
        raise e

    branch._log_manager.log(Log.create(func_call))

    if not isinstance(action_request, ActionRequest):
        action_request = ActionRequest(
            content=_request,
            sender=branch.id,
            recipient=func_call.func_tool.id,
        )

    # Add the action request/response to the message manager, if not present
    if action_request not in branch.messages:
        branch.msgs.add_message(action_request=action_request)

    branch.msgs.add_message(
        action_request=action_request,
        action_output=func_call.response,
    )

    return ActionResponseModel(
        function=action_request.function,
        arguments=action_request.arguments,
        output=func_call.response,
    )


async def act(
    branch: "Branch",
    action_request: list | ActionRequest | BaseModel | dict,
    *,
    strategy: Literal["concurrent", "sequential"] = "concurrent",
    verbose_action: bool = False,
    suppress_errors: bool = True,
    call_params: AlcallParams = None,
) -> list[ActionResponse]:
    """Execute action requests using the branch's action manager."""
    action_ctx = ActionContext(
        action_call_params=call_params or _get_default_call_params(),
        tools=None,  # Not used in this context
        strategy=strategy,
        suppress_errors=suppress_errors,
        verbose_action=verbose_action,
    )

    return await act_v1(branch, action_request, action_ctx)


async def act_v1(
    branch: "Branch",
    action_request: list | ActionRequest | BaseModel | dict,
    action_ctx: ActionContext,
) -> list[ActionResponse]:
    """Execute action requests with ActionContext."""

    match action_ctx.strategy:
        case "concurrent":
            return await _concurrent_act(
                branch,
                action_request,
                action_ctx.action_call_params,
                suppress_errors=action_ctx.suppress_errors,
                verbose_action=action_ctx.verbose_action,
            )
        case "sequential":
            return await _sequential_act(
                branch,
                action_request,
                suppress_errors=action_ctx.suppress_errors,
                verbose_action=action_ctx.verbose_action,
            )
        case _:
            raise ValueError(
                "Invalid strategy. Choose 'concurrent' or 'sequential'."
            )


async def _concurrent_act(
    branch: "Branch",
    action_request: list | ActionRequest | BaseModel | dict,
    call_params: AlcallParams,
    suppress_errors: bool = True,
    verbose_action: bool = False,
) -> list:
    """Execute actions concurrently using AlcallParams."""

    async def _wrapper(req):
        return await _act(branch, req, suppress_errors, verbose_action)

    # AlcallParams expects a list as first argument
    action_request_list = (
        action_request
        if isinstance(action_request, list)
        else [action_request]
    )

    return await call_params(action_request_list, _wrapper)


async def _sequential_act(
    branch: "Branch",
    action_request: list | ActionRequest | BaseModel | dict,
    suppress_errors: bool = True,
    verbose_action: bool = False,
) -> list:
    """Execute actions sequentially."""
    action_request = (
        action_request
        if isinstance(action_request, list)
        else [action_request]
    )
    results = []
    for req in action_request:
        result = await _act(branch, req, suppress_errors, verbose_action)
        results.append(result)
    return results


def _get_default_call_params() -> AlcallParams:
    """Get or create default AlcallParams."""
    global _DEFAULT_ALCALL_PARAMS
    if _DEFAULT_ALCALL_PARAMS is None:
        _DEFAULT_ALCALL_PARAMS = AlcallParams(output_dropna=True)
    return _DEFAULT_ALCALL_PARAMS
