# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Literal

from pydantic import BaseModel

from lionagi.fields.action import ActionResponseModel
from lionagi.ln.async_call import AlcallParams
from lionagi.protocols.types import ActionRequest, ActionResponse, Log


async def _act(
    branch,
    action_request: BaseModel | dict,
    suppress_errors: bool = False,
    verbose_action: bool = False,
) -> "ActionResponseModel":
    _request = {}

    if isinstance(action_request, BaseModel):
        if hasattr(action_request, "function") and hasattr(
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
        action_request = ActionRequest.create(
            sender=branch.id,
            recipient=func_call.func_tool.id,
            **_request,
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
    branch,
    action_request: list | ActionRequest | BaseModel | dict,
    *,
    strategy: Literal["concurrent", "sequential"] = "concurrent",
    verbose_action: bool = False,
    suppress_errors: bool = True,
    call_params: AlcallParams = None,
) -> list[ActionResponse]:
    """Execute action requests using the branch's action manager."""
    global _DEFAULT_ALCALL_PARAMS
    if call_params is None:
        if _DEFAULT_ALCALL_PARAMS is None:
            _DEFAULT_ALCALL_PARAMS = AlcallParams(output_dropna=True)
        call_params = _DEFAULT_ALCALL_PARAMS

    kw = {
        "suppress_errors": suppress_errors,
        "verbose_action": verbose_action,
    }

    match strategy:
        case "concurrent":

            return await _concurrent_act(
                branch, action_request, call_params, **kw
            )
        case "sequential":
            return await _sequential_act(branch, action_request, **kw)
        case _:
            raise ValueError(
                "Invalid strategy. Choose 'concurrent' or 'sequential'."
            )


async def _concurrent_act(
    branch,
    action_request: ActionRequest | BaseModel | dict,
    call_params: AlcallParams = None,
    **kwargs,
) -> list:
    return await call_params(action_request, branch._act, **kwargs)


async def _sequential_act(
    branch,
    action_request: ActionRequest | BaseModel | dict,
    suppress_errors: bool = True,
    verbose_action: bool = False,
) -> list:
    action_request = (
        action_request
        if isinstance(action_request, list)
        else [action_request]
    )
    results = []
    for req in action_request:
        results.append(await branch._act(req, verbose_action, suppress_errors))
    return results
