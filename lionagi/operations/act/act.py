# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from lionagi.protocols.action.manager import ActionConfig
from lionagi.protocols.types import (
    ActionRequest,
    ActionResponse,
    EventStatus,
    Log,
)
from lionagi.utils import alcall

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def _act(
    action_request: ActionRequest,
    suppress_errors: bool,
    verbose_action: bool,
    branch: "Branch",
) -> ActionResponse | None:

    func_call = await branch._action_manager.invoke(
        action_request, verbose=verbose_action
    )
    log_ = Log(content=func_call.to_dict())
    await branch._log_manager.alog(log_=log_)
    action_request.metadata["status"] = func_call.status.value

    if func_call.status == EventStatus.FAILED and not suppress_errors:
        raise ValueError(
            f"Action '{func_call.function}' failed with status: {func_call.status.value}. Log ID: {str(log_.id)}"
        )

    if action_request not in branch.messages:
        branch.msgs.add_message(action_request=action_request)

    a_res = branch.msgs.add_message(
        action_request=action_request,
        action_output=func_call.response,
    )
    a_res.metadata["tool_id"] = str(func_call.func_tool.id)
    a_res.metadata["log_id"] = str(log_.id)
    return a_res


async def act(
    branch,
    action_request: list | ActionRequest | BaseModel | dict,
    *,
    strategy: Literal["concurrent", "sequential"] = "concurrent",
    action_config: ActionConfig = None,
    verbose_action: bool = False,
    suppress_errors: bool = True,
) -> list[ActionResponse]:
    match strategy:
        case "concurrent":
            return await _concurrent_act(
                branch=branch,
                action_request=action_request,
                verbose_action=verbose_action,
                suppress_errors=suppress_errors,
                **((action_config or branch.acts.config).get_retry_kwargs()),
            )
        case "sequential":
            return await _sequential_act(
                branch=branch,
                action_request=action_request,
                verbose_action=verbose_action,
                suppress_errors=suppress_errors,
                **((action_config or branch.acts.config).get_retry_kwargs()),
            )
        case _:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Supported strategies are 'concurrent' and 'sequential'."
            )


async def _concurrent_act(
    branch, action_request: list[ActionRequest], **kwargs
) -> list:
    return await alcall(action_request, _act, branch=branch, **kwargs)


async def _sequential_act(
    branch, action_request: list[ActionRequest], **kwargs
) -> list:
    results = []
    for req in action_request:
        result = await alcall(req, _act, branch=branch, **kwargs)
        if result and len(result) > 0:
            results.append(result[0])

    return results
