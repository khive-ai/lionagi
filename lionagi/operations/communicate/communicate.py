# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.fuzzy._fuzzy_validate import fuzzy_validate_mapping
from lionagi.ln.types import Undefined
from lionagi.protocols.types import AssistantResponse, Instruction

from ..types import ChatContext, Context, ParseContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def communicate(
    branch: "Branch",
    instruction=None,
    *,
    guidance=None,
    context=None,
    plain_content=None,
    sender=None,
    recipient=None,
    progression=None,
    request_model=None,
    response_format=None,
    request_fields=None,
    imodel=None,
    chat_model=None,
    parse_model=None,
    skip_validation=False,
    images=None,
    image_detail: Literal["low", "high", "auto"] = "auto",
    num_parse_retries=3,
    fuzzy_match_kwargs=None,
    clear_messages=False,
    operative_model=None,
    include_token_usage_to_model: bool = False,
    **kwargs,
):
    # Handle deprecated parameters
    if operative_model:
        warnings.warn(
            "Parameter 'operative_model' is deprecated. Use 'response_format' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if (
        (operative_model and response_format)
        or (operative_model and request_model)
        or (response_format and request_model)
    ):
        raise ValueError(
            "Cannot specify both operative_model and response_format "
            "or operative_model and request_model as they are aliases "
            "for the same parameter."
        )

    response_format = (
        response_format or operative_model or request_model or request_fields
    )
    imodel = imodel or chat_model or branch.chat_model
    parse_model = parse_model or branch.parse_model

    if num_parse_retries > 5:
        warnings.warn(
            f"num_parse_retries={num_parse_retries} is high. Lowering to 5. Suggestion: <3",
            UserWarning,
            stacklevel=2,
        )
        num_parse_retries = 5

    # Build contexts
    chat_ctx = ChatContext(
        guidance=guidance,
        context=context,
        sender=sender or branch.user or "user",
        recipient=recipient or branch.id,
        response_format=response_format,
        progression=progression,
        tool_schemas=[],
        images=images or [],
        image_detail=image_detail,
        plain_content=plain_content or "",
        include_token_usage_to_model=include_token_usage_to_model,
        imodel=imodel,
        imodel_kw=kwargs,
    )

    parse_ctx = None
    if response_format and not skip_validation:
        from ..parse.parse import get_default_call

        fuzzy_kw = fuzzy_match_kwargs or {}
        handle_validation = fuzzy_kw.pop("handle_validation", "raise")

        parse_ctx = ParseContext(
            response_format=response_format,
            fuzzy_match_params=(
                FuzzyMatchKeysParams(**fuzzy_kw)
                if fuzzy_kw
                else FuzzyMatchKeysParams()
            ),
            handle_validation=handle_validation,
            alcall_params=get_default_call().with_updates(
                retry_attempts=num_parse_retries
            ),
            imodel=parse_model,
            imodel_kw={},
        )

    return await communicate_v1(
        branch,
        instruction=instruction,
        chat_ctx=chat_ctx,
        parse_ctx=parse_ctx,
        clear_messages=clear_messages,
        skip_validation=skip_validation,
        request_fields=request_fields,
    )


# accepts to chat, and parse contexts
async def communicate(
    branch,
    instruction,
    ctx: Context,
    *,
    response_format: dict | type[BaseModel] = None,
    skip_validation: bool = False,
    clear_messages: bool = False,
):
    if clear_messages:
        branch.msgs.clear_messages()

    from ..chat.chat import chat_v1

    ins, res = await chat_v1(
        branch, instruction, chat_ctx, return_ins_res_message=True
    )

    branch.msgs.add_message(instruction=ins)
    branch.msgs.add_message(assistant_response=res)

    if skip_validation:
        return res.response

    ...


async def communicate(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    parse_ctx: ParseContext = None,
    *,
    response_format: dict | type[BaseModel] = None,
    skip_validation: bool = False,
    clear_messages: bool = False,
):

    _cctx, _pctx = None, None
    if response_format is not None:
        _cctx = cha

    ...


async def communicate_v1(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    parse_ctx: ParseContext | None = None,
    clear_messages: bool = False,
    skip_validation: bool = False,
) -> Any:
    if clear_messages:
        branch.msgs.clear_messages()

    from ..chat.chat import chat_v1

    ins, res = await chat_v1(
        branch, instruction, chat_ctx, return_ins_res_message=True
    )

    branch.msgs.add_message(instruction=ins)
    branch.msgs.add_message(assistant_response=res)

    if skip_validation:
        return res.response

    # Handle response_format with parse
    if parse_ctx or chat_ctx.response_format:

        _pctx = parse_ctx or ParseContext()
        _pctx = _pctx.with_updates(
            response_format=_pctx.response_format or chat_ctx.response_format
        )

        from ..parse.parse import parse_v1

        try:
            out, res2 = await parse_v1(
                branch, res.response, _pctx, return_res_message=True
            )
            if res2 and isinstance(res2, AssistantResponse):
                res.metadata["original_model_response"] = res.model_response
                res.metadata["model_response"] = res2.model_response
            return out
        except ValueError as e:
            # Re-raise with more context
            raise ValueError(
                f"Failed to parse model response into {chat_ctx.response_format}: {e}"
            ) from e

    return res.response
