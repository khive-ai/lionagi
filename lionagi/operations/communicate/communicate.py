# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Any, Literal

from pydantic import JsonValue

from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.protocols.types import AssistantResponse, Instruction

from ..types import ChatContext, ParseContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def communicate(
    branch: "Branch",
    instruction: JsonValue | Instruction = None,
    # Modern API: pass contexts directly
    chat_ctx: ChatContext = None,
    parse_ctx: ParseContext = None,
    # Legacy API: individual parameters (backward compatible)
    guidance: str = None,
    context=None,
    plain_content: str = None,
    sender: str = None,
    recipient: str = None,
    progression: list = None,
    response_format=None,
    imodel=None,
    skip_validation: bool = False,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = "auto",
    num_parse_retries: int = 3,
    fuzzy_match_kwargs: dict = None,
    parse_instruction: str = None,
    parse_guidance: str = None,
    parse_context: list = None,
    parse_imodel_kwargs: dict = None,
    clear_messages: bool = False,
    include_token_usage_to_model: bool = False,
    # Deprecated aliases
    request_model=None,
    request_fields=None,
    chat_model=None,
    parse_model=None,
    operative_model=None,
    **kwargs,
) -> Any:
    """
    Execute communication with chat and optional structured parsing.

    Two usage patterns:

    1. Modern (recommended):
        chat_ctx = ChatContext(...)
        parse_ctx = ParseContext(...)  # optional
        result = await communicate(branch, instruction, chat_ctx=chat_ctx, parse_ctx=parse_ctx)

    2. Legacy (backward compatible):
        result = await communicate(branch, instruction, guidance="...", response_format=MyModel, ...)

    Args:
        branch: Branch instance for execution
        instruction: Instruction content
        chat_ctx: ChatContext object (modern API)
        parse_ctx: ParseContext object (modern API, optional)
        guidance: Additional guidance text (legacy)
        context: Context data (legacy)
        plain_content: Plain text content (legacy)
        sender: Message sender (legacy)
        recipient: Message recipient (legacy)
        progression: Message progression sequence (legacy)
        response_format: Expected response format for parsing (legacy)
        imodel: Model to use for chat (legacy)
        skip_validation: Skip parsing validation (legacy)
        images: Image attachments (legacy)
        image_detail: Image detail level (legacy)
        num_parse_retries: Parse retry attempts (legacy)
        fuzzy_match_kwargs: Fuzzy match parameters (legacy)
        parse_instruction: Parse instruction override (legacy)
        parse_guidance: Parse guidance override (legacy)
        parse_context: Parse context override (legacy)
        parse_imodel_kwargs: Parse model kwargs (legacy)
        clear_messages: Clear message history before execution
        include_token_usage_to_model: Include token usage (legacy)
        request_model: DEPRECATED - use response_format
        request_fields: DEPRECATED - use response_format
        chat_model: DEPRECATED - use imodel
        parse_model: DEPRECATED - use parse_ctx
        operative_model: DEPRECATED - use response_format
        **kwargs: Additional model parameters (legacy)

    Returns:
        Parsed result (if response_format provided) or raw response string
    """
    # Handle deprecated parameters
    if operative_model or request_model or chat_model:
        warnings.warn(
            "Parameters 'operative_model', 'request_model', and 'chat_model' are deprecated. "
            "Use 'response_format' and 'imodel' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Build contexts from whichever input was provided
    if chat_ctx is None:
        # Resolve response format from various aliases
        target_format = (
            response_format
            or operative_model
            or request_model
            or request_fields
        )
        chat_imodel = imodel or chat_model or branch.chat_model

        chat_ctx = ChatContext(
            guidance=guidance,
            context=context,
            sender=sender or branch.user or "user",
            recipient=recipient or branch.id,
            response_format=target_format,
            progression=progression,
            tool_schemas=[],
            images=images or [],
            image_detail=image_detail or "auto",
            plain_content=plain_content or "",
            include_token_usage_to_model=include_token_usage_to_model,
            imodel=chat_imodel,
            imodel_kw=kwargs,
        )

    # Build parse context if needed and not provided
    if parse_ctx is None and chat_ctx.response_format and not skip_validation:
        from ..parse.parse import ParseExecutor

        if num_parse_retries > 5:
            warnings.warn(
                f"num_parse_retries={num_parse_retries} is high. Lowering to 5.",
                UserWarning,
                stacklevel=2,
            )
            num_parse_retries = 5

        if parse_context is not None and not isinstance(parse_context, list):
            raise TypeError("parse_context must be a list when provided")

        fuzzy_kw = dict(fuzzy_match_kwargs) if fuzzy_match_kwargs else {}
        handle_validation = fuzzy_kw.pop("handle_validation", "raise")

        parse_imodel = parse_model or branch.parse_model

        parse_ctx = ParseContext(
            response_format=chat_ctx.response_format,
            fuzzy_match_params=(
                FuzzyMatchKeysParams(**fuzzy_kw)
                if fuzzy_kw
                else FuzzyMatchKeysParams()
            ),
            handle_validation=handle_validation,
            alcall_params=ParseExecutor.DEFAULT_ALCALL_PARAMS.with_updates(
                retry_attempts=num_parse_retries
            ),
            imodel=parse_imodel,
            imodel_kw=parse_imodel_kwargs or {},
            format_instruction=parse_instruction,
            format_guidance=parse_guidance,
            format_context=(
                list(parse_context) if parse_context is not None else None
            ),
        )

    # Execute communication
    if clear_messages:
        branch.msgs.clear_messages()

    from ..chat.chat import chat

    ins, res = await chat(
        branch, instruction, chat_ctx=chat_ctx, return_ins_res_message=True
    )

    branch.msgs.add_message(instruction=ins)
    branch.msgs.add_message(assistant_response=res)

    if skip_validation:
        return res.response

    # Handle structured parsing when requested
    if parse_ctx and chat_ctx.response_format and not skip_validation:
        from ..parse.parse import parse

        try:
            out, res2 = await parse(
                branch,
                res.response,
                parse_ctx=parse_ctx,
                return_res_message=True,
            )
            if res2 and isinstance(res2, AssistantResponse):
                res.metadata["original_model_response"] = res.model_response
                res.metadata["model_response"] = res2.model_response
            return out
        except ValueError as e:
            raise ValueError(
                f"Failed to parse model response into {chat_ctx.response_format}: {e}"
            ) from e

    return res.response
