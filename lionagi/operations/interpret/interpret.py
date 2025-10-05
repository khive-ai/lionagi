# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from lionagi.service.imodel import iModel

from ..types import ChatContext, InterpretContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


async def interpret(
    branch: "Branch",
    text: str,
    # Modern API: pass InterpretContext directly
    intp_ctx: InterpretContext = None,
    # Legacy API: individual parameters (backward compatible)
    domain: str = None,
    style: str = None,
    sample_writing: str = None,
    interpret_model: iModel = None,
    **kwargs,
) -> str:
    """
    Interpret and refine user input into clearer, more structured prompts.

    Two usage patterns:

    1. Modern (recommended):
        ctx = InterpretContext(domain="technical", style="formal", ...)
        result = await interpret(branch, text, intp_ctx=ctx)

    2. Legacy (backward compatible):
        result = await interpret(branch, text, domain="technical", style="formal", ...)

    Args:
        branch: Branch instance for execution
        text: User's raw instruction or question to interpret
        intp_ctx: InterpretContext object (modern API)
        domain: Domain hint for interpretation (legacy)
        style: Desired style for output (legacy)
        sample_writing: Sample writing style (legacy)
        interpret_model: Model to use for interpretation (legacy)
        **kwargs: Additional model parameters (legacy)

    Returns:
        Re-written, clarified prompt as string
    """
    # Build InterpretContext from whichever input was provided
    if intp_ctx is None:
        intp_ctx = InterpretContext(
            domain=domain or "general",
            style=style or "concise",
            sample_writing=sample_writing or "",
            imodel=interpret_model or branch.chat_model,
            imodel_kw=kwargs,
        )

    # Build instruction and guidance
    instruction = (
        "You are given a user's raw instruction or question. Your task is to rewrite it into a clearer, "
        "more structured prompt for an LLM or system, making any implicit or missing details explicit. "
        "Return only the re-written prompt. Do not assume any details not mentioned in the input, nor "
        "give additional instruction than what is explicitly stated."
    )

    guidance = (
        f"Domain hint: {intp_ctx.domain}. Desired style: {intp_ctx.style}."
    )
    if intp_ctx.sample_writing:
        guidance += f" Sample writing: {intp_ctx.sample_writing}"

    # Build ChatContext for execution
    chat_ctx = ChatContext(
        guidance=guidance,
        context=[f"User input: {text}"],
        sender=branch.user or "user",
        recipient=branch.id,
        response_format=None,
        progression=None,
        tool_schemas=[],
        images=[],
        image_detail="auto",
        plain_content="",
        include_token_usage_to_model=False,
        imodel=intp_ctx.imodel,
        imodel_kw={
            **intp_ctx.imodel_kw,
            "temperature": intp_ctx.imodel_kw.get("temperature", 0.1),
        },
    )

    from ..chat.chat import chat

    result = await chat(
        branch,
        instruction=instruction,
        chat_ctx=chat_ctx,
        return_ins_res_message=False,
    )

    return str(result)
