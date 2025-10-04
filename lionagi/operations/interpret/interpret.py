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
    domain: str | None = None,
    style: str | None = None,
    sample_writing: str | None = None,
    interpret_model: iModel | None = None,
    **kwargs,
) -> str:
    """Interpret and refine user input into clearer prompts."""

    # Build InterpretContext
    intp_ctx = InterpretContext(
        domain=domain or "general",
        style=style or "concise",
        sample_writing=sample_writing or "",
        imodel=interpret_model or branch.chat_model,
        imodel_kw=kwargs,
    )

    return await interpret_v1(branch, text, intp_ctx)


async def interpret_v1(
    branch: "Branch",
    text: str,
    intp_ctx: InterpretContext,
) -> str:
    """Execute interpretation with context - clean implementation."""

    from ..chat.chat import chat_v1

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

    # Build ChatContext
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

    result = await chat_v1(
        branch,
        instruction=instruction,
        chat_ctx=chat_ctx,
        return_ins_res_message=False,
    )

    return str(result)
