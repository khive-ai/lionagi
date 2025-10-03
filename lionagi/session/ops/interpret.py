# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from .types import ChatContext, InterpretContext


async def interpret(branch, text: str, intp_ctx: InterpretContext) -> str:

    from .chat import chat

    return await chat(
        branch,
        (
            "You are given a user's raw instruction or question. Your task is to rewrite it into a clearer,"
            "more structured prompt for an LLM or system, making any implicit or missing details explicit. "
            "Return only the re-written prompt. Do not assume any details not mentioned in the input, nor "
            "give additional instruction than what is explicitly stated."
        ),
        chat_ctx=ChatContext(
            guidance=f"Domain hint: {intp_ctx.domain or 'general'}. Writing style: {intp_ctx.style or 'concise'}. ",
            context={
                "user_input": text,
                "sample_writing": intp_ctx.sample_writing or "<Not Provided>",
            },
            imodel=intp_ctx.imodel,
            imodel_kw=intp_ctx.imodel_kw,
        ),
    )
