# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from lionagi.protocols.messages.action_request import ActionRequest
from lionagi.protocols.messages.action_response import ActionResponse
from lionagi.protocols.messages.assistant_response import (
    AssistantResponse,
    AssistantResponseContent,
)

if TYPE_CHECKING:
    from lionagi.protocols.messages.message import RoledMessage
    from lionagi.session.branch import Branch


async def run(
    branch: Branch,
    instruction=None,
    chat_model=None,
    sender=None,
    recipient=None,
    guidance=None,
    context=None,
    images=None,
    image_detail=None,
    response_format=None,
    **kwargs,
) -> AsyncGenerator[RoledMessage, None]:
    """Stream Messages from a CLI endpoint.

    Yields Instruction, AssistantResponse, ActionRequest, and ActionResponse
    messages as they arrive from the endpoint stream.

    Accepts chat_model to override the branch default, enabling
    multi-model conversations on a single branch:

        sonnet = iModel(model="claude_code/sonnet")
        codex  = iModel(model="codex/gpt-5.3-codex-spark")
        async for msg in branch.run("step 1", chat_model=sonnet): ...
        async for msg in branch.run("step 2", chat_model=codex): ...
    """
    model = chat_model or branch.chat_model
    endpoint = model.endpoint
    ins = branch.msgs.add_message(
        instruction=instruction,
        sender=sender,
        recipient=recipient,
        guidance=guidance,
        context=context,
        images=images,
        image_detail=image_detail,
        response_format=response_format,
    )
    yield ins

    # Build the request dict for the endpoint
    session_id = getattr(endpoint, "session_id", None)
    chat_msgs = []
    for msg_id in branch.msgs.progression:
        msg = branch.msgs.messages[msg_id]
        if hasattr(msg, "chat_msg") and msg.chat_msg is not None:
            chat_msgs.append(msg.chat_msg)

    request_dict: dict = {
        "messages": chat_msgs,
        **({"resume": session_id} if session_id else {}),
        **kwargs,
    }

    # Accumulation buffers
    thinking_parts: list[str] = []
    text_parts: list[str] = []

    def _flush_response() -> AssistantResponse | None:
        """Build an AssistantResponse from accumulated text/thinking chunks."""
        if not text_parts:
            return None
        text = "".join(text_parts)
        metadata: dict = {}
        if thinking_parts:
            metadata["thinking"] = "\n".join(thinking_parts)

        res = AssistantResponse(
            content=AssistantResponseContent(assistant_response=text),
            sender=branch.id,
            recipient=branch.user or "user",
        )
        if metadata:
            res.metadata.update(metadata)
        branch.msgs.add_message(assistant_response=res)
        text_parts.clear()
        thinking_parts.clear()
        return res

    # tool_id → ActionRequest, for linking tool_result chunks
    pending_requests: dict[str, ActionRequest] = {}

    async for chunk in endpoint.stream(request_dict):
        match chunk.type:
            case "system":
                # Store session_id on the endpoint for future resume
                if sid := chunk.metadata.get("session_id"):
                    endpoint.session_id = sid

            case "thinking":
                if chunk.content:
                    thinking_parts.append(chunk.content)

            case "text":
                if chunk.content:
                    text_parts.append(chunk.content)

            case "tool_use":
                # Flush any accumulated text before the tool call
                if res := _flush_response():
                    yield res

                act_req = branch.msgs.create_action_request(
                    function=chunk.tool_name or "",
                    arguments=chunk.tool_input or {},
                    sender=branch.id,
                    recipient=branch.user or "user",
                )
                if chunk.tool_id:
                    pending_requests[chunk.tool_id] = act_req
                branch.msgs.add_message(action_request=act_req)
                yield act_req

            case "tool_result":
                # Link to the originating ActionRequest when we have the id
                orig_req = (
                    pending_requests.pop(chunk.tool_id, None) if chunk.tool_id else None
                )
                if orig_req is not None:
                    act_res = branch.msgs.create_action_response(
                        action_request=orig_req,
                        action_output=chunk.tool_output,
                        sender=branch.user or "user",
                        recipient=branch.id,
                    )
                    branch.msgs.messages.include(act_res)
                else:
                    # No matching request (e.g. resumed session mid-stream)
                    act_res = ActionResponse(
                        content={
                            "function": chunk.tool_name or "",
                            "arguments": {},
                            "output": chunk.tool_output,
                        },
                        sender=branch.user or "user",
                        recipient=branch.id,
                    )
                    if chunk.is_error:
                        act_res.metadata["is_error"] = True
                    if chunk.tool_id:
                        act_res.metadata["tool_id"] = chunk.tool_id
                    branch.msgs.messages.include(act_res)
                yield act_res

            case "result":
                pass  # Stats/timing — no message needed

            case "error":
                raise RuntimeError(chunk.content or "Stream error from CLI endpoint")

    # Flush any remaining accumulated text
    if res := _flush_response():
        yield res
