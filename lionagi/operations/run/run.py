# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue

from lionagi import json_dumps
from lionagi.ln import acreate_path
from lionagi.operations.types import RunParam
from lionagi.protocols.messages.action_request import ActionRequest
from lionagi.protocols.messages.action_response import ActionResponse
from lionagi.protocols.messages.assistant_response import (
    AssistantResponse,
    AssistantResponseContent,
)
from lionagi.protocols.messages.instruction import Instruction

from ..chat._prepare import _prepare_run_kwargs

if TYPE_CHECKING:
    from lionagi.protocols.messages.message import RoledMessage
    from lionagi.session.branch import Branch


async def run(
    branch: Branch,
    instruction: JsonValue | Instruction,
    param: RunParam,
) -> AsyncGenerator[RoledMessage, None]:
    if param.imodel is not None:
        branch.chat_model = param.imodel

    if not branch.chat_model.is_cli:
        raise ValueError("run operation only supports CLI endpoints")

    ins, kw = _prepare_run_kwargs(branch, instruction, param)
    branch.msgs.add_message(instruction=ins)
    yield ins

    if branch.chat_model.provider_session_id is not None:
        kw["resume"] = branch.chat_model.provider_session_id

    # Stream persistence: JSONL buffer for crash recovery
    bfp = None
    if param.stream_persist:
        bfp = await acreate_path(
            param.persist_dir,
            str(branch.id) + ".buffer",
            ".jsonl",
            file_exist_ok=True,
        )

    def _buffer_msg(msg):
        if bfp is not None and hasattr(msg, "to_dict"):
            with open(bfp, "a") as f:
                f.write(json_dumps(msg.to_dict()) + "\n")

    # Accumulation buffers
    thinking_parts: list[str] = []
    text_parts: list[str] = []

    def _flush_response() -> AssistantResponse | None:
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

    pending_requests: dict[str, ActionRequest] = {}
    endpoint = branch.chat_model.endpoint

    try:
        async for chunk in endpoint.stream(kw):
            match chunk.type:
                case "system":
                    if sid := chunk.metadata.get("session_id"):
                        endpoint.session_id = sid

                case "thinking":
                    if chunk.content:
                        thinking_parts.append(chunk.content)

                case "text":
                    if chunk.content:
                        text_parts.append(chunk.content)

                case "tool_use":
                    if res := _flush_response():
                        _buffer_msg(res)
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
                    _buffer_msg(act_req)
                    yield act_req

                case "tool_result":
                    orig_req = (
                        pending_requests.pop(chunk.tool_id, None)
                        if chunk.tool_id
                        else None
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
                    _buffer_msg(act_res)
                    yield act_res

                case "result":
                    pass

                case "error":
                    raise RuntimeError(
                        chunk.content or "Stream error from CLI endpoint"
                    )

        # Flush remaining accumulated text
        if res := _flush_response():
            _buffer_msg(res)
            yield res
    finally:
        if param.stream_persist:
            persist_dir = Path(param.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            fp = persist_dir / f"{branch.id}.json"
            fp.write_text(json_dumps(branch.to_dict()))
            if bfp and Path(bfp).exists():
                Path(bfp).unlink()
