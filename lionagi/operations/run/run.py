# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import anyio
from pydantic import JsonValue

from lionagi.ln import acreate_path, json_dumps
from lionagi.protocols.messages import (
    ActionRequest,
    ActionResponse,
    AssistantResponse,
    AssistantResponseContent,
    Instruction,
)
from lionagi.service.connections import APICalling

from ..chat._prepare import _prepare_run_kwargs
from ..types import RunParam

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

    model = branch.chat_model
    endpoint = model.endpoint
    prev_stream_func = model.streaming_process_func
    bfp = None

    if param.stream_persist:
        # Placeholder: branch log with initial state
        fp = await acreate_path(
            param.persist_dir,
            str(branch.id),
            ".json",
            file_exist_ok=True,
        )
        async with await anyio.open_file(fp, "w") as f:
            await f.write(json_dumps(branch.to_dict()))

        # JSONL buffer for real-time monitoring
        bfp = await acreate_path(
            param.persist_dir,
            str(branch.id) + ".buffer",
            ".jsonl",
            file_exist_ok=True,
        )

        # Inject streaming persist into imodel's chunk processor
        async def _persist_chunk(chunk):
            if hasattr(chunk, "to_dict"):
                async with await anyio.open_file(bfp, "a") as f:
                    await f.write(json_dumps(chunk.to_dict()) + "\n")
            if prev_stream_func is not None:
                from lionagi.ln import is_coro_func

                if is_coro_func(prev_stream_func):
                    return await prev_stream_func(chunk)
                return prev_stream_func(chunk)
            return None

        model.streaming_process_func = _persist_chunk

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
    api_call_event = None

    try:
        async for chunk in model.stream(**kw):
            if isinstance(chunk, APICalling):
                api_call_event = chunk
                continue

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
                    yield act_res

                case "result":
                    pass

                case "error":
                    raise RuntimeError(
                        chunk.content or "Stream error from CLI endpoint"
                    )

        # Flush remaining text — attach APICalling metadata to final response
        if res := _flush_response():
            if api_call_event is not None:
                call_meta = api_call_event.to_dict()
                call_meta["execution"].pop("response", None)
                res.metadata["api_call_meta"] = call_meta
            yield res
    finally:
        # Restore original streaming func
        model.streaming_process_func = prev_stream_func

        # Consolidate: always persist branch state on any exit
        if param.stream_persist:
            fp = await acreate_path(
                param.persist_dir,
                str(branch.id),
                ".json",
                file_exist_ok=True,
            )
            async with await anyio.open_file(fp, "w") as f:
                await f.write(json_dumps(branch.to_dict()))
            if bfp is not None:
                bfp_path = anyio.Path(bfp)
                if await bfp_path.exists():
                    await bfp_path.unlink()
