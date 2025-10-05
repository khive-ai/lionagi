# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue

from lionagi.protocols.types import (
    ActionResponse,
    AssistantResponse,
    Instruction,
    Log,
)
from lionagi.service.imodel import iModel
from lionagi.utils import copy

from ..types import ChatContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


class MessageHistoryBuilder:
    """Single-pass builder for chat message history with optimizations."""

    def __init__(
        self,
        branch: "Branch",
        progression: list,
        new_instruction: Instruction,
    ):
        self.branch = branch
        self.progression = progression
        self.new_instruction = new_instruction
        self.history: list = []
        self.pending_actions: list[ActionResponse] = []
        self.system_applied = False

    def build(self) -> tuple[list, Instruction]:
        """Build message history in a single pass."""
        # Check if system exists - need to validate first message type
        has_system = (
            hasattr(self.branch.msgs, "system") and self.branch.msgs.system
        )
        first_non_action_msg = None

        # Process each message in progression
        for msg_id in self.progression:
            try:
                msg = self.branch.msgs.messages[msg_id]
            except KeyError:
                continue

            if isinstance(msg, ActionResponse):
                self.pending_actions.append(msg)
            elif isinstance(msg, Instruction):
                self._process_instruction(msg)
            elif isinstance(msg, AssistantResponse):
                # Validate: if system exists, first non-action message must be Instruction
                if has_system and first_non_action_msg is None:
                    raise ValueError(
                        "First message in progression must be an Instruction or System"
                    )
                self._process_response(msg)
            else:
                self.history.append(msg)

            # Track first non-action message for validation
            if first_non_action_msg is None and not isinstance(
                msg, ActionResponse
            ):
                first_non_action_msg = msg

        # Finalize with new instruction
        final_instruction = self._finalize_instruction()
        return self.history, final_instruction

    def _process_instruction(self, msg: Instruction) -> None:
        """Process instruction: copy, strip schemas, merge actions, inject system."""
        # Always copy instructions (they're mutated)
        msg_copy = msg.model_copy()
        msg_copy.tool_schemas = None
        msg_copy.respond_schema_info = None
        msg_copy.request_response_format = None

        # Merge pending actions into context
        if self.pending_actions:
            msg_copy = self._merge_action_context(msg_copy)
            self.pending_actions = []

        # Inject system prompt to first instruction
        if not self.system_applied and hasattr(self.branch.msgs, "system"):
            system = self.branch.msgs.system
            if system:
                system_text = getattr(system, "rendered", str(system))
                msg_copy.guidance = f"{system_text}{msg_copy.guidance or ''}"
                self.system_applied = True

        self.history.append(msg_copy)

    def _process_response(self, msg: AssistantResponse) -> None:
        """Process response: merge consecutive responses (token optimization)."""
        # Merge consecutive assistant responses
        if self.history and isinstance(self.history[-1], AssistantResponse):
            self.history[-1].response = (
                f"{self.history[-1].response}\n\n{msg.response}"
            )
        else:
            # No merge needed - use reference (no copy)
            self.history.append(msg)

    def _merge_action_context(self, instruction: Instruction) -> Instruction:
        """Merge action response contents into instruction context."""
        # Extract unique action contents
        action_contents = []
        for response in self.pending_actions:
            content = getattr(response, "content", None)
            if content and content not in action_contents:
                action_contents.append(content)

        if not action_contents:
            return instruction

        # Merge into context
        existing = instruction.context or []
        if not isinstance(existing, list):
            existing = [existing]

        for content in action_contents:
            if content not in existing:
                existing.append(content)

        instruction.context = existing
        return instruction

    def _finalize_instruction(self) -> Instruction:
        """Finalize the new instruction with any pending actions."""
        if not self.pending_actions:
            return self.new_instruction

        # Merge remaining actions
        final = copy(self.new_instruction)
        return self._merge_action_context(final)


async def chat(
    branch: "Branch",
    instruction: JsonValue | Instruction = None,
    # Modern API: pass ChatContext directly
    chat_ctx: ChatContext = None,
    # Legacy API: individual parameters (backward compatible)
    guidance: str = None,
    context=None,
    sender: str = None,
    recipient: str = None,
    response_format: type[BaseModel] = None,
    progression: list = None,
    imodel: iModel = None,
    tool_schemas: list = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    plain_content: str = None,
    return_ins_res_message: bool = False,
    include_token_usage_to_model: bool = False,
    **kwargs,
) -> tuple[Instruction, AssistantResponse] | str:
    """
    Execute a chat operation with the branch.

    Two usage patterns:

    1. Modern (recommended):
        ctx = ChatContext(guidance="...", context={...}, ...)
        result = await chat(branch, instruction, chat_ctx=ctx)

    2. Legacy (backward compatible):
        result = await chat(branch, instruction, guidance="...", context={...}, ...)

    Args:
        branch: Branch instance for execution
        instruction: Instruction content
        chat_ctx: ChatContext object (modern API)
        guidance: Additional guidance text (legacy)
        context: Context data (legacy)
        sender: Message sender (legacy)
        recipient: Message recipient (legacy)
        response_format: Expected response format (legacy)
        progression: Message progression sequence (legacy)
        imodel: Model to use (legacy)
        tool_schemas: Tool schemas for function calling (legacy)
        images: Image attachments (legacy)
        image_detail: Image detail level (legacy)
        plain_content: Plain text content (legacy)
        return_ins_res_message: Return full instruction/response objects
        include_token_usage_to_model: Include token usage in model (legacy)
        **kwargs: Additional model parameters (legacy)

    Returns:
        AssistantResponse.response string or (Instruction, AssistantResponse) tuple
    """
    # Build ChatContext from whichever input was provided
    if chat_ctx is None:
        chat_ctx = ChatContext(
            guidance=guidance,
            context=context,
            sender=sender or branch.user or "user",
            recipient=recipient or branch.id,
            response_format=response_format,
            progression=progression,
            tool_schemas=tool_schemas or [],
            images=images or [],
            image_detail=image_detail or "auto",
            plain_content=plain_content or "",
            include_token_usage_to_model=include_token_usage_to_model,
            imodel=imodel or branch.chat_model,
            imodel_kw=kwargs,
        )

    # Build instruction params
    params = chat_ctx.to_dict(
        exclude={
            "imodel",
            "imodel_kw",
            "include_token_usage_to_model",
            "progression",
        }
    )
    params["sender"] = chat_ctx.sender or branch.user or "user"
    params["recipient"] = chat_ctx.recipient or branch.id
    params["instruction"] = instruction

    # Create prepared instruction
    prepared_instruction = branch.msgs.create_instruction(**params)

    # Build message history (single pass)
    progression_seq = chat_ctx.progression or branch.msgs.progression
    builder = MessageHistoryBuilder(
        branch, progression_seq, prepared_instruction
    )
    messages, final_instruction = builder.build()

    # Prepare model invocation
    kw = (chat_ctx.imodel_kw or {}).copy()
    kw["messages"] = [msg.chat_msg for msg in messages]

    model = chat_ctx.imodel or branch.chat_model
    invoke_method = model.stream if kw.get("stream") else model.invoke

    if invoke_method is model.invoke:
        kw["include_token_usage_to_model"] = (
            chat_ctx.include_token_usage_to_model
        )

    # Execute model call
    api_call = await invoke_method(**kw)
    branch._log_manager.log(Log.create(api_call))

    # Build response
    response = AssistantResponse.create(
        assistant_response=api_call.response,
        sender=branch.id,
        recipient=branch.user,
    )

    if return_ins_res_message:
        return final_instruction, response

    return response.response
