# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import inspect
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, JsonValue

from lionagi.ln.fuzzy import string_similarity
from lionagi.protocols.types import Instruction
from lionagi.utils import is_same_dtype

from .types import ChatContext


class SelectionModel(BaseModel):
    """Model representing the selection output."""

    PROMPT: ClassVar[str] = (
        "Please select up to {max_num_selections} items from the following list {choices}. "
        "Provide the selection(s) into appropriate field in format required, and no comments from you"
    )

    selected: list[Any] = Field(default_factory=list)


def parse_to_representation(
    choices: Enum | dict | list | tuple | set,
) -> tuple[list[str], JsonValue]:
    """
    Parse choices into (keys, representations) tuple.

    Supports:
    1. Iterator of string | BaseModel
    2. dict[str, JsonValue | BaseModel]
    3. Enum[str, JsonValue | BaseModel]
    """

    if isinstance(choices, tuple | set | list):
        choices = list(choices)
        if is_same_dtype(choices, str):
            return choices, choices

    if isinstance(choices, list):
        if is_same_dtype(choices, BaseModel):
            choices = {i.__class__.__name__: i for i in choices}
        if all(
            inspect.isclass(i) and issubclass(i, BaseModel) for i in choices
        ):
            choices = {i.__name__: i for i in choices}

    if isinstance(choices, type) and issubclass(choices, Enum):
        keys = [i.name for i in choices]
        contents = [get_choice_representation(i) for i in choices]
        return keys, contents

    if isinstance(choices, dict):
        keys = list(choices.keys())
        contents = [get_choice_representation(v) for k, v in choices.items()]
        return keys, contents

    if isinstance(choices, tuple | set | list):
        choices = list(choices)
        if is_same_dtype(choices, str):
            return choices, choices

    raise NotImplementedError(
        "Choices must be list/tuple/set of strings, dict, or Enum"
    )


def get_choice_representation(choice: Any) -> str:
    """Get string representation of a choice."""
    if isinstance(choice, str):
        return choice

    if isinstance(choice, BaseModel):
        return f"{choice.__class__.__name__}:\n{choice.model_json_schema(indent=2)}"

    if isinstance(choice, Enum):
        return get_choice_representation(choice.value)

    return str(choice)


def parse_selection(selection_str: str, choices: Any):
    """Parse selection string back to original choice value."""
    select_from = []

    if isinstance(choices, dict):
        select_from = list(choices.keys())

    if inspect.isclass(choices) and issubclass(choices, Enum):
        select_from = [choice.name for choice in choices]

    if isinstance(choices, list | tuple | set):
        if is_same_dtype(choices, BaseModel):
            select_from = [i.__class__.__name__ for i in choices]
        if is_same_dtype(choices, str):
            select_from = list(choices)
        if all(
            inspect.isclass(i) and issubclass(i, BaseModel) for i in choices
        ):
            select_from = [i.__name__ for i in choices]

    if not select_from:
        raise ValueError("The values provided for choice is not valid")

    selected = string_similarity(
        selection_str, select_from, return_most_similar=True
    )

    if isinstance(choices, dict) and selected in choices:
        return choices[selected]

    if inspect.isclass(choices) and issubclass(choices, Enum):
        for i in choices:
            if i.name == selected:
                return i

    if isinstance(choices, list) and is_same_dtype(choices, str):
        if selected in choices:
            return selected

    return selected


async def select(
    branch,
    instruction: JsonValue | Instruction,
    choices: list[str] | type[Enum] | dict[str, Any],
    chat_ctx: ChatContext | None = None,
    max_num_selections: int = 1,
    verbose: bool = False,
    **operate_kwargs: Any,
) -> SelectionModel:
    """
    Select items from choices using LLM.

    Args:
        branch: Branch instance to use for selection
        instruction: User instruction for selection task
        choices: Available choices (list, dict, or Enum)
        chat_ctx: Chat context for customization (optional)
        max_num_selections: Maximum number of items to select
        verbose: Print progress messages
        **operate_kwargs: Additional kwargs passed to branch.operate()

    Returns:
        SelectionModel with corrected selections
    """
    if verbose:
        print(f"Starting selection with up to {max_num_selections} choices.")

    # Parse choices into keys and representations
    selections, contents = parse_to_representation(choices)
    prompt = SelectionModel.PROMPT.format(
        max_num_selections=max_num_selections, choices=selections
    )

    # Build instruction dict
    if isinstance(instruction, Instruction):
        instruct_dict = {
            "instruction": instruction.instruction,
            "context": instruction.context,
            "guidance": instruction.guidance,
        }
    else:
        instruct_dict = {"instruction": str(instruction)}

    # Append selection prompt to instruction
    current_instruction = instruct_dict.get("instruction", "")
    instruct_dict["instruction"] = f"{current_instruction}\n\n{prompt}\n\n"

    # Add choice representations to context
    context = instruct_dict.get("context") or []
    context = [context] if not isinstance(context, list) else context
    context.extend([{k: v} for k, v in zip(selections, contents)])
    instruct_dict["context"] = context

    # Use provided chat_ctx or create minimal one
    if chat_ctx:
        _chat_ctx = ChatContext(**chat_ctx.to_dict())
    else:
        _chat_ctx = ChatContext()

    _chat_ctx.response_format = SelectionModel

    # Call branch.operate with proper context
    from .operate import operate

    response_model: SelectionModel = await operate(
        branch,
        instruction=instruct_dict,
        chat_ctx=_chat_ctx,
        **operate_kwargs,
    )

    if verbose:
        print(f"Received selection: {response_model.selected}")

    # Parse selections back to original choice values
    selected = response_model.selected
    selected = [selected] if not isinstance(selected, list) else selected

    corrected_selections = [parse_selection(i, choices) for i in selected]

    if isinstance(response_model, BaseModel):
        response_model.selected = corrected_selections
    elif isinstance(response_model, dict):
        response_model["selected"] = corrected_selections

    return response_model
