"""Utility functions specific to lionagi.

This module contains utility functions that are specific to lionagi and not
covered by lionfuncs. These utilities should be minimal, as most general-purpose
utilities should be provided by lionfuncs.
"""

from typing import Any, Dict, List, Optional, Union

import lionfuncs.utils as utils
from lionfuncs.errors import LionError


def is_tool_call_message(message_content: Any) -> bool:
    """Check if a message content represents a tool call.

    Args:
        message_content: The content of the message to check.

    Returns:
        True if the message content represents a tool call, False otherwise.
    """
    # TODO: Implement tool call detection logic
    # This will depend on the specific format of tool calls in the LLM responses
    raise NotImplementedError("This function is not yet implemented.")


def extract_tool_calls(message_content: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from a message content.

    Args:
        message_content: The content of the message to extract tool calls from.

    Returns:
        A list of tool calls extracted from the message content.

    Raises:
        LionError: If the message content does not contain tool calls.
    """
    # TODO: Implement tool call extraction logic
    # This will depend on the specific format of tool calls in the LLM responses
    raise NotImplementedError("This function is not yet implemented.")


# Add any other lionagi-specific utilities as needed