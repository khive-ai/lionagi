# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Action request/response validation rules for structured function calling.

This module provides rules for validating action requests and responses,
which are used in tool/function calling scenarios where the model needs
to request specific actions with parameters.
"""

from typing import Any, Dict, List

from lionagi._errors import ValidationError
from lionagi.fields.action import (
    ActionRequestModel,
    ActionResponseModel,
    parse_action_request,
)

from .base import Rule, register_rule


class ActionRequestRule(Rule):
    """Rule for validating action requests using ActionRequestModel."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to action_request fields."""
        action_types = {"action", "action_request", "tool_call"}
        if annotation and annotation.lower() in action_types:
            return True

        # Check for action-related field names
        action_fields = {
            "action_request",
            "actions",
            "tool_calls",
            "function_calls",
        }
        return field in action_fields

    async def validate(self, value: Any, **kwargs) -> ActionRequestModel:
        """Validate action request using ActionRequestModel.

        Args:
            value: The action request to validate
            **kwargs: Validation options

        Returns:
            Validated ActionRequestModel instance

        Raises:
            ValidationError: If action request is invalid
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            if kwargs.get("default") is not None:
                return kwargs["default"]
            raise ValidationError("Action request cannot be None")

        # If already an ActionRequestModel, return it
        if isinstance(value, ActionRequestModel):
            return value

        try:
            # If it's a string, use the parse_action_request function
            if isinstance(value, str):
                parsed_actions = parse_action_request(value)
                if not parsed_actions:
                    raise ValidationError(
                        "No valid action requests found in string"
                    )
                # Return the first valid action request
                return ActionRequestModel.model_validate(parsed_actions[0])

            # If it's a dict, validate directly
            elif isinstance(value, dict):
                return ActionRequestModel.model_validate(value)

            # If it's a list, take the first item
            elif isinstance(value, list) and value:
                return ActionRequestModel.model_validate(value[0])

            else:
                raise ValidationError(
                    f"Cannot convert {type(value)} to ActionRequestModel"
                )

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid action request: {e}")

    async def fix(self, value: Any, **kwargs) -> ActionRequestModel:
        """Attempt to fix action request.

        Args:
            value: Value to fix
            **kwargs: Fix options

        Returns:
            Fixed ActionRequestModel instance
        """
        if value is None:
            # Return default if provided
            default = kwargs.get("default")
            if default is not None:
                return default
            # Create minimal action request
            return ActionRequestModel(function=None, arguments=None)

        try:
            # Try parsing with more aggressive approach
            if isinstance(value, str):
                parsed_actions = parse_action_request(value)
                if parsed_actions:
                    return ActionRequestModel.model_validate(parsed_actions[0])
                # If parsing failed, create minimal action
                return ActionRequestModel(function=None, arguments=None)

            elif isinstance(value, dict):
                # Fill in missing required fields
                fixed_value = value.copy()
                if "function" not in fixed_value:
                    fixed_value["function"] = None
                if "arguments" not in fixed_value:
                    fixed_value["arguments"] = None
                return ActionRequestModel.model_validate(fixed_value)

            else:
                # Last resort - create minimal action
                return ActionRequestModel(function=None, arguments=None)

        except:
            # Final fallback
            return ActionRequestModel(function=None, arguments=None)


@register_rule("action_response")
class ActionResponseRule(Rule):
    """Rule for validating action responses using ActionResponseModel."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to action_response fields."""
        response_types = {"action_response", "tool_response"}
        if annotation and annotation.lower() in response_types:
            return True

        response_fields = {"action_responses", "tool_responses", "results"}
        return field in response_fields

    async def validate(self, value: Any, **kwargs) -> ActionResponseModel:
        """Validate action response using ActionResponseModel.

        Args:
            value: The action response to validate
            **kwargs: Validation options

        Returns:
            Validated ActionResponseModel instance

        Raises:
            ValidationError: If response is invalid
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            if kwargs.get("default") is not None:
                return kwargs["default"]
            raise ValidationError("Action response cannot be None")

        # If already an ActionResponseModel, return it
        if isinstance(value, ActionResponseModel):
            return value

        try:
            # If it's a dict, validate directly
            if isinstance(value, dict):
                return ActionResponseModel.model_validate(value)

            # If it's not a dict, wrap it as output
            else:
                return ActionResponseModel(
                    function="", arguments={}, output=value
                )

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid action response: {e}")

    async def fix(self, value: Any, **kwargs) -> ActionResponseModel:
        """Attempt to fix action response.

        Args:
            value: Value to fix
            **kwargs: Fix options

        Returns:
            Fixed ActionResponseModel instance
        """
        if value is None:
            # Return default if provided
            default = kwargs.get("default")
            if default is not None:
                return default
            # Create empty response
            return ActionResponseModel()

        try:
            if isinstance(value, dict):
                # Fill in missing fields with defaults
                fixed_value = value.copy()
                if "function" not in fixed_value:
                    fixed_value["function"] = ""
                if "arguments" not in fixed_value:
                    fixed_value["arguments"] = {}
                return ActionResponseModel.model_validate(fixed_value)

            else:
                # Wrap as output
                return ActionResponseModel(
                    function="", arguments={}, output=value
                )

        except:
            # Final fallback
            return ActionResponseModel()
