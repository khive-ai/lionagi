# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionagi.ln import (
    extract_json,
    fuzzy_validate_mapping,
    get_cancelled_exc_class,
    to_list,
)
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.operations.chat.chat import chat
from lionagi.protocols.messages.assistant_response import AssistantResponse
from lionagi.session.branch import AlcallParams
from lionagi.utils import copy

from ..types import ChatContext, HandleValidation, ParseContext

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


_DEFAULT_FORMAT_INSTRUCTION = "reformat text into specified model or structure"
_DEFAULT_FORMAT_GUIDANCE = (
    "follow the required response format, using the model schema as a guide"
)


class ParseValidator:
    """Handles validation and fuzzy matching for parsed results."""

    @staticmethod
    def normalize_fuzzy_params(
        fuzzy_params: FuzzyMatchKeysParams | dict | None,
    ) -> FuzzyMatchKeysParams | None:
        """Normalize fuzzy match parameters to FuzzyMatchKeysParams."""
        if fuzzy_params is None or fuzzy_params is False:
            return None
        if isinstance(fuzzy_params, FuzzyMatchKeysParams):
            return fuzzy_params
        if isinstance(fuzzy_params, dict):
            return FuzzyMatchKeysParams(**fuzzy_params)
        raise TypeError(
            "fuzzy_match_params must be a dict or FuzzyMatchKeysParams"
        )

    @staticmethod
    def extract_first_mapping(text: str) -> dict:
        """Extract first JSON object from text."""
        payload = extract_json(
            text, fuzzy_parse=True, return_one_if_single=False
        )
        candidates = to_list(payload, flatten=True, dropna=True)
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
        raise ValueError("No JSON object found in the provided text")

    @classmethod
    def validate_dict_or_model(
        cls,
        text: str,
        response_format: type[BaseModel] | dict,
        fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
    ):
        """Validate text as dict or Pydantic model with optional fuzzy matching."""
        try:
            fuzzy_match_params = cls.normalize_fuzzy_params(fuzzy_match_params)

            if response_format is None:
                return text

            model_format = (
                type(response_format)
                if isinstance(response_format, BaseModel)
                else response_format
            )

            mapping = cls.extract_first_mapping(text)

            if fuzzy_match_params:
                keys = (
                    model_format.model_fields
                    if isinstance(model_format, type)
                    and issubclass(model_format, BaseModel)
                    else model_format
                )
                mapping = fuzzy_validate_mapping(
                    mapping,
                    keys,
                    **fuzzy_match_params.to_dict(),
                )

            if isinstance(model_format, type) and issubclass(
                model_format, BaseModel
            ):
                return model_format.model_validate(mapping)

            return mapping

        except Exception as e:
            raise ValueError(f"Failed to parse text: {e}") from e


class ParseExecutor:
    """Executes parsing with retries and validation."""

    DEFAULT_ALCALL_PARAMS = AlcallParams(
        retry_initial_delay=1,
        retry_backoff=1.85,
        retry_attempts=3,
        max_concurrent=1,
        throttle_period=1,
    )

    def __init__(
        self,
        branch: "Branch",
        text: str,
        parse_ctx: ParseContext,
    ):
        self.branch = branch
        self.text = text
        self.parse_ctx = parse_ctx
        self.validator = ParseValidator()

    async def execute(
        self, return_res_message: bool = False
    ) -> Any | tuple[Any, AssistantResponse | None]:
        """Execute parsing with validation and retries."""
        response_format = self.parse_ctx.response_format
        fuzzy_params = self.validator.normalize_fuzzy_params(
            self.parse_ctx.fuzzy_match_params
        )

        # No formatting needed - return as-is
        if response_format is None:
            return (self.text, None) if return_res_message else self.text

        # Try direct validation first (fast path)
        with contextlib.suppress(Exception):
            result = self.validator.validate_dict_or_model(
                self.text, response_format, fuzzy_params
            )
            return result if not return_res_message else (result, None)

        # Need LLM parsing (slow path)
        return await self._parse_with_llm(
            response_format, fuzzy_params, return_res_message
        )

    async def _parse_with_llm(
        self,
        response_format,
        fuzzy_params,
        return_res_message: bool,
    ) -> Any | tuple[Any, AssistantResponse | None]:
        """Parse using LLM with retries."""
        instruction = (
            self.parse_ctx.format_instruction or _DEFAULT_FORMAT_INSTRUCTION
        )
        guidance = self.parse_ctx.format_guidance or _DEFAULT_FORMAT_GUIDANCE

        # Determine response format type
        if isinstance(response_format, dict):
            request_fields = response_format
            response_model_type = None
        elif isinstance(response_format, type) and issubclass(
            response_format, BaseModel
        ):
            response_model_type = response_format
            request_fields = None
        elif isinstance(response_format, BaseModel):
            response_model_type = type(response_format)
            request_fields = None
        else:
            response_model_type = None
            request_fields = None

        # Build context payload
        ctx_payload = (
            copy(self.parse_ctx.format_context)
            if self.parse_ctx.format_context is not None
            else [{"text_to_format": self.text}]
        )
        if self.parse_ctx.format_context is not None and not isinstance(
            ctx_payload, list
        ):
            raise TypeError(
                "ParseContext.format_context must be a list when provided"
            )

        # Parse function with retry
        async def _parse_attempt(i):
            chat_ctx = ChatContext(
                guidance=guidance,
                context=ctx_payload,
                sender=self.branch.user or "user",
                recipient=self.branch.id,
                response_format=response_model_type or request_fields,
                imodel=self.parse_ctx.imodel or self.branch.parse_model,
                imodel_kw=self.parse_ctx.imodel_kw or {},
            )

            _, res = await chat(
                self.branch,
                instruction,
                chat_ctx=chat_ctx,
                return_ins_res_message=True,
            )

            res.metadata["is_parsed"] = True
            res.metadata["original_text"] = self.text

            return (
                self.validator.validate_dict_or_model(
                    res.response,
                    response_format,
                    fuzzy_params,
                ),
                res,
            )

        # Execute with retries
        alcall_params = (
            self.parse_ctx.alcall_params or self.DEFAULT_ALCALL_PARAMS
        )
        if isinstance(self.parse_ctx.alcall_params, dict):
            alcall_params = AlcallParams(**self.parse_ctx.alcall_params)

        try:
            result = await alcall_params([0], _parse_attempt)
        except get_cancelled_exc_class():
            raise
        except Exception as e:
            return self._handle_parse_error(e, return_res_message)

        return (*result[0],) if return_res_message else result[0][0]

    def _handle_parse_error(
        self, error: Exception, return_res_message: bool
    ) -> Any | tuple[Any, None]:
        """Handle parsing errors based on validation strategy."""
        match self.parse_ctx.handle_validation:
            case "raise":
                raise ValueError(
                    f"Failed to parse response: {error}"
                ) from error
            case "return_none":
                return (None, None) if return_res_message else None
            case "return_value":
                return (self.text, None) if return_res_message else self.text


async def parse(
    branch: "Branch",
    text: str,
    # Modern API: pass ParseContext directly
    parse_ctx: ParseContext = None,
    # Legacy API: individual parameters (backward compatible)
    handle_validation: HandleValidation = "return_value",
    max_retries: int = 3,
    request_type: type[BaseModel] = None,
    response_format: type[BaseModel] | dict = None,
    similarity_algo: str = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal[
        "ignore", "raise", "remove", "fill", "force"
    ] = "force",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    fuzzy_match_params: FuzzyMatchKeysParams | dict | None = None,
    return_res_message: bool = False,
    **kw,
) -> Any | tuple[Any, AssistantResponse | None]:
    """
    Parse text into structured format (dict or Pydantic model).

    Two usage patterns:

    1. Modern (recommended):
        ctx = ParseContext(response_format=MyModel, ...)
        result = await parse(branch, text, parse_ctx=ctx)

    2. Legacy (backward compatible):
        result = await parse(branch, text, request_type=MyModel, ...)

    Args:
        branch: Branch instance for execution
        text: Text to parse
        parse_ctx: ParseContext object (modern API)
        handle_validation: Error handling strategy (legacy)
        max_retries: Maximum retry attempts (legacy)
        request_type: Expected response type (legacy, alias for response_format)
        response_format: Expected response format (legacy)
        similarity_algo: Fuzzy match algorithm (legacy)
        similarity_threshold: Fuzzy match threshold (legacy)
        fuzzy_match: Enable fuzzy matching (legacy)
        handle_unmatched: How to handle unmatched keys (legacy)
        fill_value: Default value for missing fields (legacy)
        fill_mapping: Mapping for missing fields (legacy)
        strict: Strict validation mode (legacy)
        fuzzy_match_params: Fuzzy match configuration (legacy)
        return_res_message: Return AssistantResponse with result
        **kw: Additional model parameters (legacy)

    Returns:
        Parsed result or (result, AssistantResponse) tuple
    """
    # Build ParseContext from whichever input was provided
    if parse_ctx is None:
        # Normalize fuzzy params
        fuzzy_params = ParseValidator.normalize_fuzzy_params(
            fuzzy_match_params
        )
        if fuzzy_params is None:
            fuzzy_params = FuzzyMatchKeysParams(
                similarity_algo=similarity_algo,
                similarity_threshold=similarity_threshold,
                handle_unmatched=handle_unmatched,
                fill_value=fill_value,
                fill_mapping=fill_mapping,
                strict=strict,
                fuzzy_match=fuzzy_match,
            )
        elif fuzzy_params.fuzzy_match != fuzzy_match:
            fuzzy_params = fuzzy_params.with_updates(fuzzy_match=fuzzy_match)

        # Build ParseContext
        parse_ctx = ParseContext(
            response_format=response_format or request_type,
            fuzzy_match_params=fuzzy_params,
            handle_validation=handle_validation,
            alcall_params=ParseExecutor.DEFAULT_ALCALL_PARAMS.with_updates(
                retry_attempts=max_retries
            ),
            imodel=branch.parse_model,
            imodel_kw=kw,
        )

    # Execute parsing
    executor = ParseExecutor(branch, text, parse_ctx)
    return await executor.execute(return_res_message=return_res_message)
