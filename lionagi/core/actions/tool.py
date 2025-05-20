from collections.abc import Callable
from datetime import datetime
from typing import Any

from lionfuncs.utils import force_async
from pydantic import Field, field_serializer, field_validator, model_validator
from pydapter.protocols import Temporal
from typing_extensions import Self


class Tool(Temporal):
    """A tool that can be called by the LLM."""

    last_used: datetime | None = None
    func_callable: Callable[..., Any] = Field(
        description="The callable function to be wrapped by the tool",
        exclude=True,
    )
    tool_schema: dict[str, Any] | None = Field(
        default=None,
        description="Schema describing the function's parameters and structure",
    )
    request_options: type | None = Field(
        default=None,
        description="Optional Pydantic model for validating the function's input",
        exclude=True,
    )
    strict_func_call: bool = Field(
        default=False,
        description="Whether to enforce strict validation of function parameters",
    )

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        func = force_async(self.func_callable)
        return await func(*args, **kwargs)

    @field_validator("last_used", mode="before")
    def _validate_last_used(cls, value: Any) -> datetime | None:
        return cls._validate_datetime(value) if value else None

    @field_serializer("last_used")
    def _serialize_last_used(self, value: datetime | None) -> str | None:
        return self._serialize_datetime(value) if value else None

    def update_last_used(self) -> None:
        """Update the last used timestamp of the tool."""
        self.last_used = datetime.now()

    @model_validator(mode="after")
    def _validate_tool_schema(self) -> Self:
        if self.tool_schema is None:
            if self.request_options is not None:
                from lionfuncs.schema_utils import (
                    pydantic_model_to_openai_schema,
                )

                self.tool_schema = pydantic_model_to_openai_schema(
                    self.request_options
                )

            else:
                from lionfuncs.schema_utils import function_to_openai_schema

                self.tool_schema = function_to_openai_schema(
                    self.func_callable, request_options=self.request_options
                )

        return self

    @property
    def function(self) -> str:
        """Return the function name from the auto-generated schema."""
        return self.tool_schema["function"]["name"]

    @property
    def required_fields(self) -> set[str]:
        """Return the set of required parameter names from the schema."""
        return set(self.tool_schema["function"]["parameters"]["required"])

    @property
    def minimum_acceptable_fields(self) -> set[str]:
        """
        Return the set of parameters that have no default values,
        ignoring `*args` or `**kwargs`.
        """
        import inspect

        try:
            a = {
                k
                for k, v in inspect.signature(
                    self.func_callable
                ).parameters.items()
                if v.default == inspect.Parameter.empty
            }
            if "kwargs" in a:
                a.remove("kwargs")
            if "args" in a:
                a.remove("args")
            return a
        except Exception:
            return set()
