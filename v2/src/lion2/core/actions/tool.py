from collections.abc import Callable
from datetime import datetime
from typing import Any, Type

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from pydapter.protocols import Temporal
from lionfuncs import (
    is_coro_func,
    pydantic_model_to_openai_schema,
    function_to_openai_schema,
)
from lion2.core.types import GenericParams

from lion2.errors import ToolParameterValidationError


class Tool(Temporal):
    """A tool that can be called by an agent or system."""

    name: str
    """The unique name of the tool."""

    description: str
    """A detailed description of what the tool does."""

    last_used: datetime | None = None
    """Timestamp of when the tool was last used."""

    openai_schema: dict[str, Any] | None = None
    """OpenAI-compatible schema for the tool."""

    # excluded from the serialization
    func_callable: Callable[..., Any] = Field(
        description="The callable function to be executed by the tool.",
        exclude=True,
    )

    parameters_schema: Type[BaseModel] = Field(
        default_factory=lambda: GenericParams,
        description="Pydantic model defining the expected parameters for the tool.",
        exclude=True,
    )

    @field_validator("parameters_schema", mode="before")
    def _validate_parameters_schema(cls, v: Any) -> Type[BaseModel]:
        if v is not None:
            from ..utils import validate_model_to_type
            return validate_model_to_type(v)
        return GenericParams

    @field_validator("last_used", mode="before")
    def _validate_last_used(cls, value: Any) -> datetime | None:
        return cls._validate_datetime(value) if value else None

    @field_serializer("last_used")
    def _serialize_last_used(self, value: datetime | None) -> str | None:
        return self._serialize_datetime(value) if value else None

    def update_last_used(self) -> None:
        """Update the last used timestamp of the tool."""
        self.last_used = datetime.now()

    def validate_parameters(self, params: dict[str, Any]) -> BaseModel:
        """Validates input parameters against parameters_schema."""
        try:
            return self.parameters_schema(**params)
        except ValidationError as e:
            raise ToolParameterValidationError(
                f"Invalid parameters for tool '{self.name}': {e}"
            ) from e

    async def invoke(self, validated_params: BaseModel) -> Any:
        """
        Invokes the tool's underlying callable with validated parameters.
        Subclasses might override this for more complex invocation logic.
        """
        param_kwargs = validated_params.model_dump()
        self.update_last_used()
        if is_coro_func(self.func_callable):
            return await self.func_callable(**param_kwargs)
        else:
            return self.func_callable(**param_kwargs)

    @property
    def function_name(self) -> str:
        """Return the function name (which is the tool's name)."""
        return self.name

    @property
    def required_fields(self) -> set[str]:
        """Return the set of required parameter names from the parameters_schema."""
        if self.parameters_schema is GenericParams:
            return set(self.openai_schema["function"]["parameters"]["required"])

        required = set()
        for field_name, field_info in self.parameters_schema.model_fields.items():
            if field_info.is_required():
                required.add(field_name)
        return required

    @model_validator(mode="after")
    def _validate_openai_tool_schema(self):
        if self.openai_schema is None:
            if self.parameters_schema is GenericParams:
                self.openai_schema = function_to_openai_schema(self.func_callable)

                if self.name is None:
                    self.name = self.openai_schema["function"]["name"]
                if self.description is None:
                    self.description = self.openai_schema["function"]["description"]
            else:
                self.openai_schema = pydantic_model_to_openai_schema(
                    model_class=self.parameters_schema,
                    function_name=self.name,
                    function_description=self.description
                )
        return self

FuncTool = Callable | Tool
FuncToolRef = FuncTool | str
