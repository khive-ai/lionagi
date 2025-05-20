# lionagi.models.tool - Tool Pydantic model
# Copyright (c) 2023-present, HaiyangLi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

# Assuming Temporal and Adaptable are available or placeholder defined
# from pydapter.protocols import Temporal
# from pydapter.core import Adaptable

try:
    from pydapter.protocols import Temporal
except ImportError:

    class Temporal(BaseModel):  # type: ignore
        id: str = Field(default_factory=lambda: "temp_id")
        created_at: Any = Field(default_factory=lambda: "timestamp")
        updated_at: Any = Field(default_factory=lambda: "timestamp")

        class Config:
            arbitrary_types_allowed = True


try:
    from pydapter.core import Adaptable
except ImportError:

    class Adaptable:  # type: ignore
        pass


class Tool(Temporal, Adaptable):
    """
    Represents a tool that can be called by an LLM.
    Tools can be versioned and need to be serializable for configuration.
    The actual callable function (`func_ref`) is a runtime attribute
    and not part of the core serialization.
    """

    name: str
    description: str | None = None

    # Pydantic model for runtime validation of arguments passed by LLM
    # This is not directly serialized but used to generate schema_
    # and for runtime validation by ActionManager.
    argument_model: type[BaseModel] | None = Field(None, exclude=True)

    # JSON schema for the LLM, derived from argument_model or func.
    # Stored with alias "schema" for LLM compatibility.
    schema_: dict[str, Any] = Field(alias="schema")

    # Runtime attribute for the actual callable function.
    # Not serialized as it's a code reference.
    # ActionManager will hold the actual callable.
    # func_ref: Optional[Callable] = Field(None, exclude=True) # As per TDS, func_ref is not part of model

    class Config:
        arbitrary_types_allowed = True
        # Allows 'schema_' to be aliased as 'schema' during serialization/deserialization
        # and also allows fields like `argument_model` which are Types.
        protected_namespaces = ()


__all__ = ["Tool"]
