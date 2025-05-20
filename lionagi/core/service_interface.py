# lionagi.core.service_interface - Abstract representation of an LLM service
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

from typing import Any

from pydantic import BaseModel, Field

# from ..models.message import Message # Forward reference or direct import later
# from ..models.tool import Tool # Forward reference or direct import later


class ServiceInterface(BaseModel):
    """
    Abstract representation of a configured LLM service interface.
    This model holds configuration details. The actual call logic
    will be handled by implementations that use lionfuncs.network.adapters.
    It is not intended to be directly serialized as a whole state object,
    but its parameters are part of Branch/Session state.
    """

    provider: str
    default_model: str

    # Common configurations that might be passed to lionfuncs adapters
    # These are examples; actual parameters will depend on lionfuncs adapter capabilities
    api_key: str | None = Field(
        None, exclude=True
    )  # API keys should not be serialized
    max_tokens: int | None = None
    temperature: float | None = None
    # ... other potential common parameters like top_p, presence_penalty etc.

    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Placeholder for the method that would interact with lionfuncs adapter
    # async def call_llm(self, messages: List[Message], tools: Optional[List[Tool]] = None, **kwargs) -> Any:
    #     """
    #     This method would be implemented by a concrete class that wraps
    #     a lionfuncs network adapter.
    #     """
    #     raise NotImplementedError

    class Config:
        arbitrary_types_allowed = True


__all__ = ["ServiceInterface"]
