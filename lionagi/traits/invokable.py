from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .types import Execution


class Invokable(BaseModel):
    """An executable can be invoked with a request"""

    request: dict | None = None
    execution: Execution = Field(default_factory=Execution)
    response_obj: Any = Field(None, exclude=True)

    @abstractmethod
    async def invoke(self):
        pass
