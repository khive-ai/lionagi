from enum import Enum
from typing import List, Dict, Optional, Any # Added Any for _parse_embedding_response

# Added Field to pydantic import, and orjson
from pydantic import BaseModel, Field, field_validator, field_serializer
import orjson # Renamed to orjson directly, instead of 'as json' to avoid conflict if stdlib json is used elsewhere

# Assuming core_utils.py is at the same level src/lionagi/core_utils.py
# For robustness, using absolute import path from package root
from lionagi.core_utils import validate_model_to_dict

__all__ = (
    "Embedding",
    "Metadata",
    "ExecutionStatus",
    "Execution",
    "StructuredLog",
    "Embedable", # Added Embedable
)

Embedding = List[float] # Using List from typing
Metadata = Dict        # Using Dict from typing


class ExecutionStatus(str, Enum):
    """Status states for tracking action execution progress."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Execution(BaseModel):
    """Represents the execution state of an event."""

    duration: Optional[float] = None
    response: Optional[Dict] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None

    @field_validator("response", mode="before")
    def _validate_response(cls, v: Optional[BaseModel | Dict]):
        return validate_model_to_dict(v)

    @field_serializer("status")
    def _serialize_status(self, v: ExecutionStatus) -> str:
        return v.value


class StructuredLog(BaseModel):
    """Represents a structured log entry."""
    id: str
    created_at: str
    updated_at: str
    event_type: str
    content: Optional[str] = None
    embedding: Embedding = []
    duration: Optional[float] = None
    status: str
    error: Optional[str] = None
    sha256: Optional[str] = None


# Content from protocols_2/embedable.py starts here
class Embedable(BaseModel):
    """Embedable trait, contains embedding and content"""

    content: Optional[str] = None
    embedding: Embedding = Field(default_factory=list)

    @property
    def n_dim(self) -> int:
        """Get the number of dimensions of the embedding."""
        return len(self.embedding)

    @field_validator("embedding", mode="before")
    def _parse_embedding(cls, value: Optional[List[float] | str]) -> Embedding: # Adjusted type hint for value
        if value is None:
            return []
        if isinstance(value, str):
            try:
                # Using orjson directly as imported
                loaded = orjson.loads(value)
                return [float(x) for x in loaded]
            except Exception as e:
                raise ValueError("Invalid embedding string.") from e
        if isinstance(value, list):
            try:
                return [float(x) for x in value]
            except Exception as e:
                raise ValueError("Invalid embedding list.") from e
        raise ValueError("Invalid embedding type; must be list or JSON-encoded string.")

    def create_content(self):
        """override in child class to support custom content creation"""
        return self.content


def _parse_embedding_response(x: Any) -> Any: # Added type hint for x
    # parse openai response
    if (
        isinstance(x, BaseModel)
        and hasattr(x, "data")
        and isinstance(getattr(x, 'data', None), list) # Added check for list
        and len(x.data) > 0
        and hasattr(x.data[0], "embedding")
    ):
        return x.data[0].embedding

    if isinstance(x, list | tuple):
        if len(x) > 0 and all(isinstance(i, float) for i in x):
            return x
        if len(x) == 1 and isinstance(x[0], dict | BaseModel):
            return _parse_embedding_response(x[0])

    # parse dict response
    if isinstance(x, dict):
        # parse openai format response
        if "data" in x:
            data = x.get("data")
            if data is not None and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict): # Added check for list
                return _parse_embedding_response(data[0])

        # parse {"embedding": []} response
        if "embedding" in x:
            return _parse_embedding_response(x["embedding"])

    return x