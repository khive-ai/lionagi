from pydantic import ConfigDict

from .hashable_model import HashableModel

__all__ = ("BaseModel",)


class BaseModel(HashableModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        populate_by_name=True,
    )
