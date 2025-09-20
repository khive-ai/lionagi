from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from lionagi.ln import hash_dict, not_sentinel


class HashableModel(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
    )

    def to_dict(self, **kwargs) -> dict:
        """provides interface, specific methods need to be implemented in subclass kwargs for pydantic model_dump"""
        return {
            k: v
            for k, v in self.model_dump(**kwargs).items()
            if not_sentinel(v)
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> Self:
        """provides interface, specific methods need to be implemented in subclass kwargs for pydantic model_validate"""
        return cls.model_validate(data, **kwargs)

    def __hash__(self):
        # Convert kwargs to a hashable format by serializing unhashable types
        return hash_dict(self.to_dict())
