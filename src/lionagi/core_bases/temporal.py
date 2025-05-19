from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer, field_validator


from lionagi.core_utils import convert_to_datetime, validate_uuid

__all__ = ("Temporal",)

class Temporal(BaseModel): # No longer inherits Identifiable
    """
    Base class for objects with a unique identifier and created/updated timestamps
    using datetime objects.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the element.",
        frozen=True,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp for the element.",
        frozen=True,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last updated timestamp for the element.",
    )

    @field_serializer("id")
    def _serialize_id(self, v: UUID) -> str:
        return str(v)

    @field_validator("id", mode="before")
    def _validate_id(cls, v: str | UUID) -> UUID:
        return validate_uuid(v)

    def update_timestamp(self) -> None:
        """Update the last updated timestamp to the current time."""
        self.updated_at = datetime.now(timezone.utc)

    @field_serializer("updated_at", "created_at")
    def _serialize_datetime(self, v: datetime) -> str:
        return v.isoformat()

    @field_validator("updated_at", "created_at", mode="before")
    def _validate_datetime(cls, v: str | datetime) -> datetime:
        return convert_to_datetime(v)

    def __hash__(self) -> int:
        """Returns the hash of the object based on its ID."""
        return hash(self.id)