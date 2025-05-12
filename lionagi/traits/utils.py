from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


def convert_to_datetime(v):
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except Exception:
            pass

    error_msg = "Input value for field <created_at> should be a `datetime.datetime` object or `isoformat` string"
    raise ValueError(error_msg)


def validate_uuid(v: str | UUID) -> UUID:
    if isinstance(v, UUID):
        return v
    try:
        return UUID(str(v))
    except Exception as e:
        error_msg = "Input value for field <id> should be a `uuid.UUID` object or a valid `uuid` representation"
        raise ValueError(error_msg) from e


def validate_model_to_dict(v):
    """Serialize a Pydantic model to a dictionary. kwargs are passed to model_dump."""

    if isinstance(v, BaseModel):
        return v.model_dump()
    if v is None:
        return {}
    if isinstance(v, dict):
        return v

    error_msg = "Input value for field <model> should be a `pydantic.BaseModel` object or a `dict`"
    raise ValueError(error_msg)
