from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .operations.fields import (
        ActionRequestModel,
        ActionResponseModel,
        Instruct,
        Reason,
        get_default_field,
    )

_lazy_imports = {}


def __getattr__(name: str):
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "ActionRequestModel":
        from .operations.fields import ActionRequestModel

        _lazy_imports[name] = ActionRequestModel
        return ActionRequestModel

    if name == "ActionResponseModel":
        from .operations.fields import ActionResponseModel

        _lazy_imports[name] = ActionResponseModel
        return ActionResponseModel

    if name == "Instruct":
        from .operations.fields import Instruct

        _lazy_imports[name] = Instruct
        return Instruct

    if name == "Reason":
        from .operations.fields import Reason

        _lazy_imports[name] = Reason
        return Reason

    from .operations.fields import get_default_field

    if name == "get_default_field":

        _lazy_imports[name] = get_default_field
        return get_default_field

    if name == "ACTION_REQUESTS_FIELD":
        return get_default_field("action_requests")

    if name == "ACTION_RESPONSES_FIELD":
        return get_default_field("action_responses")

    if name == "ACTION_REQUIRED_FIELD":
        return get_default_field("action_required")

    if name == "INSTRUCT_FIELD":
        return get_default_field("instruct")

    if name == "LIST_INSTRUCT_FIELD_MODEL":
        return get_default_field("instruct", listable=True)

    if name == "REASON_FIELD":
        return get_default_field("reason")

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = (
    "ACTION_REQUESTS_FIELD",
    "ACTION_RESPONSES_FIELD",
    "ACTION_REQUIRED_FIELD",
    "INSTRUCT_FIELD",
    "LIST_INSTRUCT_FIELD_MODEL",
    "REASON_FIELD",
    "ActionRequestModel",
    "ActionResponseModel",
    "Instruct",
    "Reason",
    "get_default_field",
)
