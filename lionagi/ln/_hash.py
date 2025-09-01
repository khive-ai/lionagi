from __future__ import annotations

import copy

import msgspec
from pydantic import BaseModel as PydanticBaseModel

__all__ = ("hash_dict",)

# --- Canonical Representation Generator ---
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))
_TYPE_MARKER_DICT = 0
_TYPE_MARKER_LIST = 1
_TYPE_MARKER_TUPLE = 2
_TYPE_MARKER_SET = 3
_TYPE_MARKER_FROZENSET = 4
_TYPE_MARKER_PYDANTIC = 5  # Distinguishes dumped Pydantic models
_TYPE_MARKER_MSGSPEC = 6  # Distinguishes msgspec Structs


def _generate_hashable_representation(item: any) -> any:
    """
    Recursively converts a Python object into a stable, hashable representation.
    This ensures that logically identical but structurally different inputs
    (e.g., dicts with different key orders) produce the same representation.
    """
    if isinstance(item, _PRIMITIVE_TYPES):
        return item

    # Handle msgspec Structs
    if isinstance(item, msgspec.Struct):
        # Use msgspec.to_builtins for efficient conversion to built-in types
        return (
            _TYPE_MARKER_MSGSPEC,
            _generate_hashable_representation(msgspec.to_builtins(item)),
        )

    if isinstance(item, PydanticBaseModel):
        # Process the Pydantic model by first dumping it to a dict, then processing that dict.
        # The type marker distinguishes this from a regular dictionary.
        return (
            _TYPE_MARKER_PYDANTIC,
            _generate_hashable_representation(item.model_dump()),
        )

    if isinstance(item, dict):
        # Sort dictionary items by key (stringified) for order-insensitivity.
        return (
            _TYPE_MARKER_DICT,
            tuple(
                (str(k), _generate_hashable_representation(v))
                for k, v in sorted(item.items(), key=lambda x: str(x[0]))
            ),
        )

    if isinstance(item, list):
        return (
            _TYPE_MARKER_LIST,
            tuple(_generate_hashable_representation(elem) for elem in item),
        )

    if isinstance(item, tuple):
        return (
            _TYPE_MARKER_TUPLE,
            tuple(_generate_hashable_representation(elem) for elem in item),
        )

    # frozenset must be checked before set
    if isinstance(item, frozenset):
        try:  # Attempt direct sort for comparable elements
            sorted_elements = sorted(list(item))
        except TypeError:  # Fallback for unorderable mixed types

            def sort_key(x):
                # Deterministic ordering across mixed, unorderable types
                # Sort strictly by textual type then textual value.
                # This also naturally places bool before int because
                # "<class 'bool'>" < "<class 'int'>" lexicographically.
                return (str(type(x)), str(x))

            sorted_elements = sorted(list(item), key=sort_key)
        return (
            _TYPE_MARKER_FROZENSET,
            tuple(
                _generate_hashable_representation(elem)
                for elem in sorted_elements
            ),
        )

    if isinstance(item, set):
        try:
            sorted_elements = sorted(list(item))
        except TypeError:
            # For mixed types, use a deterministic, portable sort key
            def sort_key(x):
                # Sort by textual type then textual value for stability.
                return (str(type(x)), str(x))

            sorted_elements = sorted(list(item), key=sort_key)
        return (
            _TYPE_MARKER_SET,
            tuple(
                _generate_hashable_representation(elem)
                for elem in sorted_elements
            ),
        )

    # Fallback for other types (e.g., custom objects not derived from the above)
    try:
        return str(item)
    except Exception:
        try:
            return repr(item)
        except Exception:
            # If both str() and repr() fail, return a stable fallback based on type and id
            return f"<unhashable:{type(item).__name__}:{id(item)}>"


def hash_dict(data: any, strict: bool = False) -> int:
    data_to_process = data
    if strict:
        data_to_process = copy.deepcopy(data)

    hashable_repr = _generate_hashable_representation(data_to_process)

    try:
        return hash(hashable_repr)
    except TypeError as e:
        raise TypeError(
            f"The generated representation for the input data was not hashable. "
            f"Input type: {type(data).__name__}, Representation type: {type(hashable_repr).__name__}. "
            f"Original error: {e}"
        )
