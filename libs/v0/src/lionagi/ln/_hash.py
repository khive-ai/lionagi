import copy

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


def _generate_hashable_representation(item: any) -> any:
    """
    Recursively converts a Python object into a stable, hashable representation.
    This ensures that logically identical but structurally different inputs
    (e.g., dicts with different key orders) produce the same representation.
    """
    if isinstance(item, _PRIMITIVE_TYPES):
        return item

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
            sorted_elements = sorted(
                list(item), key=lambda x: (str(type(x)), str(x))
            )
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
            sorted_elements = sorted(
                list(item), key=lambda x: (str(type(x)), str(x))
            )
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
    except Exception:  # If str() fails for some reason
        return repr(item)


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
