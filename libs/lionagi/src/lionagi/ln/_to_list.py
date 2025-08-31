from collections.abc import Iterable, Mapping
from enum import Enum as _Enum
from typing import Any

import msgspec
from pydantic import BaseModel
from pydantic_core import PydanticUndefinedType

from ._hash import hash_dict
from ._types import UndefinedType, UnsetType

__all__ = ("to_list",)


_SKIP_TYPE = (
    str,
    bytes,
    bytearray,
    Mapping,
    BaseModel,
    msgspec.Struct,
    _Enum,
)
_TUPLE_SET_TYPES = (tuple, set, frozenset)
_SKIP_TUPLE_SET = (*_SKIP_TYPE, *_TUPLE_SET_TYPES)
_SINGLETONE_TYPES = (UndefinedType, UnsetType, PydanticUndefinedType)
_BYTE_LIKE_TYPES = (str, bytes, bytearray)


def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    use_values: bool = False,
    flatten_tuple_set: bool = False,
) -> list:
    """Convert input to a list with optional transformations.

    Transforms various input types into a list with configurable processing
    options for flattening, filtering, and value extraction.

    Args:
        input_: Value to convert to list.
        flatten: If True, recursively flatten nested iterables.
        dropna: If True, remove None and undefined values.
        unique: If True, remove duplicates (requires flatten=True).
        use_values: If True, extract values from enums/mappings.
        flatten_tuple_set: If True, include tuple/set items in flattening; if False (default),
            tuples/sets are treated as atomic items even when flatten=True.

    Raises:
        ValueError: If unique=True is used without flatten=True.
    """

    def _process_list(
        lst: list[Any],
        flatten: bool,
        dropna: bool,
    ) -> list[Any]:
        """Process list according to flatten and dropna options.

        Args:
            lst: Input list to process.
            flatten: Whether to flatten nested iterables.
            dropna: Whether to remove None/undefined values.
        """
        result = []
        skip_types = _SKIP_TYPE if flatten_tuple_set else _SKIP_TUPLE_SET

        for item in lst:
            if dropna and (item is None or isinstance(item, _SINGLETONE_TYPES)):
                continue

            is_iterable = isinstance(item, Iterable)
            should_skip = isinstance(item, skip_types)

            if is_iterable and not should_skip:
                item_list = list(item)
                if flatten:
                    result.extend(_process_list(item_list, flatten, dropna))
                else:
                    result.append(_process_list(item_list, flatten, dropna))
            else:
                result.append(item)

        return result

    def _to_list_type(input_: Any, use_values: bool) -> list[Any]:
        """Convert input to initial list based on type.

        Args:
            input_: Value to convert to list.
            use_values: Whether to extract values from containers.
        """
        if input_ is None or isinstance(input_, _SINGLETONE_TYPES):
            return []

        if isinstance(input_, list):
            return input_

        if isinstance(input_, type) and issubclass(input_, _Enum):
            members = input_.__members__.values()
            return [member.value for member in members] if use_values else list(members)

        if isinstance(input_, _BYTE_LIKE_TYPES):
            return list(input_) if use_values else [input_]

        if isinstance(input_, Mapping):
            return list(input_.values()) if use_values and hasattr(input_, "values") else [input_]

        # Handle both msgspec.Struct (V1 standard) and BaseModel (legacy)
        if isinstance(input_, (msgspec.Struct, BaseModel)):
            return [input_]

        if isinstance(input_, Iterable) and not isinstance(input_, _BYTE_LIKE_TYPES):
            return list(input_)

        return [input_]

    if unique and not flatten:
        raise ValueError("unique=True requires flatten=True")

    initial_list = _to_list_type(input_, use_values=use_values)
    processed = _process_list(initial_list, flatten=flatten, dropna=dropna)

    if unique:
        seen = set()
        out = []
        # Try direct hashable approach first
        use_hash_fallback = False
        for i in processed:
            try:
                if not use_hash_fallback:
                    # Direct approach - try to use the item as hash key
                    if i not in seen:
                        seen.add(i)
                        out.append(i)
                else:
                    # Hash-based approach for unhashable items
                    hash_value = (
                        hash(i)
                        if hasattr(i, "__hash__") and i.__hash__ is not None
                        else hash_dict(i)
                    )
                    if hash_value not in seen:
                        seen.add(hash_value)
                        out.append(i)
            except TypeError:
                # Switch to hash-based approach and restart
                if not use_hash_fallback:
                    use_hash_fallback = True
                    seen = set()
                    out = []
                    # Restart from beginning with hash-based approach
                    for j in processed:
                        try:
                            hash_value = hash(j)
                        except TypeError:
                            # Handle msgspec.Struct (V1 standard), BaseModel (legacy), and Mapping
                            if isinstance(j, (msgspec.Struct, BaseModel, Mapping)):
                                hash_value = hash_dict(j)
                            else:
                                raise ValueError(
                                    "Unhashable type encountered in list unique value processing."
                                )
                        if hash_value not in seen:
                            seen.add(hash_value)
                            out.append(j)
                    break
        return out

    return processed
