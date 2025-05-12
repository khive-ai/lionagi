from collections.abc import Callable
from typing import Any, Literal, Sequence


class NestedStructureUtil:
    @staticmethod
    def flatten(
        nested_structure: Any,
        /,
        *,
        parent_key: tuple = (),
        sep: str = "|",
        coerce_keys: bool = True,
        dynamic: bool = True,
        coerce_sequence: Literal["dict", "list"] | None = None,
        max_depth: int | None = None,
    ) -> dict[tuple | str, Any] | None:
        from .flatten import flatten

        return flatten(
            nested_structure,
            parent_key=parent_key,
            sep=sep,
            coerce_keys=coerce_keys,
            dynamic=dynamic,
            coerce_sequence=coerce_sequence,
            max_depth=max_depth,
        )

    @staticmethod
    def filter(
        nested_structure: dict[Any, Any] | list[Any],
        /,
        condition: Callable[[Any], bool],
    ) -> dict[Any, Any] | list[Any]:
        from .nfilter import nfilter

        return nfilter(
            nested_structure,
            condition=condition,
        )

    @staticmethod
    def get(
        nested_structure: dict[Any, Any] | list[Any],
        /,
        indices: list[int | str],
        default: Any = ...,
    ) -> Any:
        from .nget import nget

        if default is ...:
            return nget(nested_structure, indices)
        return nget(nested_structure, indices, default=default)

    @staticmethod
    def insert(
        nested_structure: dict[Any, Any] | list[Any],
        /,
        indices: list[str | int],
        value: Any,
        *,
        current_depth: int = 0,
    ) -> None:
        from .ninsert import ninsert

        ninsert(
            nested_structure,
            indices=indices,
            value=value,
            current_depth=current_depth,
        )

    @staticmethod
    def merge(
        nested_structure: Sequence[dict[str, Any] | list[Any]],
        /,
        *,
        overwrite: bool = False,
        dict_sequence: bool = False,
        sort_list: bool = False,
        custom_sort: Callable[[Any], Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        from .nmerge import nmerge

        return nmerge(
            nested_structure,
            overwrite=overwrite,
            dict_sequence=dict_sequence,
            sort_list=sort_list,
            custom_sort=custom_sort,
        )

    @staticmethod
    def pop(
        input_: dict[str, Any] | list[Any],
        /,
        indices: str | int | Sequence[str | int],
        default: Any = ...,
    ) -> Any:
        from .npop import npop

        if default is ...:
            return npop(input_, indices)
        return npop(input_, indices, default=default)

    @staticmethod
    def set(
        nested_structure: dict[str, Any] | list[Any],
        /,
        indices: str | int | Sequence[str | int],
        value: Any,
    ) -> None:
        from .nset import nset

        nset(nested_structure, indices, value)

    @staticmethod
    def unflatten(
        flat_dict: dict[str, Any], sep: str = "|", inplace: bool = False
    ) -> dict[str, Any] | list[Any]:
        from .unflatten import unflatten

        return unflatten(flat_dict, sep=sep, inplace=inplace)

    @staticmethod
    def is_structure_homogeneous(
        structure: Any, return_structure_type: bool = False
    ) -> bool | tuple[bool, type | None]:
        from .utils import is_structure_homogeneous

        return is_structure_homogeneous(
            structure, return_structure_type=return_structure_type
        )

    @staticmethod
    def is_same_dtype(
        input_: list[Any] | dict[Any, Any],
        dtype: type | None = None,
        return_dtype: bool = False,
    ) -> bool | tuple[bool, type | None]:
        from .utils import is_same_dtype

        return is_same_dtype(input_, dtype=dtype, return_dtype=return_dtype)

    @staticmethod
    def get_target_container(
        nested: list[Any] | dict[Any, Any], indices: list[int | str]
    ) -> list[Any] | dict[Any, Any]:
        from .utils import get_target_container

        return get_target_container(nested, indices)
