from typing import Any, Generator
from ...sys_util import create_copy
from ..type_conversion import to_dict, to_list


def flatten(
    nested_structure: Any,
    /,
    *,
    parent_key: str = "",
    sep: str = "[^_^]",
    max_depth: int | None = None,
    inplace: bool = False,
    dict_only: bool = False,
) -> dict | None:
    """
    Flatten a nested dictionary or list into a single-level dictionary.

    Args:
        nested_structure (Any): The nested structure to flatten.
        parent_key (str, optional): The base key to use for flattened keys.
            Defaults to "".
        sep (str, optional): The separator to use for joining keys. Defaults
            to "[^_^]".
        max_depth (int | None, optional): The maximum depth to flatten.
            Defaults to None.
        inplace (bool, optional): If True, modifies the original structure.
            Defaults to False.
        dict_only (bool, optional): If True, only flattens dictionaries.
            Defaults to False.

    Returns:
        dict | None: The flattened dictionary if inplace is False, otherwise
            None.

    Raises:
        ValueError: If inplace is True and nested_structure is not a
            dictionary.
    """
    if inplace:
        if not isinstance(nested_structure, dict):
            raise ValueError("Object must be a dictionary when 'inplace' is True.")
        _dynamic_flatten_in_place(
            nested_structure,
            parent_key=parent_key,
            sep=sep,
            max_depth=max_depth,
            dict_only=dict_only,
        )
    else:
        parent_key_tuple = tuple(parent_key.split(sep)) if parent_key else ()
        return to_dict(
            _dynamic_flatten_generator(
                nested_structure,
                parent_key=parent_key_tuple,
                sep=sep,
                max_depth=max_depth,
                dict_only=dict_only,
            )
        )


def get_flattened_keys(
    nested_structure: Any,
    /,
    *,
    sep: str = "[^_^]",
    max_depth: int | None = None,
    dict_only: bool = False,
    inplace: bool = False,
) -> list[str]:
    """
    Get all keys from a flattened representation of a nested structure.

    Args:
        nested_structure (Any): The nested structure to process.
        sep (str, optional): The separator to use for joining keys. Defaults
            to "[^_^]".
        max_depth (int | None, optional): The maximum depth to flatten.
            Defaults to None.
        dict_only (bool, optional): If True, only flattens dictionaries.
            Defaults to False.
        inplace (bool, optional): If True, modifies the original structure.
            Defaults to False.

    Returns:
        list[str]: A list of flattened keys.
    """
    if not inplace:
        return to_list(
            flatten(
                nested_structure, sep=sep, max_depth=max_depth, dict_only=dict_only
            ).keys()
        )
    obj_copy = create_copy(nested_structure, num=1)
    flatten(obj_copy, sep=sep, max_depth=max_depth, inplace=True, dict_only=dict_only)
    return to_list(obj_copy.keys())


def _dynamic_flatten_in_place(
    nested_structure: Any,
    /,
    *,
    parent_key: str = "",
    sep: str = "[^_^]",
    max_depth: int | None = None,
    current_depth: int = 0,
    dict_only: bool = False,
) -> None:
    """
    Recursively flatten a nested structure in place.

    Args:
        nested_structure (Any): The nested structure to flatten.
        parent_key (str, optional): The base key to use for flattened keys.
            Defaults to "".
        sep (str, optional): The separator to use for joining keys. Defaults
            to "[^_^]".
        max_depth (int | None, optional): The maximum depth to flatten.
            Defaults to None.
        current_depth (int, optional): The current depth of recursion.
            Defaults to 0.
        dict_only (bool, optional): If True, only flattens dictionaries.
            Defaults to False.

    Raises:
        ValueError: If nested_structure is not a dictionary.
    """
    if isinstance(nested_structure, dict):
        keys_to_delete = []
        items = to_list(nested_structure.items())  # Create a copy of the dictionary

        for k, v in items:
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict) and (max_depth is None or current_depth < max_depth):
                _dynamic_flatten_in_place(
                    v,
                    parent_key=new_key,
                    sep=sep,
                    max_depth=max_depth,
                    current_depth=(current_depth + 1),
                    dict_only=dict_only,
                )
                keys_to_delete.append(k)
                nested_structure.update(v)
            elif not dict_only and (
                isinstance(v, list) or not isinstance(v, (dict, list))
            ):
                nested_structure[new_key] = v
                if parent_key:
                    keys_to_delete.append(k)

        for k in keys_to_delete:
            del nested_structure[k]


def _dynamic_flatten_generator(
    nested_structure: Any,
    parent_key: tuple[str, ...],
    sep: str = "[^_^]",
    max_depth: int | None = None,
    current_depth: int = 0,
    dict_only: bool = False,
) -> Generator[tuple[str, Any], None, None]:
    """
    Generator to recursively flatten a nested structure.

    Args:
        nested_structure (Any): The nested structure to flatten.
        parent_key (tuple[str, ...]): The base key to use for flattened keys.
        sep (str, optional): The separator to use for joining keys. Defaults
            to "[^_^]".
        max_depth (int | None, optional): The maximum depth to flatten.
            Defaults to None.
        current_depth (int, optional): The current depth of recursion.
            Defaults to 0.
        dict_only (bool, optional): If True, only flattens dictionaries.
            Defaults to False.

    Yields:
        tuple[str, Any]: The flattened key and value pairs.
    """
    if max_depth is not None and current_depth > max_depth:
        yield sep.join(parent_key), nested_structure
        return

    if isinstance(nested_structure, dict):
        for k, v in nested_structure.items():
            new_key = parent_key + (k,)
            yield from _dynamic_flatten_generator(
                v, new_key, sep, max_depth, current_depth + 1, dict_only
            )
    elif isinstance(nested_structure, list) and not dict_only:
        for i, item in enumerate(nested_structure):
            new_key = parent_key + (str(i),)
            yield from _dynamic_flatten_generator(
                item, new_key, sep, max_depth, current_depth + 1, dict_only
            )
    else:
        yield sep.join(parent_key), nested_structure
