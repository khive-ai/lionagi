from collections import deque
from typing import Any, Generator, Mapping
from uuid import UUID

from ._concepts import Collective, Observable, Ordering


def to_uuid(value: Any) -> UUID:
    if isinstance(value, Observable):
        return value.id
    if isinstance(value, UUID):
        return value
    if hasattr(value, "id") and isinstance(value.id, UUID):
        return value.id
    if isinstance(value, str):
        return UUID(value)
    raise ValueError(f"Cannot convert {value} to UUID")


def canonical_id(obj: Any, suppress: bool = False) -> UUID:
    try:
        return to_uuid(obj)
    except Exception as e:
        if suppress:
            return None
        raise ValueError(f"Cannot get canonical ID from {obj}: {e}") from e


def validate_ids(order: Any) -> list[UUID]:
    """Validates and flattens an order specification into a list of UUIDs. raises ValueError if any item is invalid."""
    if isinstance(order, Collective):
        return list(order.keys())
    if isinstance(order, Ordering):
        return order.__list__()
    if isinstance(order, Observable):
        return [order.id]
    if isinstance(order, Mapping):
        order = list(order.keys())

    stack = [order]
    out: list[UUID] = []
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, Observable):
            out.append(cur.id)
        elif isinstance(cur, UUID):
            out.append(cur)
        elif isinstance(cur, str):
            out.append(UUID(cur))
        elif isinstance(cur, (list, tuple, set)):
            stack.extend(reversed(cur))
        else:
            raise ValueError("Invalid item in order.")
    return out


def get_ids(item: Any, as_single: bool = True) -> list[UUID]:
    """get list of UUIDs from item
    if as_single is True, treat collective/ordering as single item
    """
    if as_single and isinstance(item, (Ordering, Collective)):
        return [canonical_id(item)]
    else:
        return validate_ids(item)


def to_list_type(value, as_single: bool = True) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (Ordering, Collective)) and not as_single:
        return list(value)
    if isinstance(value, Observable):
        return [value]
    if hasattr(value, "values") and callable(value.values):
        return list(value.values())
    if isinstance(value, (list, tuple, set, deque, Generator)):
        return list(value)
    return [value]
