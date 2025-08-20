import datetime as dt
import decimal
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import orjson


def _get_default_serializers():
    return {
        dt.datetime: lambda o: o.isoformat(),
        Path: lambda o: str(o),
        UUID: lambda o: str(o),
        decimal.Decimal: lambda o: str(o),
        set: lambda o: list(o),
        frozenset: lambda o: list(o),
    }


def _get_default_serializer_order():
    return [dt.datetime, Path, UUID, decimal.Decimal, set, frozenset]


def get_orjson_default(
    order: list[type] = None,
    additional: dict[type, Callable] = None,
    extend_default: bool = True,
) -> Callable:
    """get the default function for orjson.dumps
    Args:
        order: order of types to check. Defaults to None.
        additional: additional serializers
        extend_default: when order is provided, whether to extend the default order or replace it.
    """
    dict_ = _get_default_serializers()
    dict_.update(additional or {})
    order_ = _get_default_serializer_order()

    if order:
        if len(additional or {}) > 0 and extend_default:
            order_.extend([k for k in order if k not in order_])
        else:
            order_ = list(order)
    else:
        if len(additional or {}) > 0:
            order_.extend([k for k in additional.keys() if k not in order_])

    def default(obj):
        for t in order_:
            if isinstance(obj, t) and t in dict_:
                return dict_[t](obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return default


DEFAULT_SERIALIZER = get_orjson_default()
DEFAULT_SERIALIZER_OPTION = (
    orjson.OPT_INDENT_2
    | orjson.OPT_SORT_KEYS
    | orjson.OPT_APPEND_NEWLINE
    | orjson.OPT_SERIALIZE_DATACLASS
)


def json_dumps(d_, decode=True, /) -> str:
    by_ = orjson.dumps(
        d_,
        default=DEFAULT_SERIALIZER,
        option=DEFAULT_SERIALIZER_OPTION,
    )
    if decode:
        return by_.decode("utf-8")
    return by_
