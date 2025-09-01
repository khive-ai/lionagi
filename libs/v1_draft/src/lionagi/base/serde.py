from __future__ import annotations

from typing import Any

import msgspec


def to_json(obj: Any) -> str:
    return msgspec.json.encode(obj).decode("utf-8")


def from_json(s: str, type_: Any | None = None) -> Any:
    if type_ is None:
        return msgspec.json.decode(s.encode("utf-8"))
    return msgspec.json.decode(s.encode("utf-8"), type=type_)
