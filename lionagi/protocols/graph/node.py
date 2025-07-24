# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from pydantic import field_validator
from pydapter import Adaptable, AsyncAdaptable
from pydapter.adapters import JsonAdapter, TomlAdapter
from pydapter.extras.pandas_ import SeriesAdapter

from lionagi._class_registry import LION_CLASS_REGISTRY

from .._concepts import Relational
from ..generic.element import Element

NODE_DEFAULT_ADAPTERS = (
    JsonAdapter,
    SeriesAdapter,
    TomlAdapter,
)


__all__ = ("Node",)


class Node(Element, Relational, AsyncAdaptable, Adaptable):
    """
    A base class for all Nodes in a graph, storing:
      - Arbitrary content
      - Metadata as a dict
      - An optional numeric embedding (list of floats)
      - Automatic subclass registration
    """

    content: Any = None
    embedding: list[float] | None = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize and register subclasses in the global class registry."""
        super().__pydantic_init_subclass__(**kwargs)
        LION_CLASS_REGISTRY[cls.class_name(full=True)] = cls

    @field_validator("embedding", mode="before")
    def _parse_embedding(
        cls, value: list[float] | str | None
    ) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                loaded = json.loads(value)
                if not isinstance(loaded, list):
                    raise ValueError
                return [float(x) for x in loaded]
            except Exception as e:
                raise ValueError("Invalid embedding string.") from e
        if isinstance(value, list):
            try:
                return [float(x) for x in value]
            except Exception as e:
                raise ValueError("Invalid embedding list.") from e
        raise ValueError(
            "Invalid embedding type; must be list or JSON-encoded string."
        )

    async def adapt_to_async(
        self, obj_key: str, many=False, **kwargs: Any
    ) -> Any:
        kwargs["adapt_meth"] = "to_dict"
        return await super().adapt_to_async(
            obj_key=obj_key, many=many, **kwargs
        )

    @classmethod
    async def adapt_from_async(
        cls,
        obj: Any,
        obj_key: str,
        many=False,
        **kwargs: Any,
    ) -> Node:
        kwargs["adapt_meth"] = "from_dict"
        return await super().adapt_from_async(
            obj, obj_key=obj_key, many=many, **kwargs
        )

    def adapt_to(self, obj_key: str, many=False, **kwargs: Any) -> Any:
        """
        Convert this Node to another format using a registered adapter.
        """
        kwargs["adapt_meth"] = "to_dict"
        return super().adapt_to(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(
        cls,
        obj: Any,
        obj_key: str,
        many=False,
        **kwargs: Any,
    ) -> Node:
        """
        Construct a Node from an external format using a registered adapter.
        If the adapter returns a dictionary with 'lion_class', we can
        auto-delegate to the correct subclass via from_dict.
        """
        kwargs["adapt_meth"] = "from_dict"
        return super().adapt_from(obj, obj_key=obj_key, many=many, **kwargs)


# File: lionagi/protocols/graph/node.py
