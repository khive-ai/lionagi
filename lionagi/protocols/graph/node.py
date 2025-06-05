# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, ClassVar

from pydantic import field_validator
from pydapter import AdapterRegistry
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


class NodeAdapterRegistry(AdapterRegistry):
    pass


node_adapter_registry = NodeAdapterRegistry()
for i in NODE_DEFAULT_ADAPTERS:
    node_adapter_registry.register(i)

__all__ = ("Node",)


class Node(Element, Relational):
    """
    A base class for all Nodes in a graph, storing:
      - Arbitrary content
      - Metadata as a dict
      - An optional numeric embedding (list of floats)
      - Automatic subclass registration
    """

    _adapter_registry: ClassVar[AdapterRegistry] = node_adapter_registry

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

    def adapt_to(
        self, obj_key: str, /, many: bool = False, **kwargs: Any
    ) -> Any:
        """
        Convert this Node to another format using a registered adapter.
        """
        # For JSON/TOML adapters, we need to pass the dict representation
        if obj_key in ["json", "toml"]:
            data = self.to_dict()

            # Create a simple object that has model_dump method
            class _Wrapper:
                def __init__(self, data):
                    self._data = data

                def model_dump(self):
                    return self._data

            wrapper = _Wrapper(data)
            return self._get_adapter_registry().adapt_to(
                wrapper, obj_key=obj_key, many=many, **kwargs
            )
        return self._get_adapter_registry().adapt_to(
            self, obj_key=obj_key, many=many, **kwargs
        )

    @classmethod
    def adapt_from(
        cls,
        obj: Any,
        obj_key: str,
        /,
        many: bool = False,
        **kwargs: Any,
    ) -> "Node":
        """
        Construct a Node from an external format using a registered adapter.
        If the adapter returns a dictionary with 'lion_class', we can
        auto-delegate to the correct subclass via from_dict.
        """
        result = cls._get_adapter_registry().adapt_from(
            cls, obj, obj_key=obj_key, many=many, **kwargs
        )
        # If adapter returned multiple items, choose the first or handle as needed.
        if isinstance(result, list):
            result = result[0]
        return cls.from_dict(result)

    @classmethod
    def _get_adapter_registry(cls) -> AdapterRegistry:
        if isinstance(cls._adapter_registry, type):
            cls._adapter_registry = cls._adapter_registry()
        return cls._adapter_registry

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:
        cls._get_adapter_registry().register(adapter)


# File: lionagi/protocols/graph/node.py
