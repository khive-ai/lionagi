"""Base model classes with pydapter integration.

This module contains base model classes that integrate with pydapter for
serialization and deserialization.
"""

from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel as PydanticBaseModel
from pydapter.core import Adaptable

T = TypeVar("T", bound="BaseModel")


class BaseModel(PydanticBaseModel, Adaptable):
    """Base model class with pydapter integration.

    This class extends Pydantic's BaseModel and adds pydapter's Adaptable mixin
    to provide serialization and deserialization capabilities.
    """

    # Class variables for pydapter adapter registration
    _registered_adapters: ClassVar[Dict[str, Any]] = {}
    _registered_async_adapters: ClassVar[Dict[str, Any]] = {}

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
        extra = "ignore"
        validate_assignment = True

    @classmethod
    def register_adapters(cls) -> None:
        """Register default adapters for this model.

        This method should be called during module initialization to register
        the default adapters for the model.
        """
        from pydapter.adapters.json_ import JsonAdapter
        from pydapter.adapters.toml_ import TomlAdapter
        # from pydapter.adapters.yaml_ import YamlAdapter # if needed

        cls.register_adapter(JsonAdapter)
        cls.register_adapter(TomlAdapter)
        # cls.register_adapter(YamlAdapter) # if needed

        # Register async adapters if applicable
        # from pydapter.extras.async_json import AsyncJsonAdapter # Example
        # cls.register_async_adapter(AsyncJsonAdapter)