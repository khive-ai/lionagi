"""Validation backend abstraction.

Currently Pydantic, future Rust.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, Dict
from pydantic import BaseModel, Field, create_model


class ValidationBackend(ABC):
    """Abstract validation backend.
    
    Current: Pydantic for rich validation
    Future: Rust for proven correctness
    """
    
    @abstractmethod
    def create_model(self, 
                    name: str,
                    fields: Dict[str, Any],
                    **kwargs) -> Type:
        """Create a validated model class"""
        pass
    
    @abstractmethod
    def validate(self, model_class: Type, data: Any) -> Any:
        """Validate data against model"""
        pass
    
    @abstractmethod
    def serialize(self, instance: Any) -> Dict[str, Any]:
        """Serialize validated instance"""
        pass


class PydanticBackend(ValidationBackend):
    """Pydantic validation backend - current implementation."""
    
    def create_model(self, 
                    name: str,
                    fields: Dict[str, Any],
                    **kwargs) -> Type[BaseModel]:
        """Create Pydantic model dynamically"""
        # Convert field definitions to Pydantic format
        pydantic_fields = {}
        for field_name, field_def in fields.items():
            if isinstance(field_def, tuple):
                # (type, default)
                type_hint, default = field_def
                pydantic_fields[field_name] = (type_hint, Field(default=default))
            else:
                # Just type
                pydantic_fields[field_name] = (field_def, Field())
        
        return create_model(name, **pydantic_fields, **kwargs)
    
    def validate(self, model_class: Type[BaseModel], data: Any) -> BaseModel:
        """Validate using Pydantic"""
        return model_class(**data) if isinstance(data, dict) else model_class.model_validate(data)
    
    def serialize(self, instance: BaseModel) -> Dict[str, Any]:
        """Serialize using Pydantic"""
        return instance.model_dump()


# Global backend instance
_backend: ValidationBackend = PydanticBackend()


def get_backend() -> ValidationBackend:
    """Get current validation backend"""
    return _backend


def set_backend(backend: ValidationBackend):
    """Set validation backend (for future Rust integration)"""
    global _backend
    _backend = backend