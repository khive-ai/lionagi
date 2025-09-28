"""Morphism registry - registration and discovery of morphisms.

Adapted from v1 registry pattern.
"""

from typing import Dict, Type, Optional, List, Set
from .base import Morphism, SystemMorphism


# Global registry for morphisms
_MORPHISM_REGISTRY: Dict[str, Type[Morphism]] = {}
_SYSTEM_MORPHISMS: List[Type[SystemMorphism]] = []


def register(cls: Type[Morphism]) -> Type[Morphism]:
    """Class decorator to register a morphism.

    Usage:
        @register
        class MyMorphism(Morphism):
            meta = {"name": "my_morphism"}
    """
    # Get name from meta or class name
    if hasattr(cls, 'meta') and isinstance(cls.meta, dict):
        name = cls.meta.get("name", cls.__name__)
    else:
        name = cls.__name__

    if name in _MORPHISM_REGISTRY:
        raise ValueError(f"Morphism {name} already registered")

    _MORPHISM_REGISTRY[name] = cls

    # Track system morphisms separately
    if issubclass(cls, SystemMorphism):
        _SYSTEM_MORPHISMS.append(cls)
        # Sort by priority
        _SYSTEM_MORPHISMS.sort(key=lambda m: m.meta.get("priority", 100))

    return cls


def get_morphism(name: str) -> Optional[Type[Morphism]]:
    """Get a registered morphism by name."""
    return _MORPHISM_REGISTRY.get(name)


def all_morphisms() -> List[str]:
    """Get names of all registered morphisms."""
    return sorted(_MORPHISM_REGISTRY.keys())


def get_system_morphisms() -> List[Type[SystemMorphism]]:
    """Get all system morphisms ordered by priority."""
    return _SYSTEM_MORPHISMS.copy()


def clear_registry():
    """Clear all registrations (mainly for testing)."""
    _MORPHISM_REGISTRY.clear()
    _SYSTEM_MORPHISMS.clear()


class MorphismRegistry:
    """Registry instance for scoped morphism management.

    This allows different parts of the system to have
    their own morphism registries if needed.
    """

    def __init__(self):
        self.morphisms: Dict[str, Type[Morphism]] = {}
        self.system_morphisms: List[Type[SystemMorphism]] = []

    def register(self, morphism: Type[Morphism], name: Optional[str] = None):
        """Register a morphism in this registry."""
        name = name or morphism.meta.get("name", morphism.__name__)

        if name in self.morphisms:
            raise ValueError(f"Morphism {name} already registered")

        self.morphisms[name] = morphism

        if issubclass(morphism, SystemMorphism):
            self.system_morphisms.append(morphism)
            self.system_morphisms.sort(key=lambda m: m.meta.get("priority", 100))

    def get(self, name: str) -> Optional[Type[Morphism]]:
        """Get a morphism from this registry."""
        return self.morphisms.get(name)

    def get_system_morphisms_for(
        self,
        branch,
        target: Morphism
    ) -> List[SystemMorphism]:
        """Get system morphisms that should run for a target morphism.

        This checks should_run() for each system morphism.
        """
        applicable = []
        for morphism_cls in self.system_morphisms:
            morphism = morphism_cls()  # Need instance to check should_run
            if morphism.should_run(branch, target):
                applicable.append(morphism)
        return applicable