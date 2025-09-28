"""Morphism system - composable operations with pre/apply/post lifecycle.

Core concepts:
- Morphism: Base operation protocol with invariants
- SystemMorphism: System-level morphisms that run for all operations
- Registry: Registration and discovery of morphisms
- Binding: Context-based parameter binding from Branch
"""

from .base import Morphism, MorphMeta
from .registry import MorphismRegistry, register
from .binding import BoundMorphism

__all__ = [
    "Morphism",
    "MorphMeta",
    "MorphismRegistry",
    "register",
    "BoundMorphism",
]