"""Compositional specifications for LionAGI.

Specs are immutable blueprints that can be composed, transformed, and materialized
to different backends. They form the foundation for field definitions, model schemas,
operation specifications, and capability tokens.
"""

from .base import Backend, Spec, SpecCache
from .field_spec import FieldSpec

__all__ = ("Spec", "Backend", "SpecCache", "FieldSpec")
