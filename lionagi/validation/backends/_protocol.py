from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from lionagi.ln.types import Operable, Spec


T = TypeVar("T")

ModelInstance = T
ModelClass = type[ModelInstance]


@runtime_checkable
class ValidationBackend(Protocol):

    @classmethod
    def annotate_spec(cls, spec: "Spec") -> Any:
        """Annotate a Spec for the target framework."""

    @classmethod
    def compile_type(
        cls,
        op: "Operable",
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        base_type: type | None = None,
        doc: str | None = None,
        **kw,
    ) -> ModelClass:
        """Compile an Operable into a framework-specific model type."""

    @classmethod
    def validate_instance(
        cls,
        op: "Operable",
        data: dict,
        strict: bool = False,
    ) -> ModelInstance:
        """Validate data against an operable"""

    @classmethod
    def update_instance(
        cls,
        instance: ModelInstance,
        updates: dict,
        model_cls: ModelClass | None = None,
    ) -> ModelInstance:
        """Update an existing model instance with new data."""
