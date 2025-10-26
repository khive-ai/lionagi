"""Pydantic adapter for Spec system."""

from typing import TYPE_CHECKING, Any

from lionagi.ln.types import Unset, is_sentinel

from ._protocol import SpecAdapter

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    from lionagi.ln.types import Operable, Spec


_PYDANTIC_FIELD_PARAMS = None


def _get_pydantic_field_params() -> set[str]:
    """Get valid Pydantic Field parameters (cached)."""
    global _PYDANTIC_FIELD_PARAMS
    if _PYDANTIC_FIELD_PARAMS is None:
        import inspect

        from pydantic import Field as PydanticField

        _PYDANTIC_FIELD_PARAMS = set(
            inspect.signature(PydanticField).parameters.keys()
        )
        _PYDANTIC_FIELD_PARAMS.discard("kwargs")
    return _PYDANTIC_FIELD_PARAMS


class PydanticSpecAdapter(SpecAdapter):
    """Pydantic implementation of SpecAdapter."""

    @classmethod
    def create_field(cls, spec: "Spec") -> "FieldInfo":
        """Create a Pydantic FieldInfo object from Spec."""
        from pydantic import Field as PydanticField

        # Get valid Pydantic Field parameters (cached)
        pydantic_field_params = _get_pydantic_field_params()

        # Extract metadata for FieldInfo
        field_kwargs = {}

        if not is_sentinel(spec.metadata, none_as_sentinel=True):
            for meta in spec.metadata:
                if meta.key == "default":
                    # Handle callable defaults as default_factory
                    if callable(meta.value):
                        field_kwargs["default_factory"] = meta.value
                    else:
                        field_kwargs["default"] = meta.value
                elif meta.key == "validator":
                    # Validators are handled separately in create_model
                    continue
                elif meta.key in pydantic_field_params:
                    # Pass through standard Pydantic field attributes
                    field_kwargs[meta.key] = meta.value
                elif meta.key in {"nullable", "listable"}:
                    # These are FieldTemplate markers, don't pass to FieldInfo
                    pass
                else:
                    # Filter out unserializable objects from json_schema_extra
                    if isinstance(meta.value, type):
                        # Skip type objects - can't be serialized
                        continue

                    # Any other metadata goes in json_schema_extra
                    if "json_schema_extra" not in field_kwargs:
                        field_kwargs["json_schema_extra"] = {}
                    field_kwargs["json_schema_extra"][meta.key] = meta.value

        # Handle nullable case - ensure default is set if not already
        if (
            spec.is_nullable
            and "default" not in field_kwargs
            and "default_factory" not in field_kwargs
        ):
            field_kwargs["default"] = None

        field_info = PydanticField(**field_kwargs)
        field_info.annotation = spec.annotation

        return field_info

    @classmethod
    def create_validator(cls, spec: "Spec") -> dict | None:
        """Create Pydantic field_validator from Spec metadata."""
        if (v := spec.get("validator")) is Unset:
            return None

        from pydantic import field_validator

        field_name = spec.name or "field"
        return {f"{field_name}_validator": field_validator(field_name)(v)}

    @classmethod
    def create_model(
        cls,
        op: "Operable",
        model_name: str,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        base_type: type["BaseModel"] | None = None,
        doc: str | None = None,
    ) -> type["BaseModel"]:
        """Generate Pydantic BaseModel from Operable."""
        from lionagi.models.model_params import ModelParams

        use_specs = op.get_specs(include=include, exclude=exclude)
        use_fields = {i.name: cls.create_field(i) for i in use_specs if i.name}

        model_cls = ModelParams(
            name=model_name,
            parameter_fields=use_fields,
            base_type=base_type,
            inherit_base=True,
            doc=doc,
        ).create_new_model()

        model_cls.model_rebuild()
        return model_cls

    @classmethod
    def validate_model(
        cls, model_cls: type["BaseModel"], data: dict
    ) -> "BaseModel":
        """Validate dict data into Pydantic model instance."""
        return model_cls.model_validate(data)

    @classmethod
    def dump_model(cls, instance: "BaseModel") -> dict:
        """Dump Pydantic model instance to dictionary."""
        return instance.model_dump()
