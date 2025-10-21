from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from lionagi.ln.types import Operable, Spec, is_sentinel


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


class PydanticBackend:

    @classmethod
    def annotate_spec(cls, spec: "Spec") -> "FieldInfo":
        """Annotate a Spec for the target framework."""

        from pydantic import Field as PydanticField

        pydantic_field_params = _get_pydantic_field_params()
        field_kwargs = spec.metadict(exclude_common=True)

        if spec.has_default:
            if spec.has_async_default_factory:
                raise ValueError(
                    "Pydantic does not support async default factories."
                )

        if not is_sentinel(
            spec.metadata, none_as_sentinel=True, empty_as_sentinel=True
        ):
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
                    # to avoid Pydantic serialization errors when generating JSON schema
                    from pydantic import BaseModel

                    # Skip model classes and other unserializable types
                    if isinstance(meta.value, type):
                        # Skip type objects (including model classes) - they can't be serialized
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
