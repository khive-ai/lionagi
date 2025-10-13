from typing import Any


class PydanticFieldAdapter:

    @staticmethod
    def create_validator(lspec):
        if "validator" not in lspec.metadata:
            return None

        name = lspec.metadata.get("name") or "unnamed_field"
        from pydantic import field_validator

        validator_name = f"{name}_validator"
        return {
            validator_name: field_validator(name)(lspec.metadata["validator"])
        }

    @staticmethod
    def create_serializer(lspec):
        if "serializer" not in lspec.metadata:
            return None

        name = lspec.metadata.get("name") or "unnamed_field"
        from pydantic import field_serializer

        serializer_name = f"{name}_serializer"
        return {
            serializer_name: field_serializer(name)(
                lspec.metadata["serializer"]
            )
        }

    @staticmethod
    def create_field(lspec) -> Any:
        """Create Pydantic FieldInfo with metadata applied.

        Converts metadata to Pydantic format: valid params passed directly,
        unknown moved to json_schema_extra, callable defaults → default_factory.
        """
        from pydantic import Field as PydanticField

        # Get valid Pydantic Field parameters (cached at class level)
        if not hasattr(type(lspec), "_pydantic_params"):
            import inspect

            params = set(inspect.signature(PydanticField).parameters.keys())
            params.discard("kwargs")
            type(lspec)._pydantic_params = params

        field_kwargs = {}

        for key, value in lspec.metadata.items():
            if key == "default":
                # Handle callable defaults as default_factory
                if callable(value):
                    field_kwargs["default_factory"] = value
                else:
                    field_kwargs["default"] = value

            elif key == "validator":
                # Skip validators (handled by field_validator property)
                continue

            elif key in ("nullable", "listable"):
                # Internal markers - don't pass to Pydantic
                continue

            elif key in type(lspec)._pydantic_params:
                # Valid Pydantic parameter
                field_kwargs[key] = value

            else:
                # Unknown metadata → json_schema_extra
                if "json_schema_extra" not in field_kwargs:
                    field_kwargs["json_schema_extra"] = {}
                # Skip type objects (can't serialize to JSON)
                if not isinstance(value, type):
                    field_kwargs["json_schema_extra"][key] = value

        # Handle nullable case - ensure default is set
        if (
            lspec.metadata.get("nullable")
            and "default" not in field_kwargs
            and "default_factory" not in field_kwargs
        ):
            field_kwargs["default"] = None

        # Apply type transformations in same order as annotated()
        actual_type = lspec.base_type

        # 1. Apply listable first (inner)
        if lspec.metadata.get("listable"):
            actual_type = list[actual_type]  # type: ignore

        # 2. Apply nullable second (outer)
        if lspec.metadata.get("nullable"):
            actual_type = actual_type | None  # type: ignore

        field_info = PydanticField(**field_kwargs)
        field_info.annotation = actual_type
        return field_info
