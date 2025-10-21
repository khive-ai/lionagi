from typing import TYPE_CHECKING, Any

from lionagi.ln.types import Unset, is_sentinel, not_sentinel

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




class PydanticBackend:

    @classmethod
    def annotate_spec(cls, spec: "Spec") -> "FieldInfo":

        from pydantic import Field as PydanticField
        pydantic_field_params = _get_pydantic_field_params()

        field_kwargs = {}
        if not_sentinel(spec.metadatal, empty_as_sentinel=True):
            
            
            
            
            


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


class PydanticSpecAdapter:

    @classmethod
    def create_field(cls, spec: "Spec") -> "FieldInfo":
        """Create a Pydantic FieldInfo object from this template.

        Returns:
            A Pydantic FieldInfo object with all metadata applied
        """
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

    @classmethod
    def create_validator(cls, spec: "Spec"):
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

    # ---- Validation Methods ----

    @classmethod
    def parse_json(cls, text: str, fuzzy: bool = True) -> dict | list | Any:
        """Extract and parse JSON from text.

        Args:
            text: Raw text potentially containing JSON
            fuzzy: Use fuzzy parsing (more lenient)

        Returns:
            Parsed JSON object (dict, list, or primitive)
        """
        from lionagi.ln import extract_json

        data = extract_json(text, fuzzy_parse=fuzzy)

        # Unwrap single-item lists/tuples
        if isinstance(data, (list, tuple)) and len(data) == 1:
            data = data[0]

        return data

    @classmethod
    def fuzzy_match_fields(
        cls, data: dict, model_cls: type["BaseModel"], strict: bool = False
    ) -> dict:
        """Match data keys to model fields with fuzzy matching.

        Args:
            data: Raw data dictionary
            model_cls: Target Pydantic model class
            strict: If True, raise on unmatched fields; if False, force coercion

        Returns:
            Dictionary with keys matched to model fields

        Raises:
            Exception: If strict=True and fields don't match
        """
        from lionagi.ln import fuzzy_match_keys
        from lionagi.ln.types import Undefined

        handle_mode = "raise" if strict else "force"

        matched = fuzzy_match_keys(
            data, model_cls.model_fields, handle_unmatched=handle_mode
        )

        # Filter out undefined values
        return {k: v for k, v in matched.items() if v != Undefined}

    @classmethod
    def validate_response(
        cls,
        text: str,
        model_cls: type["BaseModel"],
        strict: bool = False,
        fuzzy_parse: bool = True,
    ) -> "BaseModel | None":
        """Validate and parse response text into model instance.

        This is the main validation method that combines JSON parsing,
        fuzzy field matching, and Pydantic validation.

        Args:
            text: Raw response text
            model_cls: Target Pydantic model class
            strict: If True, raise on validation errors; if False, return None
            fuzzy_parse: Use fuzzy JSON parsing

        Returns:
            Validated model instance, or None if validation fails (when strict=False)

        Raises:
            Exception: If strict=True and validation fails

        Example:
            >>> from lionagi.ln.types import Spec, Operable
            >>> from lionagi.adapters.spec_adapters.pydantic_field import PydanticSpecAdapter
            >>>
            >>> # Create model from specs
            >>> op = Operable(name="User")
            >>> # ... add specs to op ...
            >>> UserModel = PydanticSpecAdapter.create_model(op, "UserModel")
            >>>
            >>> # Validate response
            >>> response_text = '{"username": "alice", "age": 30}'
            >>> user = PydanticSpecAdapter.validate_response(
            ...     response_text,
            ...     UserModel,
            ...     strict=False
            ... )
        """
        try:
            # Step 1: Parse JSON
            data = cls.parse_json(text, fuzzy=fuzzy_parse)

            # Step 2: Fuzzy match fields
            matched_data = cls.fuzzy_match_fields(
                data, model_cls, strict=strict
            )

            # Step 3: Validate with Pydantic
            instance = model_cls.model_validate(matched_data)

            return instance

        except Exception as e:
            if strict:
                raise
            return None

    @classmethod
    def update_model(
        cls,
        instance: "BaseModel",
        updates: dict,
        model_cls: type["BaseModel"] | None = None,
    ) -> "BaseModel":
        """Update existing model instance with new data.

        Args:
            instance: Existing model instance
            updates: Dictionary of updates
            model_cls: Optional model class (defaults to instance's class)

        Returns:
            New validated model instance with updates applied
        """
        model_cls = model_cls or type(instance)

        # Merge existing data with updates
        current_data = instance.model_dump()
        current_data.update(updates)

        # Validate merged data
        return model_cls.model_validate(current_data)
