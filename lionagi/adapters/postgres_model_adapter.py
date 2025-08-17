"""
Clean LionAGI PostgreSQL adapter for integration into lionagi core.

This adapter handles the metadata field conflict and provides seamless
PostgreSQL persistence for lionagi Nodes.
"""

from __future__ import annotations

from typing import Union, get_args, get_origin

from pydantic import BaseModel

from ._utils import check_postgres_available

_POSTGRES_AVAILABLE = check_postgres_available()
if isinstance(_POSTGRES_AVAILABLE, ImportError):
    raise _POSTGRES_AVAILABLE

from pydapter.model_adapters.postgres_model import PostgresModelAdapter
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase


class LionAGIPostgresAdapter(PostgresModelAdapter):
    """
    PostgreSQL adapter for lionagi Nodes with automatic metadata field mapping.

    Solves the core issue where lionagi's 'metadata' field conflicts with
    SQLAlchemy's reserved 'metadata' attribute by automatically mapping it
    to 'node_metadata' in the database schema.

    Features:
    - Automatic metadata field mapping (metadata → node_metadata)
    - Handles Union types like list[float] | None for embedding fields
    - Preserves all PostgreSQL-specific type support from parent
    - Transparent to lionagi users - Elements work seamlessly
    """

    # Core field mapping to resolve SQLAlchemy conflicts
    FIELD_MAPPINGS = {"metadata": "node_metadata"}

    def __init__(self):
        super().__init__()
        self._register_lionagi_types()

    def _register_lionagi_types(self):
        """Register lionagi-specific type mappings."""
        try:
            # Handle lionagi IDType as String (UUID)
            from lionagi.protocols.generic.element import IDType

            self.register_type_mapping(
                python_type=IDType,
                sql_type_factory=lambda: String(36),  # UUID string length
                python_to_sql=lambda x: str(x),
                sql_to_python=lambda x: IDType.validate(x) if x else None,
            )
        except ImportError:
            pass  # lionagi not available

    @classmethod
    def pydantic_model_to_sql(
        cls,
        model: type[BaseModel],
        *,
        table_name: str | None = None,
        pk_field: str = "id",
        schema: str | None = None,
    ) -> type[DeclarativeBase]:
        """
        Generate SQLAlchemy model with lionagi field mapping.

        Automatically handles:
        - metadata → node_metadata mapping
        - Union type resolution (e.g., list[float] | None → list[float])
        - Standard lionagi Node field structure
        """

        # Create modified field mapping for lionagi compatibility
        modified_fields = {}

        for name, info in model.model_fields.items():
            # Apply field name mapping
            field_name = cls.FIELD_MAPPINGS.get(name, name)

            # Resolve Union types by extracting non-None type
            annotation = info.annotation
            origin = get_origin(annotation)

            if origin is Union or (
                hasattr(annotation, "__class__")
                and annotation.__class__.__name__ == "UnionType"
            ):
                args = get_args(annotation)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            # Create field info with resolved annotation
            from pydantic.fields import FieldInfo

            modified_fields[field_name] = FieldInfo(
                annotation=annotation,
                default=info.default,
                default_factory=info.default_factory,
                alias=info.alias,
                title=info.title,
                description=info.description,
                json_schema_extra=info.json_schema_extra,
                frozen=info.frozen,
                validate_default=info.validate_default,
                repr=info.repr,
                init_var=info.init_var,
                kw_only=info.kw_only,
            )

        # Create temporary model with mapped fields
        class ModifiedModel(BaseModel):
            model_config = getattr(model, "model_config", {})

        ModifiedModel.model_fields = modified_fields
        ModifiedModel.__name__ = model.__name__

        # Generate SQLAlchemy model with parent's logic
        return super().pydantic_model_to_sql(
            ModifiedModel,
            table_name=table_name,
            pk_field=pk_field,
            schema=schema,
        )
