# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""SQL DDL SpecAdapter: Spec -> SQL column definitions, Operable -> CREATE TABLE.

Generates SQL DDL statements from Spec/Operable definitions:
    - FK[Model]: Foreign key references (UUID with REFERENCES constraint)
    - Vector[dim]: pgvector VECTOR(dim) for embeddings
    - Type mapping: Python types -> SQL types (TEXT, INTEGER, JSONB, etc.)

Schema Specifications (frozen dataclasses for diffing/introspection):
    - ColumnSpec, IndexSpec, TriggerSpec, CheckConstraintSpec, UniqueConstraintSpec
    - ForeignKeySpec: Full FK constraint with deferrable support
    - TableSpec: Complete table schema representation
    - SchemaSpec: Multi-table database schema

Enums for type-safe specification:
    - OnAction: FK ON DELETE/UPDATE actions (CASCADE, SET NULL, etc.)
    - IndexMethod: Index access methods (BTREE, GIN, HNSW, etc.)
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from datetime import date, datetime
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Any, ForwardRef, Union, get_args, get_origin
from uuid import UUID

from lionagi.ln.types._compat import StrEnum
from lionagi.ln.types._sentinel import Unset, UnsetType, is_sentinel
from lionagi.libs.db.types import (
    FK,
    FKMeta,
    Vector,
    VectorMeta,
    extract_db_meta as extract_kron_db_meta,
)

if TYPE_CHECKING:
    from typing import Protocol

    from lionagi.ln.types import Spec

    class Operable(Protocol):
        def get_specs(self, **kwargs: Any) -> tuple[Spec, ...]: ...


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Basic SQL identifier validation."""
    if not name.isidentifier():
        raise ValueError(f"Invalid SQL {label}: {name!r}")
    return name


def _resolve_forward_ref(fwd: ForwardRef) -> dict[str, Any]:
    from lionagi.libs.db.types import parse_forward_ref

    fk, vec, nullable = parse_forward_ref(fwd)
    if fk is not None:
        base_type = Annotated[UUID, fk]
        return {"base_type": base_type, "nullable": nullable, "listable": False}
    if vec is not None:
        base_type = Annotated[list[float], vec]
        return {"base_type": base_type, "nullable": nullable, "listable": False}
    return {"base_type": str, "nullable": nullable, "listable": False}


def resolve_annotation_to_base_types(annotation: Any) -> dict[str, Any]:
    if isinstance(annotation, ForwardRef):
        return _resolve_forward_ref(annotation)

    def resolve_nullable_inner_type(_anno: Any) -> tuple[bool, Any]:
        origin = get_origin(_anno)

        if origin is type(None):
            return True, type(None)

        if origin in (type(int | str), types.UnionType) or origin is Union:
            args = get_args(_anno)
            non_none_args = [a for a in args if a is not type(None)]
            if len(args) != len(non_none_args):
                if len(non_none_args) == 1:
                    return True, non_none_args[0]
                if non_none_args:
                    return True, reduce(lambda a, b: a | b, non_none_args)
            return False, _anno

        return False, _anno

    def resolve_listable_element_type(_anno: Any) -> Any:
        origin = get_origin(_anno)

        if origin is list:
            args = get_args(_anno)
            if args:
                return True, args[0]
            return True, Any

        return False, _anno

    _null, _inner = resolve_nullable_inner_type(annotation)
    _list, _elem = resolve_listable_element_type(_inner)

    return {
        "base_type": _elem,
        "nullable": _null,
        "listable": _list,
    }

__all__ = (
    "FK",
    "CheckConstraintSpec",
    "ColumnSpec",
    "FKMeta",
    "ForeignKeySpec",
    "IndexMethod",
    "IndexSpec",
    "OnAction",
    "SQLSpecAdapter",
    "SchemaSpec",
    "TableSpec",
    "TriggerSpec",
    "UniqueConstraintSpec",
    "Vector",
    "VectorMeta",
    "extract_kron_db_meta",
)


# =============================================================================
# Enums for Type-Safe Specification
# =============================================================================


class OnAction(StrEnum):
    """FK ON DELETE/ON UPDATE actions."""

    CASCADE = "CASCADE"
    SET_NULL = "SET NULL"
    SET_DEFAULT = "SET DEFAULT"
    RESTRICT = "RESTRICT"
    NO_ACTION = "NO ACTION"


class IndexMethod(StrEnum):
    """Index access methods."""

    BTREE = "btree"
    HASH = "hash"
    GIST = "gist"
    GIN = "gin"
    SPGIST = "spgist"
    BRIN = "brin"
    IVFFLAT = "ivfflat"  # pgvector
    HNSW = "hnsw"  # pgvector


# =============================================================================
# Type Mapping
# =============================================================================

PYTHON_TO_SQL: dict[type, str] = {
    str: "TEXT",
    int: "INTEGER",
    float: "DOUBLE PRECISION",
    bool: "BOOLEAN",
    UUID: "UUID",
    datetime: "TIMESTAMP WITH TIME ZONE",
    date: "DATE",
    bytes: "BYTEA",
    dict: "JSONB",
    list: "JSONB",
}


def python_type_to_sql(
    annotation: Any,
) -> tuple[str, bool, FKMeta | None, VectorMeta | None]:
    """Convert Python type to (sql_type, nullable, fk_meta, vector_meta)."""
    fk_raw, vec_raw = extract_kron_db_meta(annotation, metas="BOTH")
    fk = fk_raw if isinstance(fk_raw, FKMeta) else None
    vec = vec_raw if isinstance(vec_raw, VectorMeta) else None

    resolved = resolve_annotation_to_base_types(annotation)
    nullable = resolved["nullable"]
    annotation = resolved["base_type"]

    if fk is not None:
        return "UUID", nullable, fk, None

    if vec is not None:
        return f"VECTOR({vec.dim})", nullable, None, vec

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if args:
            annotation = args[0]
            for arg in args[1:]:
                if isinstance(arg, FKMeta):
                    return "UUID", nullable, arg, None
                if isinstance(arg, VectorMeta):
                    return f"VECTOR({arg.dim})", nullable, None, arg

    if annotation in PYTHON_TO_SQL:
        return PYTHON_TO_SQL[annotation], nullable, None, None

    if get_origin(annotation) in (dict, list):
        return "JSONB", nullable, None, None

    if hasattr(annotation, "__members__"):
        return "TEXT", nullable, None, None

    try:
        from pydantic import BaseModel

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return "JSONB", nullable, None, None
    except ImportError:
        pass

    return "TEXT", nullable, None, None


# =============================================================================
# Schema Specification Dataclasses (Frozen for Hashability/Diffing)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Specification for a database column.

    Frozen dataclass for hashability and immutability, enabling schema diffing.

    Attributes:
        name: Column name.
        type: SQL type (e.g., "TEXT", "UUID", "VECTOR(1536)").
        nullable: Whether column allows NULL values.
        default: DB-level default expression (e.g., "gen_random_uuid()").
        is_primary_key: Whether this is the primary key.
        is_unique: Whether column has unique constraint.
    """

    name: str
    type: str
    nullable: bool = True
    default: str | None = None
    is_primary_key: bool = False
    is_unique: bool = False

    def to_ddl(self) -> str:
        """Generate column DDL fragment."""
        validate_identifier(self.name, "column name")
        parts = [f'"{self.name}"', self.type]

        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        elif not self.nullable:
            parts.append("NOT NULL")

        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")

        if self.is_unique and not self.is_primary_key:
            parts.append("UNIQUE")

        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ForeignKeySpec:
    """Specification for a foreign key constraint.

    Supports deferrable constraints for complex transactions.

    Attributes:
        name: Constraint name.
        columns: Local column(s) forming the FK.
        ref_table: Referenced table name.
        ref_columns: Referenced column(s).
        on_delete: ON DELETE action.
        on_update: ON UPDATE action.
        deferrable: Whether constraint is deferrable.
        initially_deferred: Whether constraint is initially deferred.
    """

    name: str
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...] = ("id",)
    on_delete: OnAction = OnAction.CASCADE
    on_update: OnAction = OnAction.CASCADE
    deferrable: bool = False
    initially_deferred: bool = False

    def to_ddl(self, table_name: str) -> str:
        """Generate ALTER TABLE ADD CONSTRAINT DDL."""
        validate_identifier(table_name, "table name")
        validate_identifier(self.name, "constraint name")
        validate_identifier(self.ref_table, "referenced table name")
        for col in self.columns:
            validate_identifier(col, "column name")
        for col in self.ref_columns:
            validate_identifier(col, "referenced column name")

        cols = ", ".join(f'"{c}"' for c in self.columns)
        refs = ", ".join(f'"{c}"' for c in self.ref_columns)

        ddl = (
            f'ALTER TABLE "{table_name}" ADD CONSTRAINT "{self.name}" '
            f'FOREIGN KEY ({cols}) REFERENCES "{self.ref_table}" ({refs}) '
            f"ON DELETE {self.on_delete.value} ON UPDATE {self.on_update.value}"
        )

        if self.deferrable:
            ddl += " DEFERRABLE"
            if self.initially_deferred:
                ddl += " INITIALLY DEFERRED"

        return ddl


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """Specification for a database index.

    Supports partial indexes, covering indexes, and vector index methods.

    Attributes:
        name: Index name.
        columns: Column(s) in the index.
        unique: Whether index enforces uniqueness.
        method: Index access method (btree, gin, hnsw, etc.).
        where: Partial index condition.
        concurrently: Whether to create index concurrently.
        include: Columns to include in covering index.
    """

    name: str
    columns: tuple[str, ...]
    unique: bool = False
    method: IndexMethod = IndexMethod.BTREE
    where: str | None = None
    concurrently: bool = False
    include: tuple[str, ...] = ()

    def to_ddl(self, table_name: str, schema: str = "public") -> str:
        """Generate CREATE INDEX DDL."""
        validate_identifier(table_name, "table name")
        validate_identifier(schema, "schema name")
        validate_identifier(self.name, "index name")
        for col in self.columns:
            validate_identifier(col, "column name")
        for col in self.include:
            validate_identifier(col, "included column name")

        parts = ["CREATE"]

        if self.unique:
            parts.append("UNIQUE")

        parts.append("INDEX")

        if self.concurrently:
            parts.append("CONCURRENTLY")

        parts.append(f'IF NOT EXISTS "{self.name}"')
        parts.append(f'ON "{schema}"."{table_name}"')

        if self.method != IndexMethod.BTREE:
            parts.append(f"USING {self.method.value}")

        cols = ", ".join(f'"{c}"' for c in self.columns)
        parts.append(f"({cols})")

        if self.include:
            include_cols = ", ".join(f'"{c}"' for c in self.include)
            parts.append(f"INCLUDE ({include_cols})")

        if self.where:
            parts.append(f"WHERE {self.where}")

        return " ".join(parts) + ";"


@dataclass(frozen=True, slots=True)
class TriggerSpec:
    """Specification for a database trigger.

    Enables database-level business rules (immutability, audit, computed columns).

    Attributes:
        name: Trigger name.
        timing: BEFORE, AFTER, or INSTEAD OF.
        events: Events that fire the trigger (INSERT, UPDATE, DELETE).
        function: Function to call (with schema, e.g., "public.audit_log").
        for_each: ROW or STATEMENT.
        when: Optional WHEN condition.
    """

    name: str
    timing: str  # BEFORE, AFTER, INSTEAD OF
    events: tuple[str, ...]  # INSERT, UPDATE, DELETE
    function: str  # Function name with schema
    for_each: str = "ROW"
    when: str | None = None

    def to_ddl(self, table_name: str, schema: str = "public") -> str:
        """Generate CREATE TRIGGER DDL."""
        validate_identifier(table_name, "table name")
        validate_identifier(schema, "schema name")
        validate_identifier(self.name, "trigger name")

        events_str = " OR ".join(self.events)

        ddl = (
            f'CREATE TRIGGER "{self.name}" '
            f"{self.timing} {events_str} "
            f'ON "{schema}"."{table_name}" '
            f"FOR EACH {self.for_each} "
        )

        if self.when:
            ddl += f"WHEN ({self.when}) "

        ddl += f"EXECUTE FUNCTION {self.function}();"

        return ddl


@dataclass(frozen=True, slots=True)
class CheckConstraintSpec:
    """Specification for a CHECK constraint.

    Enables database-level validation complementing application rules.

    Attributes:
        name: Constraint name.
        expression: CHECK expression (SQL boolean expression).

    Warning:
        The `expression` field accepts raw SQL and is NOT validated.
        Only use expressions from trusted sources. Never pass user input
        directly as this creates SQL injection vulnerabilities.
    """

    name: str
    expression: str

    def to_ddl(self, table_name: str, schema: str = "public") -> str:
        """Generate ALTER TABLE ADD CONSTRAINT DDL."""
        validate_identifier(table_name, "table name")
        validate_identifier(schema, "schema name")
        validate_identifier(self.name, "constraint name")
        return (
            f'ALTER TABLE "{schema}"."{table_name}" '
            f'ADD CONSTRAINT "{self.name}" CHECK ({self.expression});'
        )


@dataclass(frozen=True, slots=True)
class UniqueConstraintSpec:
    """Specification for a UNIQUE constraint.

    Attributes:
        name: Constraint name.
        columns: Column(s) in the constraint.
    """

    name: str
    columns: tuple[str, ...]

    def to_ddl(self, table_name: str, schema: str = "public") -> str:
        """Generate ALTER TABLE ADD CONSTRAINT DDL."""
        validate_identifier(table_name, "table name")
        validate_identifier(schema, "schema name")
        validate_identifier(self.name, "constraint name")
        for col in self.columns:
            validate_identifier(col, "column name")
        cols = ", ".join(f'"{c}"' for c in self.columns)
        return f'ALTER TABLE "{schema}"."{table_name}" ADD CONSTRAINT "{self.name}" UNIQUE ({cols});'


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Complete specification for a database table.

    Hashable, diffable representation enabling state-based migrations.

    Attributes:
        name: Table name.
        schema: Schema name (default: public).
        columns: Tuple of column specifications.
        primary_key: Primary key column(s).
        foreign_keys: Tuple of FK specifications.
        indexes: Tuple of index specifications.
        triggers: Tuple of trigger specifications.
        check_constraints: Tuple of CHECK constraint specifications.
        unique_constraints: Tuple of UNIQUE constraint specifications.
    """

    name: str
    schema: str = "public"
    columns: tuple[ColumnSpec, ...] = ()
    primary_key: tuple[str, ...] = ("id",)
    foreign_keys: tuple[ForeignKeySpec, ...] = ()
    indexes: tuple[IndexSpec, ...] = ()
    triggers: tuple[TriggerSpec, ...] = ()
    check_constraints: tuple[CheckConstraintSpec, ...] = ()
    unique_constraints: tuple[UniqueConstraintSpec, ...] = ()

    @property
    def qualified_name(self) -> str:
        """Get fully qualified table name."""
        validate_identifier(self.schema, "schema name")
        validate_identifier(self.name, "table name")
        return f'"{self.schema}"."{self.name}"'

    def get_column(self, name: str) -> ColumnSpec | None:
        """Get column spec by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_create_table_ddl(self, if_not_exists: bool = True) -> str:
        """Generate CREATE TABLE DDL (without FKs - added separately)."""
        col_defs = [col.to_ddl() for col in self.columns]
        col_separator = ",\n    "
        col_lines = col_separator.join(col_defs)

        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        return (
            f"CREATE TABLE {exists_clause}{self.qualified_name} (\n    {col_lines}\n);"
        )

    def to_full_ddl(self) -> list[str]:
        """Generate all DDL statements for this table.

        Returns statements in execution order:
        1. CREATE TABLE
        2. UNIQUE constraints
        3. CHECK constraints
        4. Foreign keys
        5. Indexes
        6. Triggers
        """
        statements = [self.to_create_table_ddl()]

        for uc in self.unique_constraints:
            statements.append(uc.to_ddl(self.name, self.schema))

        for cc in self.check_constraints:
            statements.append(cc.to_ddl(self.name, self.schema))

        for fk in self.foreign_keys:
            statements.append(fk.to_ddl(self.name) + ";")

        for idx in self.indexes:
            statements.append(idx.to_ddl(self.name, self.schema))

        for trigger in self.triggers:
            statements.append(trigger.to_ddl(self.name, self.schema))

        return statements

    @classmethod
    def from_operable(
        cls,
        op: Operable,
        name: str,
        *,
        schema: str = "public",
        primary_key: str = "id",
        indexes: list[dict[str, Any]] | None = None,
        triggers: list[dict[str, Any]] | None = None,
        check_constraints: list[dict[str, Any]] | None = None,
        unique_constraints: list[dict[str, Any]] | None = None,
    ) -> TableSpec:
        """Create TableSpec from an Operable.

        Args:
            op: Operable with Specs defining the table structure.
            name: Table name.
            schema: Database schema (default "public").
            primary_key: Primary key column name (default "id").
            indexes: List of index definitions.
            triggers: List of trigger definitions.
            check_constraints: List of CHECK constraint definitions.
            unique_constraints: List of UNIQUE constraint definitions.

        Returns:
            TableSpec with columns and constraints derived from Operable.
        """
        columns: list[ColumnSpec] = []
        foreign_keys: list[ForeignKeySpec] = []

        for spec in op.get_specs():
            if not spec.name:
                continue

            sql_type, type_nullable, fk, _ = python_type_to_sql(spec.annotation)
            nullable = type_nullable or spec.is_nullable

            # Check for default value
            default_value = None
            if not is_sentinel(spec.metadata):
                for meta in spec.metadata:
                    if meta.key == "default":
                        val = meta.value
                        if isinstance(val, str):
                            default_value = f"'{val}'"
                        elif isinstance(val, bool):
                            default_value = str(val).upper()
                        elif isinstance(val, (int, float)):
                            default_value = str(val)
                        break

            is_pk = spec.name == primary_key
            col_spec = ColumnSpec(
                name=spec.name,
                type=sql_type,
                nullable=nullable and not is_pk,
                default=default_value,
                is_primary_key=is_pk,
            )
            columns.append(col_spec)

            # Create FK constraint
            if fk is not None:
                fk_spec = ForeignKeySpec(
                    name=f"fk_{name}_{spec.name}",
                    columns=(spec.name,),
                    ref_table=fk.table_name,
                    ref_columns=(fk.column,),
                    on_delete=OnAction(fk.on_delete),
                    on_update=OnAction(fk.on_update),
                    deferrable=fk.deferrable,
                    initially_deferred=fk.initially_deferred,
                )
                foreign_keys.append(fk_spec)

        # Build index specs
        index_specs: list[IndexSpec] = []
        for idx_def in indexes or []:
            idx_cols = tuple(idx_def.get("columns", []))
            idx_name = idx_def.get("name") or f"idx_{name}_{'_'.join(idx_cols)}"
            idx_spec = IndexSpec(
                name=idx_name,
                columns=idx_cols,
                unique=idx_def.get("unique", False),
                method=IndexMethod(idx_def.get("method", "btree")),
                where=idx_def.get("where"),
                include=tuple(idx_def.get("include", [])),
            )
            index_specs.append(idx_spec)

        # Build trigger specs
        trigger_specs: list[TriggerSpec] = []
        for trg_def in triggers or []:
            trg_spec = TriggerSpec(
                name=trg_def["name"],
                timing=trg_def["timing"],
                events=tuple(trg_def["events"]),
                function=trg_def["function"],
                for_each=trg_def.get("for_each", "ROW"),
                when=trg_def.get("when"),
            )
            trigger_specs.append(trg_spec)

        # Build CHECK constraint specs
        check_specs: list[CheckConstraintSpec] = []
        for chk_def in check_constraints or []:
            chk_spec = CheckConstraintSpec(
                name=chk_def["name"],
                expression=chk_def["expression"],
            )
            check_specs.append(chk_spec)

        # Build UNIQUE constraint specs
        unique_specs: list[UniqueConstraintSpec] = []
        for uq_def in unique_constraints or []:
            uq_spec = UniqueConstraintSpec(
                name=uq_def["name"],
                columns=tuple(uq_def["columns"]),
            )
            unique_specs.append(uq_spec)

        return cls(
            name=name,
            schema=schema,
            columns=tuple(columns),
            primary_key=(primary_key,),
            foreign_keys=tuple(foreign_keys),
            indexes=tuple(index_specs),
            triggers=tuple(trigger_specs),
            check_constraints=tuple(check_specs),
            unique_constraints=tuple(unique_specs),
        )


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """Complete database schema specification.

    Represents entire database schema for diffing and migration planning.

    Attributes:
        tables: Tuple of table specifications.
        version: Schema version hash (computed from table specs).
    """

    tables: tuple[TableSpec, ...] = ()
    version: str | None = None

    def get_table(self, name: str) -> TableSpec | None:
        """Get table spec by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    @classmethod
    def from_operables(
        cls,
        operables: dict[str, Operable],
        *,
        schema: str = "public",
    ) -> SchemaSpec:
        """Create SchemaSpec from a mapping of table names to Operables.

        Args:
            operables: Mapping of table name -> Operable.
            schema: Default schema for all tables.

        Returns:
            SchemaSpec with version hash computed from table definitions.
        """
        from lionagi.ln._hash import compute_hash

        tables = [
            TableSpec.from_operable(op, name, schema=schema)
            for name, op in sorted(operables.items())
        ]

        # Compute version hash
        table_data = [
            {
                "name": t.name,
                "schema": t.schema,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.type,
                        "nullable": c.nullable,
                        "default": c.default,
                    }
                    for c in t.columns
                ],
                "foreign_keys": [
                    {"name": fk.name, "columns": fk.columns, "ref_table": fk.ref_table}
                    for fk in t.foreign_keys
                ],
                "indexes": [
                    {"name": idx.name, "columns": idx.columns, "unique": idx.unique}
                    for idx in t.indexes
                ],
            }
            for t in tables
        ]
        version = compute_hash(table_data)

        return cls(tables=tuple(tables), version=version)


# =============================================================================
# SQLSpecAdapter
# =============================================================================


class SQLSpecAdapter:
    """SQL DDL adapter: Spec -> column definition, Operable -> CREATE TABLE.

    One-way adapter for DDL generation. Does not support instance operations.

    Usage:
        op = Operable([Spec(str, name="title"), Spec(int, name="views", default=0)])
        ddl = SQLSpecAdapter.compose_structure(op, "articles", schema="public")
    """

    @classmethod
    def create_field(cls, spec: Spec) -> str:
        """Convert Spec to SQL column definition, e.g., '"name" TEXT NOT NULL'."""
        annotation = spec.annotation
        sql_type, type_nullable, _, _ = python_type_to_sql(annotation)
        nullable = type_nullable or spec.is_nullable

        has_default = False
        default_value = None
        if not is_sentinel(spec.metadata):
            for meta in spec.metadata:
                if meta.key == "default":
                    has_default = True
                    default_value = meta.value
                    break

        # Validate identifier before use
        validate_identifier(spec.name, "column")

        parts = [f'"{spec.name}"', sql_type]

        if not nullable and not has_default:
            parts.append("NOT NULL")

        if has_default and default_value is not None:
            if isinstance(default_value, str):
                parts.append(f"DEFAULT '{default_value}'")
            elif isinstance(default_value, bool):
                parts.append(f"DEFAULT {str(default_value).upper()}")
            elif isinstance(default_value, (int, float)):
                parts.append(f"DEFAULT {default_value}")

        return " ".join(parts)

    @classmethod
    def compose_structure(
        cls,
        op: Operable,
        name: str,
        /,
        *,
        include: set[str] | UnsetType = Unset,
        exclude: set[str] | UnsetType = Unset,
        **kwargs: Any,
    ) -> str:
        """Generate CREATE TABLE DDL from Operable.

        Args:
            op: Operable with Specs
            name: Table name
            include/exclude: Field name filters
            **kwargs: schema (default "public"), if_not_exists (default True),
                primary_key (column name), base_columns (prepend definitions)

        Returns:
            CREATE TABLE DDL statement with FK constraints
        """
        schema = kwargs.get("schema", "public")
        if_not_exists = kwargs.get("if_not_exists", True)
        primary_key = kwargs.get("primary_key")
        base_columns: list[str] = kwargs.get("base_columns", [])

        # Validate table and schema names
        validate_identifier(name, "table")
        validate_identifier(schema, "schema")

        specs = op.get_specs(include=include, exclude=exclude)

        columns: list[str] = list(base_columns)
        foreign_keys: list[str] = []

        for spec in specs:
            if not spec.name:
                continue

            col_def = cls.create_field(spec)

            if primary_key and spec.name == primary_key:
                col_def = col_def.replace(" NOT NULL", "") + " PRIMARY KEY"

            columns.append(col_def)

            fk = extract_kron_db_meta(spec.annotation, metas="FK")
            if isinstance(fk, FKMeta):
                # Validate FK-related identifiers
                validate_identifier(spec.name, "column")
                validate_identifier(fk.table_name, "referenced table")
                validate_identifier(fk.column, "referenced column")

                fk_constraint = (
                    f'CONSTRAINT "fk_{name}_{spec.name}" '
                    f'FOREIGN KEY ("{spec.name}") '
                    f'REFERENCES "{fk.table_name}"("{fk.column}") '
                    f"ON DELETE {OnAction(fk.on_delete)} ON UPDATE {OnAction(fk.on_update)}"
                )

                if fk.deferrable:
                    fk_constraint += " DEFERRABLE"
                    if fk.initially_deferred:
                        fk_constraint += " INITIALLY DEFERRED"

                foreign_keys.append(fk_constraint)

        all_defs = columns + foreign_keys
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        qualified_name = f'"{schema}"."{name}"'

        ddl = f"CREATE TABLE {exists_clause}{qualified_name} (\n"
        ddl += ",\n".join(f"    {col}" for col in all_defs)
        ddl += "\n);"

        return ddl

    @classmethod
    def extract_specs(cls, structure: Any) -> tuple[Spec, ...]:
        """Extract Specs from Pydantic model. Delegates to PydanticSpecAdapter."""
        from lionagi.ln.types.adapters._pydantic import PydanticSpecAdapter

        return PydanticSpecAdapter.extract_specs(structure)

    @classmethod
    def create_index(
        cls,
        table_name: str,
        column: str,
        *,
        index_name: str | None = None,
        unique: bool = False,
        method: str | IndexMethod = IndexMethod.BTREE,
        schema: str = "public",
    ) -> str:
        """Generate CREATE INDEX statement with configurable method."""
        validate_identifier(table_name, "table")
        validate_identifier(column, "column")
        validate_identifier(schema, "schema")

        idx_name = index_name or f"idx_{table_name}_{column}"
        validate_identifier(idx_name, "index")

        method_val = method.value if isinstance(method, IndexMethod) else method
        unique_clause = "UNIQUE " if unique else ""
        qualified_table = f'"{schema}"."{table_name}"'

        return (
            f"CREATE {unique_clause}INDEX IF NOT EXISTS {idx_name} "
            f'ON {qualified_table} USING {method_val} ("{column}");'
        )

    @classmethod
    def create_vector_index(
        cls,
        table_name: str,
        column: str = "embedding",
        *,
        index_name: str | None = None,
        method: str | IndexMethod = IndexMethod.IVFFLAT,
        lists: int = 100,
        schema: str = "public",
    ) -> str:
        """Generate pgvector index (ivfflat or hnsw with vector_cosine_ops).

        Raises:
            ValueError: If method is not 'ivfflat' or 'hnsw'
        """
        validate_identifier(table_name, "table")
        validate_identifier(column, "column")
        validate_identifier(schema, "schema")

        idx_name = index_name or f"idx_{table_name}_{column}_vec"
        validate_identifier(idx_name, "index")

        qualified_table = f'"{schema}"."{table_name}"'
        method_val = method.value if isinstance(method, IndexMethod) else method

        if method_val == "ivfflat":
            return (
                f"CREATE INDEX IF NOT EXISTS {idx_name} "
                f'ON {qualified_table} USING ivfflat ("{column}" vector_cosine_ops) '
                f"WITH (lists = {lists});"
            )
        elif method_val == "hnsw":
            return (
                f"CREATE INDEX IF NOT EXISTS {idx_name} "
                f'ON {qualified_table} USING hnsw ("{column}" vector_cosine_ops);'
            )
        else:
            raise ValueError(f"Unsupported vector index method: {method_val}")

    @classmethod
    def create_table_spec(
        cls,
        op: Operable,
        name: str,
        *,
        schema: str = "public",
        primary_key: str = "id",
        **kwargs: Any,
    ) -> TableSpec:
        """Create TableSpec from Operable for advanced schema operations.

        Args:
            op: Operable with Specs
            name: Table name
            schema: Database schema
            primary_key: Primary key column
            **kwargs: indexes, triggers, check_constraints, unique_constraints

        Returns:
            TableSpec for diffing, introspection, or full DDL generation.
        """
        return TableSpec.from_operable(
            op,
            name,
            schema=schema,
            primary_key=primary_key,
            indexes=kwargs.get("indexes"),
            triggers=kwargs.get("triggers"),
            check_constraints=kwargs.get("check_constraints"),
            unique_constraints=kwargs.get("unique_constraints"),
        )
