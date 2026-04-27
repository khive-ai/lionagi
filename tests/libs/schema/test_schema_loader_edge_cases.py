import pytest

import lionagi.libs.schema.load_pydantic_model_from_schema as schema_mod
from lionagi.libs.schema.load_pydantic_model_from_schema import (
    load_pydantic_model_from_schema,
)


def test_schema_loader_raises_runtime_error_when_codegen_fallback_unavailable(
    monkeypatch,
):
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": ["string", "unsupported"]},
        },
    }

    monkeypatch.setattr(schema_mod, "_HAS_DATAMODEL_CODE_GENERATOR", False)

    with pytest.raises(RuntimeError, match="datamodel-code-generator"):
        load_pydantic_model_from_schema(schema, "BadModel")
