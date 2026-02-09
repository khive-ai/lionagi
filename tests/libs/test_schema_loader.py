# tests/libs/test_schema_loader.py
import importlib
from pathlib import Path

import pytest
from pydantic import BaseModel


def _install_fake_generator(module):
    """
    Patch the target module with a fake datamodel_code_generator API.
    The fake 'generate' writes minimal Pydantic model source code to the output file.
    """

    class _DMT:
        PydanticV2BaseModel = "pyd_v2"

    class _PV:
        PY_312 = "3.12"

    class _IFT:
        JsonSchema = "json"

    def _gen_with_named_class(
        schema_input_data,
        input_file_type,
        input_filename,
        output,
        output_model_type,
        target_python_version,
        base_class,
    ):
        # Class name derived from schema title is used by loader;
        # this generator will emit class matching the resolved name 'UserProfile'.
        code = (
            "from pydantic import BaseModel\n"
            "class UserProfile(BaseModel):\n"
            "    name: str\n"
            "    age: int\n"
        )
        Path(output).write_text(code, encoding="utf-8")

    def _gen_with_model_only(
        schema_input_data,
        input_file_type,
        input_filename,
        output,
        output_model_type,
        target_python_version,
        base_class,
    ):
        code = "from pydantic import BaseModel\nclass Model(BaseModel):\n    value: int\n"
        Path(output).write_text(code, encoding="utf-8")

    module._HAS_DATAMODEL_CODE_GENERATOR = True
    module.DataModelType = _DMT
    module.InputFileType = _IFT
    module.PythonVersion = _PV

    return _gen_with_named_class, _gen_with_model_only


def test_loader_raises_when_generator_not_installed(
    mod_paths, ensure_fake_lionagi
):
    mod = importlib.import_module(mod_paths["schema_mod"])
    # Ensure the loader enforces the presence of datamodel-code-generator
    mod._HAS_DATAMODEL_CODE_GENERATOR = False
    with pytest.raises(ImportError):
        mod.load_pydantic_model_from_schema({"title": "X"}, "X")


def test_loader_happy_path_generates_class_from_title(
    mod_paths, ensure_fake_lionagi
):
    mod = importlib.import_module(mod_paths["schema_mod"])
    gen_named, _ = _install_fake_generator(mod)
    mod.generate = gen_named

    schema = {
        "title": "UserProfile",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    cls = mod.load_pydantic_model_from_schema(schema)
    assert isinstance(cls, type) and issubclass(cls, BaseModel)
    assert cls.__name__ == "UserProfile"
    inst = cls(name="Ocean", age=30)
    assert inst.model_dump() == {"name": "Ocean", "age": 30}


def test_loader_title_sanitization_and_fallback_to_Model(
    mod_paths, ensure_fake_lionagi
):
    mod = importlib.import_module(mod_paths["schema_mod"])
    _, gen_model = _install_fake_generator(mod)
    mod.generate = gen_model

    # Title sanitizes to 'UserProfile', but generator will only emit 'Model'
    cls = mod.load_pydantic_model_from_schema({"title": "User Profile!"})
    assert cls.__name__ == "Model"
    inst = cls(value=42)
    assert inst.model_dump() == {"value": 42}


def test_loader_invalid_json_string_raises(mod_paths, ensure_fake_lionagi):
    mod = importlib.import_module(mod_paths["schema_mod"])
    mod._HAS_DATAMODEL_CODE_GENERATOR = True
    # Need generate present even though we fail earlier on JSON parsing
    mod.generate = lambda *a, **k: None
    with pytest.raises(ValueError):
        mod.load_pydantic_model_from_schema("not a json string")


def test_loader_wrong_type_raises(mod_paths, ensure_fake_lionagi):
    mod = importlib.import_module(mod_paths["schema_mod"])
    mod._HAS_DATAMODEL_CODE_GENERATOR = True
    mod.generate = lambda *a, **k: None
    with pytest.raises(TypeError):
        mod.load_pydantic_model_from_schema(12345)
