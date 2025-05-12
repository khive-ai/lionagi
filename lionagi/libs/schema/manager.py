from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class SchemaUtil:
    @staticmethod
    def load_pydantic_model_from_schema(
        schema: str | dict[str, Any],
        model_name: str = "DynamicModel",
        /,
        pydantic_version=None,
        python_version=None,
    ) -> type[BaseModel]:
        from .load_pydantic_model_from_schema import (
            load_pydantic_model_from_schema,
        )

        return load_pydantic_model_from_schema(
            schema,
            model_name=model_name,
            pydantic_version=pydantic_version,
            python_version=python_version,
        )

    @staticmethod
    def extract_json_schema(
        data: Any,
        *,
        sep: str = "|",
        coerce_keys: bool = True,
        dynamic: bool = True,
        coerce_sequence: Literal["dict", "list"] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        from .json_schema import extract_json_schema

        return extract_json_schema(
            data,
            sep=sep,
            coerce_keys=coerce_keys,
            dynamic=dynamic,
            coerce_sequence=coerce_sequence,
            max_depth=max_depth,
        )

    @staticmethod
    def json_schema_to_cfg(
        schema: dict[str, Any], start_symbol: str = "S"
    ) -> list[tuple[str, list[str]]]:
        from .json_schema import json_schema_to_cfg

        return json_schema_to_cfg(schema, start_symbol=start_symbol)

    @staticmethod
    def json_schema_to_regex(schema: dict[str, Any]) -> str:
        from .json_schema import json_schema_to_regex

        return json_schema_to_regex(schema)

    @staticmethod
    def print_cfg(productions: list[tuple[str, list[str]]]):
        from .json_schema import print_cfg

        print_cfg(productions)

    @staticmethod
    def function_to_schema(
        f_,
        style: Literal["google", "rest"] = "google",
        *,
        request_options: dict[str, Any] | None = None,
        strict: bool = None,
        func_description: str = None,
        parametert_description: dict[str, str] = None,
        return_obj: bool = False,
    ) -> dict:
        from .function_to_schema import function_to_schema

        return function_to_schema(
            f_,
            style=style,
            request_options=request_options,
            strict=strict,
            func_description=func_description,
            parametert_description=parametert_description,
            return_obj=return_obj,
        )

    @staticmethod
    def extract_docstring(
        func: Any, style: Literal["google", "rest"] = "google"
    ) -> tuple[str | None, dict[str, str]]:
        from .extract_docstring import extract_docstring

        return extract_docstring(func, style=style)

    @staticmethod
    def extract_code_block(
        str_to_parse: str,
        return_as_list: bool = False,
        languages: list[str] | None = None,
        categorize: bool = False,
    ) -> str | list[str] | dict[str, list[str]]:
        from .extract_code_block import extract_code_block

        return extract_code_block(
            str_to_parse,
            return_as_list=return_as_list,
            languages=languages,
            categorize=categorize,
        )

    @staticmethod
    def as_readable(
        input_: Any,
        /,
        *,
        md: bool = False,
        format_curly: bool = False,
        display_str: bool = False,
        max_chars: int | None = None,
    ) -> str:
        from .as_readable import as_readable

        return as_readable(
            input_,
            md=md,
            format_curly=format_curly,
            display_str=display_str,
            max_chars=max_chars,
        )

    @staticmethod
    def hash_dict(
        input_: dict[str, Any],
        /,
        *,
        hash_type: Literal["sha256", "md5"] = "sha256",
        hash_str: bool = False,
    ) -> str:
        from .hash_dict import hash_dict

        return hash_dict(input_, hash_type=hash_type, hash_str=hash_str)
