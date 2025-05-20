# lionagi.core.text_processing.base - Base for token transformation
# Copyright (c) 2023-present, HaiyangLi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field

from lionagi.tools.base import Resource, ResourceCategory # TODO: Update import path after tools refactor

here = Path(__file__).parent.resolve()
MAPPING_PATH = "synthlang_/resources/mapping" # This relative path might need adjustment


class TokenMappingTemplate(str, Enum):
    RUST_CHINESE = "rust_chinese"
    LION_EMOJI = "lion_emoji"
    PYTHON_MATH = "python_math"

    @property
    def fp(self) -> Path:
        # TODO: Verify this path construction after moving files.
        # It assumes synthlang_ resources are moved alongside or made accessible.
        return here / MAPPING_PATH / f"{self.value}_mapping.toml"


class TokenMapping(Resource):
    category: ResourceCategory = Field(
        default=ResourceCategory.UTILITY, frozen=True
    )
    content: dict

    @classmethod
    def load_from_template(
        cls, template: TokenMappingTemplate | str
    ) -> TokenMapping:
        if isinstance(template, str):
            template = template.lower().strip()
            template = (
                template.replace(".toml", "")
                .replace(" ", "_")
                .replace("-", "_")
                .strip()
            )
            if template.endswith("_mapping"):
                template = template[:-8]
            if "/" in template:
                template = template.split("/")[-1]
            template = TokenMappingTemplate(template)

        if isinstance(template, TokenMappingTemplate):
            # TODO: Ensure template.fp correctly resolves after file moves.
            # This might require `synthlang_` to be a sub-package of `text_processing`
            # or for MAPPING_PATH to be an absolute or configurable path.
            # For now, assuming it works if synthlang_ resources are moved correctly.
            template_file_path = template.fp
            # Assuming Resource.adapt_from can handle Path objects and .toml
            # This part depends on pydapter's capabilities for Resource model
            if not template_file_path.exists():
                 raise FileNotFoundError(f"Token mapping template file not found: {template_file_path}")
            # The adapt_from method needs to be available on Resource or its parent.
            # This is a placeholder for how it might work with pydapter.
            # return cls.adapt_from(template_file_path, format_key=".toml", many=False) # Hypothetical
            # Actual implementation will depend on how Resource/Adaptable handles file loading.
            # For now, let's assume a direct load if adapt_from isn't suitable here.
            import toml
            data = toml.load(template_file_path)
            return cls(content=data.get("content", {}), name=template.value)


        raise ValueError(
            f"Invalid template: {template}. Must be a TokenMappingTemplate or a valid path."
        )

__all__ = ["TokenMappingTemplate", "TokenMapping"]