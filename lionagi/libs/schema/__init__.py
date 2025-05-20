# lionagi.libs.schema - Deprecated
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

"""
DEPRECATED: This module (`lionagi.libs.schema`) is deprecated as of lionagi v0.2.0.
Schema generation and manipulation functionalities (e.g., for OpenAI tools)
should now be sourced from `lionfuncs.schema_utils`.
"""

import warnings

warnings.warn(
    "The `lionagi.libs.schema` module is deprecated. Please use `lionfuncs.schema_utils`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = []
