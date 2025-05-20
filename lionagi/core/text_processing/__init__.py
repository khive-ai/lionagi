# lionagi.core.text_processing - Core text transformation and processing utilities
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
This module houses core text processing functionalities, including refactored
components from the old `lionagi.libs.token_transform` that are essential to
the Lionagi Vibe. Internal utilities leverage `lionfuncs` where appropriate.
"""

from .synthlang import (
    SynthlangFramework,
    SynthlangTemplate,
    translate_to_synthlang,
)

__all__ = [
    "translate_to_synthlang",
    "SynthlangFramework",
    "SynthlangTemplate",
]
