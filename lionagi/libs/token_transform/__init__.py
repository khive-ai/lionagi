# lionagi.libs.token_transform - Deprecated / Under Review
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
DEPRECATED/UNDER REVIEW: This module (`lionagi.libs.token_transform`) is deprecated
as of lionagi v0.2.0 or under review for refactoring.
Core text processing functionalities may move to `lionagi.core.text_processing`
and leverage `lionfuncs` where possible. Specialized transformations might be
spun out or contributed to `lionfuncs` if broadly applicable.
"""

import warnings

warnings.warn(
    "The `lionagi.libs.token_transform` module is deprecated or under review. "
    "Its functionalities may be moved or replaced.",
    DeprecationWarning,
    stacklevel=2,
)

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
