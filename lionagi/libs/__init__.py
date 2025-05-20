# lionagi.libs - Deprecated
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
DEPRECATED: This module (`lionagi.libs`) and its submodules are largely
deprecated as of lionagi v0.2.0.
Functionalities have been moved to the `lionfuncs` library, standard Python
libraries, or specific `lionagi.core` modules where appropriate.

Please update your imports to use the new locations.
"""

import warnings

warnings.warn(
    "The `lionagi.libs` module and its submodules are deprecated. "
    "Please migrate to `lionfuncs` or other appropriate modules.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = []
