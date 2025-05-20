# lionagi._errors - Deprecated Error Definitions
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
DEPRECATED: This module is deprecated as of lionagi v0.2.0.
Error definitions have been moved to `lionagi.error` (for lionagi-specific
exceptions, which subclass `lionfuncs.errors.LionError`) or are sourced
directly from `lionfuncs.errors`.

Please update your imports accordingly.
"""

import warnings

warnings.warn(
    "The `lionagi._errors` module is deprecated and will be removed in a future version. "
    "Please use `lionagi.error` or `lionfuncs.errors`.",
    DeprecationWarning,
    stacklevel=2,
)

# For a transition period, we could re-export from lionagi.error,
# but it's better to encourage direct imports from the new location.
# from .error import * # Example of re-exporting if needed

__all__ = []  # Intentionally empty
