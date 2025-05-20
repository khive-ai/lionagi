# lionagi.error - Custom exceptions for the lionagi package
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

# Assuming lionfuncs.errors.LionError is available
# from lionfuncs.errors import LionError # Actual import
try:
    from lionfuncs.errors import LionError
except ImportError:
    class LionError(Exception): # type: ignore
        """Base class for lionfuncs errors."""
        pass

class LionagiError(LionError):
    """Base class for lionagi specific errors."""
    pass

class LionagiConfigurationError(LionagiError):
    """Errors related to lionagi configuration."""
    pass

class LionagiExecutionError(LionagiError):
    """Errors related to the execution flow within lionagi."""
    pass

class LionagiSessionError(LionagiExecutionError):
    """Errors specific to Session operations."""
    pass

class LionagiBranchError(LionagiExecutionError):
    """Errors specific to Branch operations."""
    pass

class LionagiToolError(LionagiExecutionError):
    """Errors related to tool definition or execution."""
    pass

class LionagiMessageError(LionagiError):
    """Errors related to message validation or processing."""
    pass

class LionagiMailError(LionagiError):
    """Errors related to the mail system."""
    pass


__all__ = [
    "LionagiError",
    "LionagiConfigurationError",
    "LionagiExecutionError",
    "LionagiSessionError",
    "LionagiBranchError",
    "LionagiToolError",
    "LionagiMessageError",
    "LionagiMailError",
]