# lionagi - An Intelligence Operating System
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
lionagi is a high-level abstraction layer for building AI/LLM applications,
focusing on intuitive, elegant, and powerful orchestration.
"""

# Public API Exports (to be populated based on TDS Section 3.1 and 5.5)
# from .core.session import Session
# from .core.branch import Branch
# from .models.message import Message
# from .models.tool import Tool
# from .services.factory import Service

from .version import __version__

# Placeholder for actual exports once core components are defined
__all__ = [
    "__version__",
    # "Session",
    # "Branch",
    # "Message",
    # "Tool",
    # "Service",
]

# TODO: Add logging configuration if needed, or rely on lionfuncs/user config
