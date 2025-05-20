# lionagi.models.mail - Mail system Pydantic models (Package, Mail)
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

from typing import Any, Optional

from pydapter.core import Adaptable
from pydapter.protocols import Temporal


class Package(Temporal, Adaptable):
    """
    Represents a package of information to be sent via the mail system.
    Inherits id, created_at, updated_at from Temporal.
    """

    category: str  # e.g., "message", "tool_config", "branch_state"
    item: Any  # The actual content being sent
    request_source: str | None = None  # ID of the original requester, if any

    class Config:
        arbitrary_types_allowed = True


class Mail(Temporal, Adaptable):
    """
    Represents a piece of mail in the mail system.
    Inherits id, created_at, updated_at from Temporal.
    """

    sender: str  # ID of sending component (e.g., Branch ID)
    recipient: str  # ID of receiving component (e.g., Branch ID)
    package: Package  # The package being sent

    class Config:
        arbitrary_types_allowed = True


__all__ = ["Package", "Mail"]
