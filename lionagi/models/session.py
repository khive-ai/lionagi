# lionagi.models.session - Session Pydantic model (data part)
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

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# from .branch import BranchModel # Forward reference or direct import later

# Assuming Temporal and Adaptable are available or placeholder defined
try:
    from pydapter.protocols import Temporal
except ImportError:

    class Temporal(BaseModel):  # type: ignore
        id: str = Field(default_factory=lambda: "temp_id")
        created_at: Any = Field(default_factory=lambda: "timestamp")
        updated_at: Any = Field(default_factory=lambda: "timestamp")

        class Config:
            arbitrary_types_allowed = True


try:
    from pydapter.core import Adaptable
except ImportError:

    class Adaptable:  # type: ignore
        pass


# Placeholder for BranchModel to avoid circular import during initial file creation
class BranchModel(BaseModel):  # Placeholder
    id: str
    name: str | None = None
    # ... other fields


class SessionModel(Temporal, Adaptable):
    """
    Pydantic model representing the data state of a Session.
    The behavioral aspects will be in lionagi.core.session.Session.
    """

    name: str | None = None
    branches: dict[str, BranchModel] = Field(
        default_factory=dict
    )  # branch_id -> BranchModel
    default_branch_id: str | None = None
    mail_manager_id: str | None = None  # ID of its mail manager

    # Default LLM service configuration for new branches in this session.
    # This could be a direct dict or a model like ServiceInterface if that's made adaptable.
    # TDS Sec 4 Session model shows: default_service_config: Optional[Dict[str, Any]] = None
    default_service_config: dict[str, Any] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


__all__ = ["SessionModel"]
