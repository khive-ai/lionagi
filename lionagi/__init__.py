# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING

from . import ln as ln
from .version import __version__

if TYPE_CHECKING:
    from pydantic import BaseModel, Field

    from . import _types as types
    from .operations.builder import OperationGraphBuilder as Builder
    from .operations.node import Operation
    from .protocols.action.manager import load_mcp_tools
    from .service.imodel import iModel
    from .session.session import Branch, Session


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Module-level lazy loading cache
_lazy_imports = {}


def __getattr__(name: str):
    """Lazy loading for expensive imports."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    # Lazy load core components
    if name == "Session":
        from .session.session import Session

        _lazy_imports[name] = Session
        return Session
    elif name == "Branch":
        from .session.session import Branch

        _lazy_imports[name] = Branch
        return Branch
    # Lazy load Pydantic components
    elif name == "BaseModel":
        from pydantic import BaseModel

        _lazy_imports[name] = BaseModel
        return BaseModel
    elif name == "Field":
        from pydantic import Field

        _lazy_imports[name] = Field
        return Field
    # Lazy load operations
    elif name == "Operation":
        from .operations.node import Operation

        _lazy_imports[name] = Operation
        return Operation
    elif name == "iModel":
        from .service.imodel import iModel

        _lazy_imports[name] = iModel
        return iModel
    elif name == "types":
        from . import _types as types

        _lazy_imports["types"] = types
        return types
    elif name == "Builder":
        from .operations.builder import OperationGraphBuilder as Builder

        _lazy_imports["Builder"] = Builder
        return Builder
    elif name == "load_mcp_tools":
        from .protocols.action.manager import load_mcp_tools

        _lazy_imports["load_mcp_tools"] = load_mcp_tools
        return load_mcp_tools

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
    "Session",
    "Branch",
    "iModel",
    "types",
    "__version__",
    "BaseModel",
    "Field",
    "logger",
    "Builder",
    "Operation",
    "load_mcp_tools",
    "ln",
)
