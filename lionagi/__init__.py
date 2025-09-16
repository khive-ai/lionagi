# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import logging

from pydantic import BaseModel, Field

# Eager imports for commonly used components
from . import ln as ln
from .operations.node import Operation
from .service.imodel import iModel
from .session.session import Branch, Session
from .version import __version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Module-level lazy loading cache
_lazy_imports = {}


def __getattr__(name: str):
    """Lazy loading for expensive imports."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "types":
        from . import _types as types

        _lazy_imports["types"] = types
        return types
    elif name == "Builder":
        from .operations.builder import OperationGraphBuilder as Builder

        _lazy_imports["Builder"] = Builder
        return Builder

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
    "ln",
)
