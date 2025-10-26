# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING

from . import ln as ln
from .ln.types import DataClass, Operable, Params, Spec, Undefined, Unset
from .version import __version__

if TYPE_CHECKING:
    from pydantic import BaseModel, Field

    from . import _types as types
    from .models.field_model import FieldModel
    from .models.operable_model import OperableModel
    from .operations.builder import OperationGraphBuilder as Builder
    from .operations.node import Operation
    from .protocols.action.manager import load_mcp_tools
    from .protocols.types import (
        Edge,
        Element,
        Event,
        Graph,
        Node,
        Pile,
        Progression,
    )
    from .service.broadcaster import Broadcaster
    from .service.hooks import HookedEvent, HookRegistry
    from .service.imodel import iModel
    from .session.session import Branch, Session

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_lazy_imports = {}


def _get_obj(name: str, module: str):
    global _lazy_imports
    from lionagi.ln import import_module

    obj_ = import_module("lionagi", module_name=module, import_name=name)

    _lazy_imports[name] = obj_
    return obj_


def __getattr__(name: str):
    global _lazy_imports
    if name in _lazy_imports:
        return _lazy_imports[name]

    match name:
        case "Session":
            return _get_obj("Session", "session.session")
        case "Branch":
            return _get_obj("Branch", "session.branch")
        case "iModel":
            return _get_obj("iModel", "service.imodel")
        case "Builder":
            return _get_obj("OperationGraphBuilder", "operations.builder")
        case "Operation":
            return _get_obj("Operation", "operations.node")
        case "load_mcp_tools":
            return _get_obj("load_mcp_tools", "protocols.action.manager")
        case "FieldModel":
            return _get_obj("FieldModel", "models.field_model")
        case "OperableModel":
            return _get_obj("OperableModel", "models.operable_model")
        case "Element":
            return _get_obj("Element", "protocols.generic.element")
        case "Pile":
            return _get_obj("Pile", "protocols.generic.pile")
        case "Progression":
            return _get_obj("Progression", "protocols.generic.progression")
        case "Node":
            return _get_obj("Node", "protocols.graph.node")
        case "Edge":
            return _get_obj("Edge", "protocols.graph.edge")
        case "Graph":
            return _get_obj("Graph", "protocols.graph.graph")
        case "Event":
            return _get_obj("Event", "protocols.generic.event")
        case "HookRegistry":
            return _get_obj("HookRegistry", "service.hooks.hook_registry")
        case "HookedEvent":
            return _get_obj("HookedEvent", "service.hooks.hooked_event")
        case "Broadcaster":
            return _get_obj("Broadcaster", "service.broadcaster")
        case "BaseModel":
            from pydantic import BaseModel

            _lazy_imports["BaseModel"] = BaseModel
            return BaseModel
        case "Field":
            from pydantic import Field

            _lazy_imports["Field"] = Field
            return Field
        case "types":
            from . import _types as types

            _lazy_imports["types"] = types
            return types
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
    "__version__",
    "BaseModel",
    "Branch",
    "Broadcaster",
    "Builder",
    "DataClass",
    "Edge",
    "Element",
    "Event",
    "Field",
    "FieldModel",
    "Graph",
    "HookRegistry",
    "HookedEvent",
    "Node",
    "Operable",
    "OperableModel",
    "Operation",
    "Params",
    "Pile",
    "Progression",
    "Session",
    "Spec",
    "Undefined",
    "Unset",
    "iModel",
    "ln",
    "load_mcp_tools",
    "logger",
    "types",
)
