# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Composable LLM pipeline stages: generate, parse, structure, operate, act, react."""

from __future__ import annotations

from .act import ActParams, act
from .generate import GenerateParams, generate
from .operate import OperateParams, operate
from .parse import ParseParams, parse
from .react import (
    Analysis,
    PlannedAction,
    ReActAnalysis,
    ReActParams,
    react,
    react_stream,
)
from .run import RunParams, run, run_and_collect
from .specs import (
    Action,
    ActionResult,
    Instruct,
    get_action_result_spec,
    get_action_spec,
    get_instruct_spec,
)
from .structure import StructureParams, structure
from .utils import ReturnAs


def _param_value(params, name: str):
    if params is None:
        return None
    if isinstance(params, dict):
        return params.get(name)
    if hasattr(params, "is_sentinel_field") and params.is_sentinel_field(name):
        return None
    return getattr(params, name, None)


def _imodel_service_rights(params, *, session=None, **_) -> set[str]:
    from lionagi.ln.types._sentinel import is_sentinel

    imodel = _param_value(params, "imodel")
    if is_sentinel(imodel, additions={"none", "empty"}):
        imodel = None
    if imodel is None:
        imodel = getattr(session, "default_gen_model", None)
    if isinstance(imodel, str):
        return {f"service.call:{imodel}"}
    name = getattr(imodel, "name", None)
    return {f"service.call:{name}"} if name else set()


def builtin_operation_declarations():
    """Return built-in operation declarations for per-session registration."""
    from lionagi.beta.session.session import OperationDecl

    return {
        "generate": OperationDecl(
            generate,
            requires=frozenset({"net.out:*"}),
            provides=frozenset({"llm.response"}),
            required_rights=_imodel_service_rights,
        ),
        "act": OperationDecl(
            act,
            requires=frozenset({"session.active"}),
            provides=frozenset({"action.responses"}),
        ),
        "parse": OperationDecl(
            parse,
            provides=frozenset({"structured.data"}),
        ),
        "structure": OperationDecl(
            structure,
            requires=frozenset({"net.out:*"}),
            provides=frozenset({"structured.data"}),
        ),
        "operate": OperationDecl(
            operate,
            requires=frozenset({"net.out:*"}),
            provides=frozenset({"structured.response"}),
        ),
        "run": OperationDecl(
            run,
            requires=frozenset({"net.out:*"}),
            provides=frozenset({"stream.text"}),
            required_rights=_imodel_service_rights,
        ),
        "react": OperationDecl(
            react,
            requires=frozenset({"net.out:*"}),
            provides=frozenset({"react.answer"}),
        ),
        "react_stream": OperationDecl(
            react_stream,
            requires=frozenset({"net.out:*"}),
            provides=frozenset({"react.answer"}),
        ),
    }

__all__ = (
    "ActParams",
    "Action",
    "ActionResult",
    "Analysis",
    "GenerateParams",
    "Instruct",
    "OperateParams",
    "ParseParams",
    "PlannedAction",
    "ReActAnalysis",
    "ReActParams",
    "ReturnAs",
    "RunParams",
    "StructureParams",
    "act",
    "builtin_operation_declarations",
    "generate",
    "get_action_result_spec",
    "get_action_spec",
    "get_instruct_spec",
    "operate",
    "parse",
    "react",
    "react_stream",
    "run",
    "run_and_collect",
    "structure",
)
