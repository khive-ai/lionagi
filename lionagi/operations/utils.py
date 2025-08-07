# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from lionagi.fields.instruct import Instruct
from lionagi.session.session import Branch, Session
from lionagi.utils import coalesce, validate_param


def prepare_session(
    session: Session | None = None,
    branch: Branch | None = None,
    branch_kwargs=None,
) -> tuple[Session, Branch]:
    """Prepare a session and branch for operations.

    Uses param validation utilities for cleaner null handling.
    """
    branch_kwargs = validate_param(branch_kwargs, "branch_kwargs", default={})

    if session is not None:
        if branch is not None:
            branch: Branch = session.branches[branch]
        else:
            branch = session.new_branch(**branch_kwargs)
    else:
        session = Session()
        if isinstance(branch, Branch):
            session.branches.include(branch)
            session.default_branch = branch
        if branch is None:
            branch = session.new_branch(**branch_kwargs)

    return session, branch


def prepare_instruct(instruct: Instruct | dict, prompt: str):
    if isinstance(instruct, Instruct):
        instruct = instruct.to_dict()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel object or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{prompt}\n{guidance}"
    return instruct
