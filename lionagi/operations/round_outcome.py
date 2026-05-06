# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""RoundOutcome — algebraic data type for a single LNDL round's result.

The LNDL react loop in `lionagi.lndl.orchestrator` runs N rounds and
matches on the outcome to drive the next step. Loop exhaustion is signalled
by exiting the loop normally without a Success / Failed; the orchestrator
raises `MissingOutBlockError` with the last accumulated error attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = (
    "Continue",
    "Failed",
    "Retry",
    "RoundOutcome",
    "Success",
)


@dataclass(slots=True, frozen=True)
class Success:
    """Round produced a valid OUT{} block that resolved to a typed model."""

    output: Any


@dataclass(slots=True, frozen=True)
class Continue:
    """Round had no OUT{} block. Any note.X commits already landed in
    scratchpad; chat history holds the narrative. Loop again."""

    notes_committed: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class Retry:
    """Round had an OUT{} but parse or resolve failed. Scratchpad state
    from prior rounds is intact; the error message is fed to the model
    next round so it can self-correct."""

    error: str
    note_keys: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class Failed:
    """Unrecoverable error — no point retrying. Caller should raise."""

    error: BaseException


RoundOutcome = Success | Continue | Retry | Failed
