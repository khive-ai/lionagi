# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""RoundOutcome — algebraic data type for a single LNDL round's result.

The LNDL react loop runs N rounds, each one returning a RoundOutcome. The
outer loop matches on the outcome to decide what to do next. This replaces
the prior 4 ad-hoc branches (strict→fuzzy parse, fuzzy fail, llm_reparse,
resolve fail) with a single state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = (
    "Continue",
    "Exhausted",
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
class Exhausted:
    """Hit max_rounds without producing a Success. Carries the last
    error (if any) so the caller can report something useful."""

    last_error: str | None = None


@dataclass(slots=True, frozen=True)
class Failed:
    """Unrecoverable error — no point retrying. Caller should raise."""

    error: BaseException


RoundOutcome = Success | Continue | Retry | Exhausted | Failed
