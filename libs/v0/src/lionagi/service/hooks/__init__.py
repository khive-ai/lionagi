# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from lionagi.protocols.types import DataLogger

from ._types import AssosiatedEventInfo, HookDict, HookEventTypes
from .hook_event import HookEvent
from .hook_registry import HookRegistry

global_hook_logger = DataLogger(
    persist_dir="./data/logs",
    subfolder="hooks",
    file_prefix="hook",
    capacity=1000,
)


__all__ = (
    "HookEventTypes",
    "HookDict",
    "AssosiatedEventInfo",
    "HookEvent",
    "HookRegistry",
    "global_hook_logger",
)
