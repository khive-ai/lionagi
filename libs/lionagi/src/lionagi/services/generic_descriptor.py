# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Generic HTTP descriptor for flexible JSON service configuration."""

from __future__ import annotations

from typing import Any, Mapping

import msgspec


class HttpDescriptor(msgspec.Struct, kw_only=True):
    """Minimal shape to drive a generic HTTP JSON call."""
    method: str = "POST"
    path: str = "/"
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    query: dict[str, str] = msgspec.field(default_factory=dict)
    # Optional key to pull JSON payload from RequestModel (default: entire request)
    payload_field: str | None = None

    def build_url(self, base_url: str) -> str:
        if self.query:
            from urllib.parse import urlencode
            return f"{base_url.rstrip('/')}{self.path}?{urlencode(self.query)}"
        return f"{base_url.rstrip('/')}{self.path}"