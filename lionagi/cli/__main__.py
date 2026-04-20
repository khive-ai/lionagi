# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``python -m lionagi.cli``.

Used by the ``--background`` flag in ``li o flow`` to respawn the CLI
in a detached subprocess. Direct users should call ``li`` instead.
"""

from __future__ import annotations

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
