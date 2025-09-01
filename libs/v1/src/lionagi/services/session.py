"""Services wrapper for Session.

Exposes the high-level Session API from the kernel execution layer.
"""

from ..kernel.execution.session import Session as _Session

__all__ = ["Session"]


class Session(_Session):
    pass

