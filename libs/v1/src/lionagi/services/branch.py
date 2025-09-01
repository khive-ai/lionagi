"""Services wrapper for Branch.

Exposes the high-level Branch API from the kernel execution layer.
"""

from ..kernel.execution.branch import Branch as _Branch

__all__ = ["Branch"]


class Branch(_Branch):
    pass

