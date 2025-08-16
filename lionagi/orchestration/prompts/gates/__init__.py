"""Quality gate prompts for orchestrated validations"""

from .factory import get_gate_prompt, list_available_gates

__all__ = [
    "get_gate_prompt",
    "list_available_gates",
]
