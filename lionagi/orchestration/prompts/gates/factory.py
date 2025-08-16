"""Quality gate prompt factory for composable validation instructions"""

from .design import DESIGN_GATE_PROMPT
from .documentation import DOCUMENTATION_GATE_PROMPT
from .performance import PERFORMANCE_GATE_PROMPT
from .security import SECURITY_GATE_PROMPT
from .testing import TESTING_GATE_PROMPT


def get_gate_prompt(gate_type: str) -> str:
    """Get the default prompt for a specific gate type"""
    gate_prompts = {
        "design": DESIGN_GATE_PROMPT,
        "security": SECURITY_GATE_PROMPT,
        "performance": PERFORMANCE_GATE_PROMPT,
        "testing": TESTING_GATE_PROMPT,
        "documentation": DOCUMENTATION_GATE_PROMPT,
    }
    return gate_prompts.get(gate_type, "")


def list_available_gates() -> list[str]:
    """List all available gate types"""
    return ["design", "security", "performance", "testing", "documentation"]
