"""Validation morphism - Rules as system morphisms.

This shows how the Rule system integrates as morphisms
that are executed by the runtime system.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .base import SystemMorphism
from ..rule.base import Rule
from ..ln.types import Params

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


@dataclass
class ValidationContext(Params):
    """Context for validation morphism."""
    field_name: str
    value: Any
    field_type: type = None
    auto_fix: bool = True
    strict: bool = False


class ValidationMorphism(SystemMorphism):
    """System morphism that applies validation rules.

    This shows how Rules become morphisms in the new architecture.
    Instead of Branch managing rules, validation is a morphism
    that runs as part of the execution pipeline.
    """

    meta = {
        "name": "validation",
        "description": "Field validation system morphism",
        "priority": 30,  # Run early but after auth/rate-limit
    }

    def __init__(self, rule: Rule):
        """Initialize with a specific rule.

        Args:
            rule: The validation rule to apply
        """
        self.rule = rule
        self.requires = {"validation.execute"}

    async def should_run(self, branch: "Branch", target) -> bool:
        """Determine if validation should run for target morphism.

        Validation runs for morphisms that:
        1. Have validation context
        2. Are not themselves validation morphisms
        """
        # Don't validate validation morphisms (avoid recursion)
        if isinstance(target, ValidationMorphism):
            return False

        # Check if target has validation context
        if hasattr(target, 'ctx') and isinstance(target.ctx, ValidationContext):
            return True

        # Check if target declares it needs validation
        if hasattr(target, 'needs_validation') and target.needs_validation:
            return True

        return False

    async def pre(self, branch: "Branch", /, **kw) -> bool:
        """Pre-validation checks."""
        # Could check if branch has validation capability
        return True

    async def _invoke(self, branch: "Branch", /, **kw) -> Dict[str, Any]:
        """Apply validation rule."""
        # Get validation context
        ctx = kw.get('validation_context')
        if not ctx:
            return {"validated": False, "reason": "No validation context"}

        field_name = ctx.get('field_name')
        value = ctx.get('value')
        field_type = ctx.get('field_type')

        # Check if rule applies
        if not await self.rule.apply(field_name, value, field_type, **kw):
            return {
                "validated": False,
                "reason": f"Rule {self.rule.__class__.__name__} does not apply"
            }

        # Apply rule
        try:
            await self.rule.validate(value, field_type, **kw)
            return {
                "validated": True,
                "value": value,
                "rule": self.rule.__class__.__name__
            }
        except Exception as e:
            # Try auto-fix if enabled
            if ctx.get('auto_fix', True) and self.rule.auto_fix:
                try:
                    fixed_value = await self.rule.perform_fix(value, field_type)
                    return {
                        "validated": True,
                        "value": fixed_value,
                        "fixed": True,
                        "original": value,
                        "rule": self.rule.__class__.__name__
                    }
                except Exception as fix_error:
                    return {
                        "validated": False,
                        "error": str(e),
                        "fix_error": str(fix_error),
                        "rule": self.rule.__class__.__name__
                    }
            else:
                return {
                    "validated": False,
                    "error": str(e),
                    "rule": self.rule.__class__.__name__
                }

    async def post(self, branch: "Branch", /, result: dict) -> bool:
        """Post-validation checks."""
        # Could verify validation result integrity
        return True


# Convenience functions to create validation morphisms
def create_string_validation() -> ValidationMorphism:
    """Create string validation morphism."""
    from ..rule.string import StringRule
    return ValidationMorphism(StringRule())


def create_number_validation() -> ValidationMorphism:
    """Create number validation morphism."""
    from ..rule.number import NumberRule
    return ValidationMorphism(NumberRule())


def create_choice_validation(choices: list) -> ValidationMorphism:
    """Create choice validation morphism."""
    from ..rule.choice import ChoiceRule
    rule = ChoiceRule()
    rule.keys = choices
    return ValidationMorphism(rule)