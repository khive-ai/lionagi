from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..rule import Rule
from ..rule import (
    BooleanRule, StringRule, NumberRule,
    MappingRule, ModelRule, ChoiceRule
)


@dataclass
class ValidationResult:
    """Result of validation."""
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    fixed: bool = False


class Validator:
    """
    Simple validator using the modernized rule system.

    Examples:
        # Basic validation
        validator = Validator()
        result = await validator.validate("yes", field_type=bool)
        assert result.value is True

        # Dict validation with schema
        data = {"age": "25", "active": "true"}
        schema = {"age": int, "active": bool}
        fixed_data = await validator.validate_dict(data, schema)
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        """
        Initialize validator with rules.

        Args:
            rules: List of Rule instances. Defaults to standard rules if None.
        """
        if rules is None:
            rules = [
                BooleanRule(),
                StringRule(),
                NumberRule(),
                MappingRule(),
                ModelRule(),
                ChoiceRule(),
            ]
        self.rules = rules

    async def validate(
        self,
        value: Any,
        field_name: Optional[str] = None,
        field_type: Optional[type] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Validate a value using registered rules.

        Args:
            value: Value to validate
            field_name: Optional field name for FIELD qualifier
            field_type: Optional type hint for ANNOTATION qualifier
            **kwargs: Additional parameters passed to rules

        Returns:
            ValidationResult with success status and validated/fixed value
        """
        for rule in self.rules:
            try:
                # Check if rule applies using the qualifier system
                if await rule.apply(
                    k=field_name or "",
                    v=value,
                    t=field_type,
                    **kwargs
                ):
                    # Invoke validation and potential fix
                    result = await rule.invoke(
                        k=field_name or "",
                        v=value,
                        t=field_type
                    )

                    return ValidationResult(
                        success=True,
                        value=result if result is not None else value,
                        fixed=result != value
                    )

            except Exception as e:
                # First applicable rule that fails stops validation
                return ValidationResult(
                    success=False,
                    value=value,
                    error=e
                )

        # No rules applied - return original value
        return ValidationResult(
            success=True,
            value=value,
            fixed=False
        )

    async def validate_dict(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, type]] = None,
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Validate dictionary fields against optional schema.

        Args:
            data: Dictionary to validate
            schema: Optional mapping of field names to types
            raise_on_error: If True, raise on validation error. If False, skip failed fields.

        Returns:
            Dictionary with validated/fixed values

        Raises:
            Exception: If raise_on_error=True and validation fails
        """
        result = {}

        for field_name, value in data.items():
            field_type = schema.get(field_name) if schema else None

            validation = await self.validate(
                value,
                field_name=field_name,
                field_type=field_type
            )

            if validation.success:
                result[field_name] = validation.value
            elif raise_on_error:
                raise validation.error
            else:
                # Keep original value if validation fails and not raising
                result[field_name] = value

        return result

    def add_rule(self, rule: Rule, index: Optional[int] = None):
        """Add a rule to the validator."""
        if index is not None:
            self.rules.insert(index, rule)
        else:
            self.rules.append(rule)

    def remove_rule(self, rule: Rule):
        """Remove a rule from the validator."""
        self.rules.remove(rule)