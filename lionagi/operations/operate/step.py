"""Step factory methods for creating configured Operative instances.

This module provides convenient factory methods for common operation patterns,
particularly for ReAct and multi-step workflows.
"""

from typing import TYPE_CHECKING, Literal

from lionagi.ln.types import Spec

from ..fields import get_default_field
from .operative import Operative

if TYPE_CHECKING:
    from pydantic import BaseModel


class Step:
    """Factory methods for common Operative patterns.

    Provides convenient methods to create Operative instances with
    pre-configured field specifications for common patterns like ReAct,
    question-answering, and task execution.

    Example:
        >>> # Create ReAct operative
        >>> op = Step.request_operative(
        ...     name="ReActStep",
        ...     reason=True,
        ...     actions=True,
        ...     fields={
        ...         "observation": Spec(str, description="Environment state")
        ...     }
        ... )
        >>>
        >>> # Validate response
        >>> response = op.validate_response('{"reason": "...", "action_required": true}')
    """

    @staticmethod
    def request_operative(
        *,
        name: str | None = None,
        adapter: Literal["pydantic"] = "pydantic",
        reason: bool = False,
        actions: bool = False,
        fields: dict[str, Spec] | None = None,
        max_retries: int = 3,
        auto_retry_parse: bool = True,
        base_type: type["BaseModel"] | None = None,
    ) -> Operative:
        """Create request-configured Operative with common field patterns.

        Args:
            name: Operative name
            adapter: Validation framework to use
            reason: Add reasoning trace field
            actions: Add action request/response fields
            fields: Additional custom field specs
            max_retries: Max validation retries
            auto_retry_parse: Auto-retry on parse failure
            base_type: Base Pydantic model to extend

        Returns:
            Configured Operative instance

        Example:
            >>> op = Step.request_operative(
            ...     name="ReActStep",
            ...     reason=True,
            ...     actions=True,
            ...     fields={
            ...         "observation": Spec(str, description="Environment state"),
            ...         "confidence": Spec(float, ge=0.0, le=1.0),
            ...     }
            ... )
        """
        from lionagi.ln.types import Operable

        # Build fields list upfront
        all_fields = []

        # Add common fields
        if reason:
            all_fields.append(get_default_field("reason"))

        if actions:
            all_fields.append(get_default_field("action_required"))
            all_fields.append(get_default_field("action_requests"))
            all_fields.append(
                get_default_field("action_responses")
            )  # Add response field too

        # Add custom fields
        if fields:
            for field_name, spec in fields.items():
                # Ensure spec has name
                if not hasattr(spec, "name") or not spec.name:
                    spec = (
                        spec.with_updates(name=field_name)
                        if hasattr(spec, "with_updates")
                        else spec
                    )
                all_fields.append(spec)

        # Create single Operable with all fields
        operable = Operable(
            __op_fields__=frozenset(all_fields),
            name=name or (base_type.__name__ if base_type else "Operative"),
        )

        # Create Operative with the Operable
        # Request will exclude action_responses
        request_exclude = {"action_responses"} if actions else set()

        return Operative(
            name=name,
            adapter=adapter,
            max_retries=max_retries,
            auto_retry_parse=auto_retry_parse,
            base_type=base_type,
            operable=operable,
            request_exclude=request_exclude,
        )

    @staticmethod
    def respond_operative(
        operative: Operative,
        additional_fields: dict[str, Spec] | None = None,
    ) -> Operative:
        """Create response type from operative.

        Args:
            operative: Source operative with all fields
            additional_fields: Extra fields for response (not commonly used)

        Returns:
            Operative with response type configured

        Example:
            >>> op = Step.request_operative(reason=True, actions=True)
            >>> op = Step.respond_operative(op)
        """
        # If additional fields provided, we need to create a new Operative
        if additional_fields:
            from lionagi.ln.types import Operable

            # Get existing fields
            existing_fields = list(operative.operable.__op_fields__)

            # Add new fields
            for field_name, spec in additional_fields.items():
                if not hasattr(spec, "name") or not spec.name:
                    spec = (
                        spec.with_updates(name=field_name)
                        if hasattr(spec, "with_updates")
                        else spec
                    )
                existing_fields.append(spec)

            # Create new Operable with all fields
            new_operable = Operable(
                __op_fields__=frozenset(existing_fields), name=operative.name
            )

            # Create new Operative
            return Operative(
                name=operative.name,
                adapter=operative.adapter,
                max_retries=operative.max_retries,
                auto_retry_parse=operative.auto_retry_parse,
                base_type=operative.base_type,
                operable=new_operable,
                request_exclude=operative.request_exclude,
            )

        # Otherwise just create response model on existing operative
        operative.create_response_model()
        return operative


__all__ = ("Step",)
