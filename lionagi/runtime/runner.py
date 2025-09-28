"""Runner - Executes morphisms with IPU observation and system morphisms.

Adapted from v1 runner with focus on async execution and system morphisms.
"""

import asyncio
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .ipu import IPU
from .eventbus import EventBus, emit_morphism_start, emit_morphism_finish
from .policy import policy_check
from ..morphism.registry import get_system_morphisms

if TYPE_CHECKING:
    from lionagi.session.branch import Branch
    from lionagi.morphism.base import Morphism, SystemMorphism


@dataclass
class ExecutionContext:
    """Context for a single morphism execution."""
    branch: "Branch"
    morphism: "Morphism"
    system_morphisms: List["SystemMorphism"] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


class Runner:
    """Executes morphisms with system-level processing.

    The Runner:
    1. Checks policies
    2. Runs system morphisms (pre)
    3. Executes the target morphism with IPU observation
    4. Runs system morphisms (post)
    5. Handles results and errors
    """

    def __init__(
        self,
        ipu: Optional[IPU] = None,
        event_bus: Optional[EventBus] = None,
        use_system_morphisms: bool = True
    ):
        """Initialize runner.

        Args:
            ipu: IPU for invariant checking
            event_bus: Event bus for service communication
            use_system_morphisms: Whether to run system morphisms
        """
        self.ipu = ipu or IPU(event_bus=event_bus)
        self.event_bus = event_bus or EventBus()
        self.use_system_morphisms = use_system_morphisms

        # Install default event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up default event handlers."""

        async def on_start(event):
            """Log morphism start."""
            # Could integrate with logging system
            pass

        async def on_finish(event):
            """Log morphism completion."""
            # Could integrate with metrics system
            pass

        self.event_bus.subscribe("morphism.start", on_start)
        self.event_bus.subscribe("morphism.finish", on_finish)

    async def execute(
        self,
        branch: "Branch",
        morphism: "Morphism",
        **params
    ) -> Dict[str, Any]:
        """Execute a single morphism with full processing.

        Args:
            branch: Branch providing context
            morphism: Morphism to execute
            **params: Parameters for morphism

        Returns:
            Morphism result

        Raises:
            PermissionError: If policy check fails
            AssertionError: If invariants are violated (strict mode)
        """
        # Create execution context
        ctx = ExecutionContext(
            branch=branch,
            morphism=morphism,
            params=params
        )

        # 1. Policy check
        if not await self._check_policy(ctx):
            raise PermissionError(
                f"Policy denied for morphism {morphism.name}: "
                f"requires {morphism.requires}"
            )

        # 2. Get applicable system morphisms
        if self.use_system_morphisms:
            ctx.system_morphisms = await self._get_system_morphisms(ctx)

        # 3. Emit start event
        await emit_morphism_start(self.event_bus, branch, morphism)

        # 4. Run pre-phase system morphisms
        await self._run_system_morphisms_pre(ctx)

        # 5. IPU pre-observation
        await self.ipu.before_morphism(branch, morphism)

        # 6. Execute morphism pre-condition
        if not await morphism.pre(branch, **params):
            raise ValueError(f"Pre-condition failed for {morphism.name}")

        # 7. Execute main morphism
        try:
            # Merge branch context with params
            merged_params = self._merge_params(branch, params)
            result = await morphism.invoke(branch)
            ctx.result = result
        except Exception as e:
            ctx.errors.append(str(e))
            # Allow system morphisms to handle errors
            if not await self._handle_error(ctx, e):
                raise

        # 8. Execute morphism post-condition
        if ctx.result and not await morphism.post(branch, ctx.result):
            raise ValueError(f"Post-condition failed for {morphism.name}")

        # 9. IPU post-observation
        if ctx.result:
            await self.ipu.after_morphism(branch, morphism, ctx.result)

        # 10. Run post-phase system morphisms
        await self._run_system_morphisms_post(ctx)

        # 11. Emit finish event
        if ctx.result:
            await emit_morphism_finish(
                self.event_bus, branch, morphism, ctx.result
            )

        return ctx.result or {}

    async def _check_policy(self, ctx: ExecutionContext) -> bool:
        """Check if morphism execution is allowed."""
        # Compute dynamic requirements if available
        dynamic_reqs = None
        if hasattr(ctx.morphism, 'required_rights'):
            try:
                dynamic_reqs = ctx.morphism.required_rights(**ctx.params)
            except Exception:
                # Fall back to static requirements
                pass

        return policy_check(ctx.branch, ctx.morphism, dynamic_reqs)

    async def _get_system_morphisms(
        self,
        ctx: ExecutionContext
    ) -> List["SystemMorphism"]:
        """Get applicable system morphisms."""
        all_system = get_system_morphisms()
        applicable = []

        for morphism_cls in all_system:
            morphism = morphism_cls()
            if await morphism.should_run(ctx.branch, ctx.morphism):
                applicable.append(morphism)

        # Sort by priority
        applicable.sort(key=lambda m: m.priority)
        return applicable

    async def _run_system_morphisms_pre(self, ctx: ExecutionContext):
        """Run pre-phase of system morphisms."""
        for sys_morphism in ctx.system_morphisms:
            try:
                if not await sys_morphism.pre(ctx.branch, **ctx.params):
                    # System morphism can veto execution
                    raise ValueError(
                        f"System morphism {sys_morphism.name} "
                        f"vetoed execution"
                    )
            except Exception as e:
                # Log but don't stop for non-critical system morphisms
                if sys_morphism.priority < 20:  # Critical priority
                    raise
                ctx.errors.append(f"System morphism {sys_morphism.name}: {e}")

    async def _run_system_morphisms_post(self, ctx: ExecutionContext):
        """Run post-phase of system morphisms."""
        if not ctx.result:
            return

        for sys_morphism in reversed(ctx.system_morphisms):
            try:
                if not await sys_morphism.post(ctx.branch, ctx.result):
                    # System morphism detected issue with result
                    ctx.errors.append(
                        f"System morphism {sys_morphism.name} "
                        f"rejected result"
                    )
            except Exception as e:
                ctx.errors.append(f"System morphism {sys_morphism.name}: {e}")

    def _merge_params(
        self,
        branch: "Branch",
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge branch context with parameters.

        This implements the v1 pattern where branch.ctx provides
        default values that can be overridden by explicit params.
        """
        merged = {}

        # Start with branch context if available
        if hasattr(branch, 'ctx'):
            merged.update(branch.ctx)

        # Override with explicit params
        merged.update(params)

        return merged

    async def _handle_error(
        self,
        ctx: ExecutionContext,
        error: Exception
    ) -> bool:
        """Handle execution error.

        Returns:
            True if error was handled, False to propagate
        """
        # Emit error event
        await self.event_bus.emit(
            "morphism.error",
            {
                "branch_id": str(ctx.branch.id),
                "morphism_name": ctx.morphism.name,
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )

        # Check if any system morphism handles errors
        for sys_morphism in ctx.system_morphisms:
            if hasattr(sys_morphism, 'handle_error'):
                if await sys_morphism.handle_error(ctx.branch, error):
                    return True

        return False


# Convenience function for simple execution
async def execute_morphism(
    branch: "Branch",
    morphism: "Morphism",
    **params
) -> Dict[str, Any]:
    """Execute a morphism with default runner.

    This is a convenience function for simple cases.
    """
    runner = Runner()
    return await runner.execute(branch, morphism, **params)