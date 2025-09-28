"""Morphism binding - context-based parameter binding.

Adapted from v1 binders.py - binds Branch context to morphism parameters.
"""

from typing import Dict, Any, Mapping, TYPE_CHECKING
from .base import Morphism

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


def bind_context_to_params(
    branch: "Branch",
    runtime_params: Dict[str, Any],
    bindings: Mapping[str, str],
    defaults: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build morphism parameters by binding Branch context.

    This is the core v1 pattern - morphism parameters come from:
    1. Branch context via bindings
    2. Default values
    3. Runtime overrides

    Args:
        branch: Branch providing context
        runtime_params: Runtime parameter overrides
        bindings: Map of param_name -> context_path
        defaults: Default parameter values

    Returns:
        Combined parameters for morphism invocation
    """
    # Start with context bindings
    bound_params: Dict[str, Any] = {}

    # 1. Bind from context using binding paths
    for param, ctx_path in bindings.items():
        value = _get_from_context(branch, ctx_path)
        if value is not None:
            bound_params[param] = value

    # 2. Apply defaults for missing params
    for key, value in defaults.items():
        bound_params.setdefault(key, value)

    # 3. Runtime params override everything
    bound_params.update(runtime_params)

    return bound_params


def _get_from_context(branch: "Branch", path: str) -> Any:
    """Extract value from branch context using dot notation path.

    Args:
        branch: Branch containing context
        path: Dot notation path (e.g., "form.validation_level")

    Returns:
        Value at path or None if not found
    """
    # Branch may have ctx attribute (to be added)
    if not hasattr(branch, 'ctx'):
        return None

    parts = path.split(".")
    value = branch.ctx

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return None

    return value


class BoundMorphism(Morphism):
    """Morphism with context binding - wraps another morphism.

    This is the v1 BoundOp pattern adapted for current lionagi.
    Before invoking the inner morphism, it binds parameters
    from Branch context.
    """

    def __init__(
        self,
        inner: Morphism,
        bindings: Mapping[str, str] = None,
        defaults: Mapping[str, Any] = None
    ):
        """Initialize bound morphism.

        Args:
            inner: The morphism to wrap
            bindings: Parameter bindings (param -> context_path)
            defaults: Default parameter values
        """
        self.inner = inner
        self.bindings = dict(bindings or {})
        self.defaults = dict(defaults or {})

        # Inherit properties from inner
        self.meta = inner.meta
        self.requires = inner.requires
        self.io = inner.io

    async def pre(self, branch: "Branch", /, **kw) -> bool:
        """Pre-condition with bound parameters."""
        bound_params = bind_context_to_params(
            branch, kw, self.bindings, self.defaults
        )
        return await self.inner.pre(branch, **bound_params)

    async def _invoke(self, branch: "Branch", /, **kw) -> Dict[str, Any]:
        """Invoke with bound parameters."""
        bound_params = bind_context_to_params(
            branch, kw, self.bindings, self.defaults
        )
        return await self.inner._invoke(branch, **bound_params)

    async def post(self, branch: "Branch", /, result: dict) -> bool:
        """Post-condition - no parameter binding needed."""
        return await self.inner.post(branch, result)


def create_bound_morphism(
    morphism: Morphism,
    bindings: Dict[str, str],
    defaults: Dict[str, Any] = None
) -> BoundMorphism:
    """Factory function to create a bound morphism.

    Example:
        # Bind validation strictness from branch context
        bound = create_bound_morphism(
            ValidationMorphism(),
            bindings={"strict_mode": "form.validation_level"},
            defaults={"strict_mode": False}
        )
    """
    return BoundMorphism(morphism, bindings, defaults or {})