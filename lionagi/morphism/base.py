"""Base morphism implementation - core operation protocol.

Migrated from operations/morph.py and enhanced with v1 patterns.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypedDict, Set, Any, Dict

from lionagi.ln.types import DataClass, Params
from lionagi.protocols._concepts import Invariant

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


__all__ = (
    "Morphism",
    "MorphMeta",
    "SystemMorphism",
)


class MorphMeta(TypedDict, total=False):
    """Metadata for morphism identification and documentation."""
    name: str
    description: str
    version: str
    priority: int  # For system morphisms ordering


@dataclass(slots=True, frozen=True, init=False)
class Morphism(Invariant):
    """Base morphism - smallest composable operation unit.

    Enhanced with v1 patterns:
    - Explicit capability requirements (requires)
    - IO flag for operations that do external I/O
    - Result shape expectations
    """

    ctx_cls: ClassVar[type[DataClass]]
    """The context class for this morphism."""

    # Core fields from current implementation
    meta: MorphMeta
    params: Params
    ctx: DataClass

    # Enhanced fields from v1
    requires: Set[str] = frozenset()  # Required capabilities
    io: bool = False  # Does this morphism perform I/O?

    @property
    def name(self) -> str:
        return self.meta.get("name", self.__class__.__name__)

    @property
    def request(self) -> dict:
        """Combined params and context for invocation."""
        _dict = self.params.to_dict() if self.params else {}
        _dict.update(self.ctx.to_dict() if self.ctx else {})
        return _dict

    async def invoke(self, branch: "Branch", /) -> dict:
        """Main invocation - calls _invoke with request parameters."""
        return await self._invoke(branch, **self.request)

    @abstractmethod
    async def _invoke(self, branch: "Branch", /, **kw) -> Dict[str, Any]:
        """Override in subclass - actual operation logic."""
        pass

    async def pre(self, branch: "Branch", /, **kw) -> bool:
        """Pre-condition check before operation.

        Returns:
            bool: True if operation should proceed
        """
        return True

    async def post(self, branch: "Branch", /, result: dict) -> bool:
        """Post-condition check after operation.

        Args:
            branch: Current branch context
            result: Operation result

        Returns:
            bool: True if result is valid
        """
        return True

    # V1 pattern: Dynamic capability requirements
    def required_rights(self, **kwargs) -> Set[str]:
        """Compute dynamic capability requirements based on parameters.

        This allows morphisms to have context-dependent requirements.
        Override in subclasses for dynamic behavior.
        """
        return self.requires


class SystemMorphism(Morphism):
    """System-level morphism that runs for all operations.

    Examples:
    - ValidationMorphism: Validates inputs/outputs
    - LoggingMorphism: Logs all operations
    - RateLimitMorphism: Enforces rate limits
    - AuditMorphism: Creates audit trail
    """

    # System morphisms have priority for ordering
    @property
    def priority(self) -> int:
        """Priority for system morphism execution order.

        Lower numbers run first. Default is 100.
        Common priorities:
        - 0-19: Critical pre-processing (auth, rate limit)
        - 20-39: Validation and sanitization
        - 40-59: Enhancement and enrichment
        - 60-79: Business logic preparation
        - 80-99: Logging and metrics
        - 100+: Post-processing
        """
        return self.meta.get("priority", 100)

    async def should_run(self, branch: "Branch", target: Morphism) -> bool:
        """Determine if this system morphism should run for target.

        Override to implement conditional system morphisms.

        Args:
            branch: Current branch context
            target: The morphism being executed

        Returns:
            bool: True if this system morphism should run
        """
        return True