"""Identifiable - clarity-first mixin that adds immutable metadata and
optional integrity hashing to any - model.

Public API
~~~~~~~~~~
class MyDomainObject(Identifiable):
    foo: int

obj  = MyDomainObject(foo=42)
blob = obj.prepare_for_storage()          # -> copy carrying fresh hash
assert blob.verify_integrity()            # True
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .integrity import Algo, digest


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #
class Metadata(BaseModel):
    """Creation facts + optional integrity tag (algorithm + hex digest)."""

    model_config = ConfigDict(slots=True, frozen=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    lion_class: str
    algo: str | None = None
    content_digest: str | None = None

    @property
    def view(self) -> MappingProxyType:
        """Immutable mapping (helpful for logs / str())."""
        return MappingProxyType(self.model_dump())


# --------------------------------------------------------------------------- #
# Identifiable                                                                #
# --------------------------------------------------------------------------- #
class Identifiable(BaseModel):
    """Mixin that injects :class:`Metadata` and *optional* integrity helpers.

    * No hidden side-effects during normal validation / serialisation.
    * Hash is computed **only** when you call :py:meth:`prepare_for_storage`
      or :py:meth:`verify_integrity`.
    """

    model_config = ConfigDict(extra="forbid", slots=True)
    metadata: Metadata | None = None

    # private cache for deterministic dumps (excludes metadata)
    _cached_dump: dict[str, Any] | None = PrivateAttr(default=None)

    # --------------------------------------------------------------------- #
    # pydantic lifecycle                                                    #
    # --------------------------------------------------------------------- #
    def model_post_init(self, __ctx) -> None:
        if self.metadata is None:
            self.metadata = Metadata(lion_class=self.__class__.__qualname__)

    # --------------------------------------------------------------------- #
    # Integrity helpers                                                     #
    # --------------------------------------------------------------------- #
    def _content_dump(self) -> dict[str, Any]:
        if self._cached_dump is None:
            self._cached_dump = self.model_dump(exclude={"metadata"})
        return self._cached_dump

    def compute_digest(self, *, algo: Algo = Algo.SHA256) -> str:
        """Return hex digest of the *content* (excluding metadata)."""
        return digest(self._content_dump(), algo=algo)

    def prepare_for_storage(
        self, *, algo: Algo = Algo.SHA256
    ) -> "Identifiable":
        """Return a **copy** whose metadata carries a fresh digest."""
        new_meta = self.metadata.model_copy(
            update={
                "algo": algo.name,
                "content_digest": self.compute_digest(algo=algo),
            }
        )
        return self.model_copy(update={"metadata": new_meta})

    def verify_integrity(self) -> bool:
        """Compare stored digest (if present) with freshly computed one."""
        if not (self.metadata.content_digest and self.metadata.algo):
            raise ValueError("Object has no integrity tag to verify")
        algo = Algo[self.metadata.algo]
        return self.compute_digest(algo=algo) == self.metadata.content_digest

    # --------------------------------------------------------------------- #
    # Convenience shims                                                     #
    # --------------------------------------------------------------------- #
    @property
    def id(self) -> UUID:
        return self.metadata.id

    # hash/equality semantics
    def __hash__(self) -> int:
        return hash(self.metadata.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Identifiable) and self.id == other.id
