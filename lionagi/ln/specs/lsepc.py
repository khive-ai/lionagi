from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from ..types import Meta


class LSpec:

    base_type: type[Any]
    metadata: dict[str, Any]


@dataclass(init=False, slots=True, frozen=True)
class Spec:

    metadata: tuple[Meta, ...]
    base_type: type[Any]
    sha256: str = None  # only when new obj gets created

    _unique_meta: ClassVar[bool] = True  # metadata keys must be unique

    def __init__(self, base_type: type, **kw):
        metas = self._convert_kw_to_meta(**kw)
        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", metas)

        self._validate_meta()

    def with_updates(self, **kw):
        # create a new instances with updated metadata
        ...

    def meta_dict(self) -> dict[str, Any]:
        """Convert metadata tuple to dict for easy access."""
        return {m.key: m.value for m in self.metadata}

    def _validate_meta(self):
        # validate metadata keys and values
        # add a sha256
        ...

    def __hash__(self):
        # deterministic hash based on base_type and metadata
        ...

    def __eq__(self, other): ...
