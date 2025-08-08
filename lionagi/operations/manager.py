from collections.abc import Callable

from lionagi.protocols._concepts import Manager
from lionagi.models import Note

from .morphism import Morphism, MorphismContext

"""
experimental
"""





class OperationManager(Manager):

    def __init__(self):
        super().__init__()
        self._registry: Note = Note()

    def register_morphism(self, morphism: Morphism) -> None:
        if not isinstance(morphism, Morphism):
            raise TypeError(f"{morphism} is not a valid Morphism.")
        if morphism.meta.name in self._registry:
            raise ValueError(f"Morphism {morphism.meta.name} is already registered.")

        # note["name", "obj"] = morphism
        self._registry[morphism.meta.name, "obj"] = morphism
    
    def _get_morphism(self, name: str, ) -> type[Morphism] | None:
        return self._registry.get([name, "obj"], None)

    def create_operation(
        self,
        morphism_name: str,
        branch,
        params: dict | None = None,
        stream_morphism: bool = False,
    ):
        morphism = self._get_morphism(morphism_name)
        if not morphism:
            raise ValueError(f"Morphism {morphism_name} is not registered.")
        from .node import Operation
        return Operation.create(
            morphism=morphism,
            branch=branch,
            params=params,
            stream_morphism=stream_morphism,
        )
