from __future__ import annotations

from lionagi.ln import DataClass
from lionagi.protocols._concepts import Manager
from lionagi.session.branch import Branch

from .morph import Morphism
from .node import Operation

_DEFAULT_MORPHISMS = None


def _get_morphisms():
    global _DEFAULT_MORPHISMS
    if _DEFAULT_MORPHISMS is not None:
        return _DEFAULT_MORPHISMS

    from .chat.morph import ChatMorphism
    from .communicate.morph import CommunicateMorphism
    from .interpret.morph import IntepretMorphism
    from .operate.morph import OperateMorphism
    from .ReAct.morph import ReActMorphism
    from .select.morph import SelectMorphism
    from .translate.morph import TranslateMorphism

    _DEFAULT_MORPHISMS = {
        "chat": ChatMorphism,
        "operate": OperateMorphism,
        "communicate": CommunicateMorphism,
        "ReAct": ReActMorphism,
        "interpret": IntepretMorphism,
        "translate": TranslateMorphism,
        "select": SelectMorphism,
        "ReActStream": ReActMorphism,  # Alias for backward compatibility
        "react": ReActMorphism,
        "reactstream": ReActMorphism,  # Alias for backward compatibility
    }
    return _DEFAULT_MORPHISMS


def _validate_morphism_type(morph_cls: type[Morphism]) -> None:
    if not isinstance(morph_cls, type) or not issubclass(morph_cls, Morphism):
        raise TypeError(f"Expected Morphism type, got {type(morph_cls)}")


def _validate_context_type(ctx_cls: type[DataClass]) -> None:
    if not isinstance(ctx_cls, type) or not issubclass(ctx_cls, DataClass):
        raise TypeError(f"Expected DataClass type, got {type(ctx_cls)}")


class OperationManager(Manager):
    def __init__(self, strict: bool = False):
        """
        If strict is True, the manager will raise error when invalid parameters are passed in when creating a morphism, else will ignore them.
        """
        super().__init__()
        self.registry: dict[str, type[Morphism]] = _get_morphisms()
        self.strict = strict

    def create_operation(self, name: str, /, **kw) -> Operation:
        if (morph_cls := self.registry.get(name)) is None:
            raise ValueError(f"Operation {name} not found in registry.")

        ctx_cls = morph_cls.ctx_cls
        _ctx = (
            ctx_cls(**{k: v for k, v in kw.items() if k in ctx_cls.allowed()})
            if not self.strict
            else ctx_cls(**kw)
        )
        morph = morph_cls(ctx=_ctx)
        return Operation(
            operation=morph.name,
            morph=morph,
            metadata={
                "morphism": morph.meta,
            },
        )

    def register_morph(
        self, morph_cls: type[Morphism], /, update: bool = False
    ) -> None:
        if morph_cls.name in self.registry:
            if not update:
                raise ValueError(
                    f"Operation {morph_cls.name} already registered."
                )

        _validate_morphism_type(morph_cls)

        ctx_cls: type[DataClass] = getattr(morph_cls, "ctx_cls", None)
        _validate_context_type(ctx_cls)

        self.registry[morph_cls.name] = morph_cls
