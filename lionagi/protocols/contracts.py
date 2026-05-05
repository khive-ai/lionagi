# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Runtime-checkable protocols, @implements decorator, and structural interfaces.

Protocols define structural interfaces (duck typing) with isinstance() support.
Use @implements(Protocol) to declare and validate protocol implementations.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

__all__ = (
    "Allowable",
    "Communicatable",
    "Containable",
    "Deserializable",
    "Hashable",
    "Invocable",
    "LegacyObservable",
    "MorphismProtocol",
    "Observable",
    "ObservableProto",
    "Serializable",
    "SignatureMismatchError",
    "implements",
)


class SignatureMismatchError(TypeError):
    pass


@runtime_checkable
class ObservableProto(Protocol):
    """Has unique identity."""

    @property
    def id(self) -> object: ...


Observable = ObservableProto

from ._concepts import Observable as LegacyObservable  # noqa: E402


@runtime_checkable
class Serializable(Protocol):
    def to_dict(self, **kw: Any) -> dict[str, Any]: ...


@runtime_checkable
class Deserializable(Protocol):
    @classmethod
    def from_dict(cls, data: dict[str, Any], **kw: Any) -> Any: ...


@runtime_checkable
class Containable(Protocol):
    def __contains__(self, item: Any) -> bool: ...


@runtime_checkable
class Invocable(Protocol):
    async def invoke(self) -> Any: ...


@runtime_checkable
class Hashable(Protocol):
    def __hash__(self) -> int: ...


@runtime_checkable
class Allowable(Protocol):
    def allowed(self) -> set[str]: ...


@runtime_checkable
class Communicatable(Protocol):
    @property
    def id(self) -> UUID: ...

    @property
    def mailbox(self) -> Any: ...


@runtime_checkable
class MorphismProtocol(Protocol):
    """Atomic unit of execution with capability requirements."""

    @property
    def name(self) -> str: ...

    @property
    def requires(self) -> frozenset[str]: ...

    @property
    def provides(self) -> frozenset[str]: ...

    async def pre(self, ctx: Any) -> bool: ...
    async def apply(self, ctx: Any) -> dict[str, Any]: ...
    async def post(self, ctx: Any, result: dict[str, Any]) -> bool: ...


def _get_signature_params(func: Any) -> dict[str, inspect.Parameter] | None:
    if isinstance(func, (classmethod, staticmethod)):
        func = func.__func__
    if isinstance(func, property):
        return None
    if not callable(func):
        return None
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    return {
        name: param
        for name, param in sig.parameters.items()
        if name not in ("self", "cls")
    }


def _check_signature_compatibility(
    protocol_params: dict[str, inspect.Parameter],
    impl_params: dict[str, inspect.Parameter],
) -> list[str]:
    errors = []
    impl_has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in impl_params.values()
    )
    proto_has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in protocol_params.values()
    )

    if proto_has_var_keyword and not impl_has_var_keyword:
        errors.append("  - protocol accepts **kw but implementation doesn't")

    for param_name, proto_param in protocol_params.items():
        if proto_param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in impl_params:
            impl_param = impl_params[param_name]
            if impl_param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if (
                proto_param.default is not inspect.Parameter.empty
                and impl_param.default is inspect.Parameter.empty
            ):
                errors.append(
                    f"  - '{param_name}': protocol optional, implementation required"
                )
        elif not impl_has_var_keyword:
            errors.append(f"  - '{param_name}': required by protocol but missing")

    return errors


def implements(
    *protocols: type,
    signature_check: Literal["error", "warn", "skip"] = "warn",
    allow_inherited: bool = False,
    capabilities: frozenset[str] | set[str] | None = None,
):
    """Declare and validate protocol implementations."""

    def decorator(cls):
        all_signature_errors = []

        for protocol in protocols:
            protocol_members = {}
            for name, obj in inspect.getmembers(protocol):
                if name.startswith("_"):
                    continue
                if callable(obj) or isinstance(obj, (property, classmethod)):
                    protocol_members[name] = obj

            for member_name, protocol_member in protocol_members.items():
                in_class_body = member_name in cls.__dict__
                if not in_class_body and hasattr(cls, "__annotations__"):
                    in_class_body = member_name in cls.__annotations__

                has_member = hasattr(cls, member_name)

                if allow_inherited:
                    if not has_member:
                        raise TypeError(
                            f"{cls.__name__} declares @implements({protocol.__name__}) "
                            f"but '{member_name}' is not defined or inherited"
                        )
                else:
                    if not in_class_body:
                        raise TypeError(
                            f"{cls.__name__} declares @implements({protocol.__name__}) "
                            f"but does not define '{member_name}' in its class body. "
                            f"Use allow_inherited=True to accept inherited implementations."
                        )

                if signature_check != "skip":
                    impl_member = (
                        cls.__dict__.get(member_name)
                        if in_class_body
                        else getattr(cls, member_name, None)
                    )
                    if impl_member is None and hasattr(cls, "__annotations__"):
                        continue
                    proto_params = _get_signature_params(protocol_member)
                    impl_params = _get_signature_params(impl_member)
                    if proto_params is not None and impl_params is not None:
                        errors = _check_signature_compatibility(
                            proto_params, impl_params
                        )
                        if errors:
                            all_signature_errors.append(
                                f"{cls.__name__}.{member_name} incompatible with "
                                f"{protocol.__name__}.{member_name}:\n"
                                + "\n".join(errors)
                            )

        if all_signature_errors:
            full_message = "\n\n".join(all_signature_errors)
            if signature_check == "error":
                raise SignatureMismatchError(full_message)
            elif signature_check == "warn":
                warnings.warn(full_message, stacklevel=2)

        cls.__protocols__ = protocols

        if capabilities is not None:
            cls.__capabilities__ = frozenset(capabilities)
        elif hasattr(cls, "requires") and hasattr(cls, "provides"):
            cls.__capabilities__ = frozenset(
                (cls.requires if isinstance(cls.requires, (set, frozenset)) else set())
                | (
                    cls.provides
                    if isinstance(cls.provides, (set, frozenset))
                    else set()
                )
            )

        return cls

    return decorator
