# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Runtime-checkable protocols, @implements decorator, and Morphism protocol.

Protocols define structural interfaces (duck typing) with isinstance() support.
Use @implements(Protocol) to declare and validate protocol implementations.
Morphism is the atomic unit of execution with explicit capability requirements.
"""

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
    "MorphismProtocol",
    "Observable",
    "Serializable",
    "SignatureMismatchError",
    "implements",
)


class SignatureMismatchError(TypeError):
    """@implements detected incompatible method signature."""

    pass


@runtime_checkable
class Observable(Protocol):
    """Has unique UUID identity."""

    @property
    def id(self) -> UUID: ...


@runtime_checkable
class Serializable(Protocol):
    """Can serialize to dict."""

    def to_dict(self, **kw: Any) -> dict[str, Any]: ...


@runtime_checkable
class Deserializable(Protocol):
    """Can deserialize from dict."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kw: Any) -> Any: ...


@runtime_checkable
class Containable(Protocol):
    """Supports 'in' operator."""

    def __contains__(self, item: Any) -> bool: ...


@runtime_checkable
class Invocable(Protocol):
    """Async executable."""

    async def invoke(self) -> Any: ...


@runtime_checkable
class Hashable(Protocol):
    """Hashable for sets/dicts."""

    def __hash__(self) -> int: ...


@runtime_checkable
class Allowable(Protocol):
    """Has defined allowed values."""

    def allowed(self) -> set[str]: ...


@runtime_checkable
class Communicatable(Protocol):
    """Entity with mailbox for message exchange."""

    @property
    def id(self) -> UUID: ...

    @property
    def mailbox(self) -> Any: ...


@runtime_checkable
class MorphismProtocol(Protocol):
    """Atomic unit of execution with explicit capability requirements.

    A Morphism declares what capabilities it requires to run and what
    capabilities it provides upon completion. The pre/apply/post lifecycle
    enables validation before execution, the execution itself, and
    verification of results.

    Capability strings follow the format: "domain.action[:resource]"
    Examples: "fs.read:/data/*", "net.out", "llm.call", "tool.execute:search"

    The capability algebra:
        - Union: parallel morphisms pool their provides
        - Subset: M.requires ⊆ available → M can execute
        - Satisfiability: OpGraph is valid iff every node's requires
          are satisfied by predecessors' provides ∪ ambient capabilities
    """

    @property
    def name(self) -> str: ...

    @property
    def requires(self) -> frozenset[str]:
        """Capabilities this morphism needs to execute."""
        ...

    @property
    def provides(self) -> frozenset[str]:
        """Capabilities this morphism makes available after execution."""
        ...

    async def pre(self, ctx: Any) -> bool:
        """Validate preconditions. Return False to abort."""
        ...

    async def apply(self, ctx: Any) -> dict[str, Any]:
        """Execute the operation. Returns result dict."""
        ...

    async def post(self, ctx: Any, result: dict[str, Any]) -> bool:
        """Validate postconditions on result. Return False to reject."""
        ...


def _get_signature_params(func: Any) -> dict[str, inspect.Parameter] | None:
    """Extract params from callable, excluding self/cls."""
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
    """Check impl signature compatibility with protocol. Returns error messages."""
    errors = []

    impl_has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in impl_params.values()
    )
    impl_has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in impl_params.values()
    )
    proto_has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in protocol_params.values()
    )
    proto_has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in protocol_params.values()
    )

    if proto_has_var_keyword and not impl_has_var_keyword:
        errors.append("  - 'kw': protocol accepts **kw but implementation doesn't")

    if proto_has_var_positional and not impl_has_var_positional:
        errors.append("  - 'args': protocol accepts *args but implementation doesn't")

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

            if proto_param.default is inspect.Parameter.empty:
                if (
                    impl_param.kind == inspect.Parameter.KEYWORD_ONLY
                    and proto_param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ):
                    errors.append(
                        f"  - '{param_name}': protocol allows positional, "
                        f"but implementation requires keyword-only"
                    )
            else:
                if impl_param.default is inspect.Parameter.empty:
                    errors.append(
                        f"  - '{param_name}': protocol makes this optional, "
                        f"but implementation requires it"
                    )
        else:
            proto_kind = proto_param.kind

            if proto_kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if not (impl_has_var_positional or impl_has_var_keyword):
                    errors.append(
                        f"  - '{param_name}': required by protocol "
                        f"but not in implementation"
                    )
            elif (
                proto_kind == inspect.Parameter.KEYWORD_ONLY
                and not impl_has_var_keyword
            ):
                errors.append(
                    f"  - '{param_name}': keyword-only param required by "
                    f"protocol but not in implementation"
                )

    for param_name, impl_param in impl_params.items():
        if impl_param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if (
            impl_param.default is inspect.Parameter.empty
            and param_name not in protocol_params
        ):
            can_satisfy = False
            if impl_param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                can_satisfy = proto_has_var_positional or proto_has_var_keyword
            elif impl_param.kind == inspect.Parameter.KEYWORD_ONLY:
                can_satisfy = proto_has_var_keyword

            if not can_satisfy:
                errors.append(
                    f"  - '{param_name}': implementation requires this param "
                    f"but protocol doesn't provide it"
                )

    return errors


def implements(
    *protocols: type,
    signature_check: Literal["error", "warn", "skip"] = "warn",
    allow_inherited: bool = False,
    capabilities: frozenset[str] | set[str] | None = None,
):
    """Declare and validate protocol implementations.

    Validates members exist and optionally checks signature compatibility.
    Stores validated protocols in cls.__protocols__ and declared capabilities
    in cls.__capabilities__.

    Args:
        *protocols: Protocol classes to implement.
        signature_check: "error"=raise, "warn"=warning, "skip"=no check.
        allow_inherited: Accept inherited implementations.
        capabilities: Declared capability strings for this implementation.
            Used by OpGraph for compile-time satisfiability checking.

    Example:
        @implements(Morphism, capabilities={"fs.read:/data/*", "net.out"})
        class MyOp:
            name = "my_op"
            requires = frozenset({"fs.read:/data/*"})
            provides = frozenset({"data.parsed"})

            async def pre(self, ctx): return True
            async def apply(self, ctx): return {"result": "done"}
            async def post(self, ctx, result): return True
    """

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
                            f"{cls.__name__} declares @implements("
                            f"{protocol.__name__}) but '{member_name}' "
                            f"is not defined or inherited"
                        )
                else:
                    if not in_class_body:
                        raise TypeError(
                            f"{cls.__name__} declares @implements("
                            f"{protocol.__name__}) but does not define "
                            f"'{member_name}' in its class body. "
                            f"Use allow_inherited=True to accept "
                            f"inherited implementations."
                        )

                if signature_check != "skip":
                    if in_class_body:
                        impl_member = cls.__dict__.get(member_name)
                    else:
                        impl_member = getattr(cls, member_name, None)

                    if impl_member is None and hasattr(cls, "__annotations__"):
                        continue

                    proto_params = _get_signature_params(protocol_member)
                    impl_params = _get_signature_params(impl_member)

                    if proto_params is not None and impl_params is not None:
                        errors = _check_signature_compatibility(
                            proto_params, impl_params
                        )
                        if errors:
                            error_msg = (
                                f"{cls.__name__}.{member_name} signature "
                                f"incompatible with "
                                f"{protocol.__name__}.{member_name}:\n"
                                + "\n".join(errors)
                            )
                            all_signature_errors.append(error_msg)

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
