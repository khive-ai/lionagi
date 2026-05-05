from dataclasses import dataclass
from typing import Any, ClassVar

from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.libs.schema import minimal_yaml

from .role import Role, RoledContent


@dataclass(slots=True)
class ActionRequest(RoledContent):
    """Action/function call request."""

    role: ClassVar[Role] = Role.ACTION

    function: MaybeUnset[str] = Unset
    arguments: MaybeUnset[dict[str, Any]] = Unset

    def render(self, *_args, **_kwargs) -> str:
        doc: dict[str, Any] = {}
        if not self._is_sentinel(self.function):
            doc["function"] = self.function
        doc["arguments"] = {} if self._is_sentinel(self.arguments) else self.arguments
        return minimal_yaml(doc)

    def render_compact(self) -> str:
        """Function-call representation for round summaries.

        Args render in full — token budget is the caller's concern, not this
        layer's. Strings get repr() so quotes/escapes are explicit.
        """
        func = self.function if not self._is_sentinel(self.function) else "unknown"
        args = self.arguments if not self._is_sentinel(self.arguments) else {}
        parts = [f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in args.items()]
        return f"{func}({', '.join(parts)})"

    @classmethod
    def create(cls, function: str, arguments: dict[str, Any] = Unset):
        if cls._is_sentinel(arguments):
            arguments = {}
        return cls(function=function, arguments=arguments)


@dataclass(slots=True)
class ActionResponse(RoledContent):
    """Function call response."""

    role: ClassVar[Role] = Role.ACTION

    request_id: MaybeUnset[str] = Unset
    result: MaybeUnset[Any] = Unset
    error: MaybeUnset[str] = Unset

    def render(self, *_args, **_kwargs) -> str:
        doc: dict[str, Any] = {"success": self.success}
        if not self._is_sentinel(self.request_id):
            doc["request_id"] = str(self.request_id)[:8]
        if self.success:
            if not self._is_sentinel(self.result):
                doc["result"] = self.result
        else:
            doc["error"] = self.error
        return minimal_yaml(doc)

    def render_summary(self) -> str:
        """Render result content for round-level aggregation.

        Returns the actual result, not metadata. Models need to see the data
        a tool produced to reason about it in the next step. If you need a
        size-only view, call render() and inspect length yourself.
        """
        if not self.success:
            err = self.error if not self._is_sentinel(self.error) else "unknown"
            return f"error: {err}"
        if self._is_sentinel(self.result):
            return "ok"
        if isinstance(self.result, str):
            return self.result
        if isinstance(self.result, (dict, list)):
            return minimal_yaml(self.result)
        return str(self.result)

    @property
    def success(self) -> bool:
        return self._is_sentinel(self.error)

    @classmethod
    def create(
        cls,
        request_id: str | None = None,
        result: Any = Unset,
        error: str | None = None,
    ) -> "ActionResponse":
        return cls(
            request_id=Unset if request_id is None else request_id,
            result=result,
            error=Unset if error is None else error,
        )
