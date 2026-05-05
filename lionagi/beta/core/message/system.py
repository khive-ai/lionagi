from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.ln._utils import now_utc

from .role import Role, RoledContent


@dataclass(slots=True)
class System(RoledContent):
    """System message with optional timestamp."""

    role: ClassVar[Role] = Role.SYSTEM

    system_message: MaybeUnset[str] = Unset
    system_datetime: MaybeUnset[str | Literal[True]] = Unset
    datetime_factory: MaybeUnset[Callable[[], str]] = Unset

    def render(self, *_args, **_kwargs) -> str:
        parts: list[str] = []
        if not self._is_sentinel(self.system_datetime):
            timestamp = (
                now_utc().isoformat(timespec="seconds")
                if self.system_datetime is True
                else self.system_datetime
            )
            parts.append(f"System Time: {timestamp}")
        elif not self._is_sentinel(self.datetime_factory):
            factory = self.datetime_factory
            parts.append(f"System Time: {factory()}")

        if not self._is_sentinel(self.system_message):
            parts.append(self.system_message)

        return "\n\n".join(parts)

    @classmethod
    def create(
        cls,
        system_message: str | None = None,
        system_datetime: str | Literal[True] | None = None,
        datetime_factory: Callable[[], str] | None = None,
    ) -> "System":
        if not cls._is_sentinel(system_datetime) and not cls._is_sentinel(datetime_factory):
            raise ValueError("Cannot set both system_datetime and datetime_factory")
        return cls(
            system_message=Unset if system_message is None else system_message,
            system_datetime=Unset if system_datetime is None else system_datetime,
            datetime_factory=Unset if datetime_factory is None else datetime_factory,
        )
