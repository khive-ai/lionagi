import sys
from collections.abc import Sequence
from enum import Enum
from typing import Annotated

__all__ = (
    "BaseExceptionGroup",
    "ExceptionGroup",
    "Self",
    "StrEnum",
    "get_annotated_type",
    "get_exception_group_exceptions",
    "is_exception_group",
)

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup, ExceptionGroup
    from enum import StrEnum
    from typing import Self
else:
    from typing_extensions import Self

    class StrEnum(str, Enum):
        pass

    try:
        from exceptiongroup import BaseExceptionGroup, ExceptionGroup
    except ImportError:

        class BaseExceptionGroup(BaseException):  # type: ignore
            def __init__(
                self, message: str, exceptions: Sequence[BaseException]
            ) -> None:
                super().__init__(message)
                self.message = message
                self.exceptions = tuple(exceptions)

            def __str__(self) -> str:
                return f"{self.message} ({len(self.exceptions)} sub-exceptions)"

            def split(
                self,
                condition: type | tuple[type, ...],
            ) -> tuple["BaseExceptionGroup | None", "BaseExceptionGroup | None"]:
                match, rest = [], []
                for exc in self.exceptions:
                    (match if isinstance(exc, condition) else rest).append(exc)
                return (
                    type(self)(self.message, match) if match else None,
                    type(self)(self.message, rest) if rest else None,
                )

        class ExceptionGroup(BaseExceptionGroup, Exception):  # type: ignore
            pass


def is_exception_group(exc: BaseException) -> bool:
    return isinstance(exc, BaseExceptionGroup)


def get_exception_group_exceptions(
    exc: BaseException,
) -> Sequence[BaseException]:
    if is_exception_group(exc):
        return getattr(exc, "exceptions", (exc,))
    return (exc,)


def get_annotated_type(args: tuple) -> type:
    try:
        return Annotated.__class_getitem__(tuple(args))  # type: ignore
    except AttributeError:
        import operator

        return operator.getitem(Annotated, tuple(args))  # type: ignore
