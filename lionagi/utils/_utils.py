import importlib.util
from pathlib import Path
from typing import Literal

from lionagi._errors import NotFoundError

from ._models import Params


def is_import_installed(package_name: str) -> bool:
    """
    Check if a package is installed.

    Args:
        package_name: The name of the package to check.
    """
    return importlib.util.find_spec(package_name) is not None


_HAS_XMLTODICT = is_import_installed("xmltodict")

AVAILABLE_PARAMS = Literal["alcall", "bcall", "lcall", "to_list"]


class LU:
    """LionUtils, a stateless utility class for LionAI."""

    @staticmethod
    def get_params(func_name: AVAILABLE_PARAMS, /) -> type[Params]:
        match func_name:
            case "alcall":
                from ._async_call import AlcallParams

                return AlcallParams
            case "bcall":
                from ._async_call import BcallParams

                return BcallParams
            case "lcall":
                from ._list_call import LcallParams

                return LcallParams
            case "to_list":
                from ._to_list import ToListParams

                return ToListParams
            case _:
                raise NotFoundError.from_value(
                    func_name,
                    expected="'alcall', 'bcall', 'lcall', or 'to_list'",
                    message=f"Function '{func_name}' is not available in LionUtils.",
                )

    @staticmethod
    def to_list(input_, /, param_=None, **kw):
        if param_ is not None:
            from ._to_list import ToListParams

            if isinstance(param_, ToListParams):
                return param_(input_, **kw)
            raise TypeError(
                f"Expected LcallParams instance, got {param_.__class__.__name__}"
            )
        from ._to_list import to_list

        return to_list(input_, **kw)

    @staticmethod
    def lcall(input_, func, /, *args, param_=None, **kwargs):
        if param_ is not None:
            from ._list_call import LcallParams

            if isinstance(param_, LcallParams):
                return param_(input_, func, *args, **kwargs)
            raise TypeError(
                f"Expected LcallParams instance, got {param_.__class__.__name__}"
            )
        from ._list_call import lcall

        return lcall(input_, func, *args, **kwargs)

    @staticmethod
    async def alcall(input_, func, /, *args, param_=None, **kwargs):
        if param_ is not None:
            from ._async_call import AlcallParams

            if isinstance(param_, AlcallParams):
                return param_(input_, func, *args, **kwargs)
            raise TypeError(
                f"Expected AlcallParams instance, got {param_.__class__.__name__}"
            )
        from ._async_call import alcall

        return alcall(input_, func, *args, **kwargs)

    @staticmethod
    async def bcall(input_, func, /, *args, param_=None, **kwargs):
        if param_ is not None:
            from ._async_call import BcallParams

            if isinstance(param_, BcallParams):
                return param_(input_, func, *args, **kwargs)
            raise TypeError(
                f"Expected BcallParams instance, got {param_.__class__.__name__}"
            )
        from ._async_call import bcall

        return bcall(input_, func, *args, **kwargs)

    @staticmethod
    def create_path(
        directory: Path | str,
        filename: str,
        *args,
        param_=None,
        **kwargs,
    ):
        if param_ is not None:
            from ._create_path import CreatePathParams

            if isinstance(param_, CreatePathParams):
                return param_(directory, filename)
            raise TypeError(
                f"Expected CreatePathParams instance, got {param_.__class__.__name__}"
            )
        from ._create_path import create_path

        return create_path(directory, filename, *args, **kwargs)

    @staticmethod
    def fuzzy_parse(s_: str, /):
        """Parse a string to extract a float, int, or bool value."""
        from ._fuzzy_parse import fuzzy_parse_json

        return fuzzy_parse_json(s_)

    @staticmethod
    def hash_dict(d: dict) -> int:
        """Hash a dictionary, handling unhashable types."""
        from ._hash import hash_dict

        return hash_dict(d)

    @staticmethod
    def xml_to_dict(xml_str: str, /, **kw):
        """Convert an XML string to a dictionary."""
        if not _HAS_XMLTODICT:
            raise ImportError(
                'xmltodict is not installed. Please install it with `uv add "lionagi[xml]"`'
            )
        import xmltodict

        return xmltodict.parse(xml_str, **kw)

    @staticmethod
    def dict_to_xml(d: dict, /, **kw):
        """Convert a dictionary to an XML string."""
        if not _HAS_XMLTODICT:
            raise ImportError(
                'xmltodict is not installed. Please install it with `uv add "lionagi[xml]"`'
            )
        import xmltodict

        kw["pretty"] = kw.get("pretty", True)
        kw["full_document"] = kw.get("full_document", False)
        return xmltodict.unparse(d, **kw)
