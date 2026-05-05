"""Surgical gap-fill tests for provider_config.py missing branches.

Targets ~8 missing statements in lionagi/service/connections/provider_config.py:
Lines: 58 (LazyType bad ref), 70 (LazyType.__hash__), 73-75 (LazyType.__eq__),
       112 (options raw non-LazyType), 162 (register decorator-only return),
       166 (available classmethod)
"""

from enum import Enum

import pytest

from lionagi.service.connections.provider_config import LazyType, ProviderConfig
from lionagi.service.connections.registry import EndpointType

# ---------------------------------------------------------------------------
# LazyType — basic construction and validation
# ---------------------------------------------------------------------------


def test_lazy_type_invalid_ref_raises():
    """Line 58: LazyType without ':' raises ValueError."""
    with pytest.raises(ValueError, match="LazyType ref must be 'module:Class'"):
        LazyType("no_colon_here")


def test_lazy_type_valid_ref():
    """Baseline: valid ref is accepted."""
    lt = LazyType("pydantic:BaseModel")
    assert lt._ref == "pydantic:BaseModel"
    assert lt._resolved is None


def test_lazy_type_hash():
    """Line 70: __hash__ returns hash of the ref string."""
    lt = LazyType("pydantic:BaseModel")
    assert hash(lt) == hash("pydantic:BaseModel")


def test_lazy_type_eq_same_ref():
    """Lines 73-74: __eq__ compares refs."""
    lt1 = LazyType("pydantic:BaseModel")
    lt2 = LazyType("pydantic:BaseModel")
    assert lt1 == lt2


def test_lazy_type_eq_different_ref():
    """Line 74: different refs are not equal."""
    lt1 = LazyType("pydantic:BaseModel")
    lt2 = LazyType("pydantic:Field")
    assert lt1 != lt2


def test_lazy_type_eq_non_lazy_type():
    """Line 75: comparing with non-LazyType returns NotImplemented."""
    lt = LazyType("pydantic:BaseModel")
    result = lt.__eq__("string")
    assert result is NotImplemented


def test_lazy_type_eq_used_via_operator():
    """Line 73-75: != with non-LazyType doesn't raise."""
    lt = LazyType("pydantic:BaseModel")
    # Python will call __ne__ which calls __eq__ returning NotImplemented,
    # then falls back to identity check — should return True (not the same obj)
    assert lt != "some string"


def test_lazy_type_repr_pending():
    """Repr shows 'pending' when not yet resolved."""
    lt = LazyType("pydantic:BaseModel")
    r = repr(lt)
    assert "pending" in r
    assert "pydantic:BaseModel" in r


def test_lazy_type_repr_resolved():
    """Repr shows 'resolved' after resolve() is called."""
    lt = LazyType("pydantic:BaseModel")
    lt.resolve()
    r = repr(lt)
    assert "resolved" in r


def test_lazy_type_resolve_caches():
    """resolve() caches result."""
    from pydantic import BaseModel as PydanticBaseModel

    lt = LazyType("pydantic:BaseModel")
    result1 = lt.resolve()
    result2 = lt.resolve()
    assert result1 is result2
    assert result1 is PydanticBaseModel


# ---------------------------------------------------------------------------
# ProviderConfig — build minimal concrete enums
# Use stdlib Enum (not lionagi.ln.types.Enum) + set class attrs after definition
# as real provider configs do (see providers/openrouter/_config.py)
# ---------------------------------------------------------------------------


class _TestProviderConfig(ProviderConfig, Enum):
    CHAT = (
        "chat/completions",
        ["chat"],
        EndpointType.API,
        LazyType("pydantic:BaseModel"),
        "https://api.test.com/v1",
        "bearer",
        "application/json",
    )

    # Short form — raw (non-LazyType) options value, no HTTP config
    SHORT = (
        "short_endpoint",
        ["short"],
        EndpointType.AGENTIC,
        None,  # no options (non-LazyType path)
    )


_TestProviderConfig._PROVIDER = "testprovider"
_TestProviderConfig._PROVIDER_ALIASES = ["tp"]


def test_provider_config_options_lazy_type_resolves():
    """Line 111: options with LazyType → resolve() is called."""
    from pydantic import BaseModel as PydanticBaseModel

    opts = _TestProviderConfig.CHAT.options
    assert opts is PydanticBaseModel


def test_provider_config_options_non_lazy_type_returned_directly():
    """Line 112: options with non-LazyType raw value returns as-is."""
    opts = _TestProviderConfig.SHORT.options
    assert opts is None


def test_provider_config_available():
    """Line 166: available() returns frozenset of endpoint paths."""
    avail = _TestProviderConfig.available()
    assert isinstance(avail, frozenset)
    assert "chat/completions" in avail
    assert "short_endpoint" in avail


def test_provider_config_register_returns_decorator():
    """Line 162: register(cls=None) returns the decorator itself."""
    config_member = _TestProviderConfig.CHAT
    decorator = config_member.register()  # cls=None → returns decorator callable
    assert callable(decorator)


def test_provider_config_as_registry_kwargs():
    """as_registry_kwargs returns expected keys."""
    kwargs = _TestProviderConfig.CHAT.as_registry_kwargs()
    assert kwargs["provider"] == "testprovider"
    assert kwargs["endpoint"] == "chat/completions"
    assert kwargs["base_url"] == "https://api.test.com/v1"
    assert kwargs["auth_type"] == "bearer"
    assert kwargs["content_type"] == "application/json"


def test_provider_config_provider_aliases():
    """provider_aliases property returns list."""
    aliases = _TestProviderConfig.CHAT.provider_aliases
    assert isinstance(aliases, list)
    assert "tp" in aliases


def test_provider_config_endpoint_path():
    """endpoint_path returns first element."""
    assert _TestProviderConfig.CHAT.endpoint_path == "chat/completions"


def test_provider_config_aliases():
    """aliases returns second element."""
    assert _TestProviderConfig.CHAT.aliases == ["chat"]


def test_provider_config_base_url_none_for_short_form():
    """base_url is None for short form without index 4."""
    assert _TestProviderConfig.SHORT.base_url is None


def test_provider_config_content_type_default():
    """content_type defaults to 'application/json' for short form."""
    ct = _TestProviderConfig.SHORT.content_type
    assert ct == "application/json"


def test_provider_config_register_with_cls():
    """Line 161: register(cls=SomeClass) directly registers and returns the class."""

    class _DummyEndpoint:
        pass

    config_member = _TestProviderConfig.SHORT
    result = config_member.register(cls=_DummyEndpoint)
    # register_endpoint injects _ENDPOINT_META and returns the class
    assert result is _DummyEndpoint
    assert hasattr(_DummyEndpoint, "_ENDPOINT_META")
