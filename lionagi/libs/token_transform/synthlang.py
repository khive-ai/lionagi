import warnings

warnings.warn(
    "The module 'lionagi.libs.token_transform.synthlang' is deprecated and will be removed in a future version. "
    "Use 'lionagi.core.text_processing.synthlang' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .synthlang_.base import SynthlangFramework, SynthlangTemplate
from .synthlang_.translate_to_synthlang import translate_to_synthlang

# backwards compatibility
__all__ = (
    "translate_to_synthlang",
    "SynthlangFramework",
    "SynthlangTemplate",
)
