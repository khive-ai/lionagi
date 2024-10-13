from time import timezone

from lion_core.setting import LionIDConfig

DEFAULT_TIMEZONE = timezone.UTC

DEFAULT_LION_ID_CONFIG = LionIDConfig(
    n=42,
    random_hyphen=True,
    num_hyphens=4,
    hyphen_start_index=6,
    hyphen_end_index=-6,
    prefix="ln",
    postfix="",
)


__all__ = ["DEFAULT_TIMEZONE", "DEFAULT_LION_ID_CONFIG"]
