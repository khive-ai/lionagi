import importlib.util
import uuid
from datetime import datetime, timezone
from pathlib import Path as StdPath
from typing import Any

from anyio import Path as AsyncPath

__all__ = (
    "now_utc",
    "acreate_path",
    "get_bins",
    "import_module",
    "is_import_installed",
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def acreate_path(
    directory: StdPath | AsyncPath | str,
    filename: str,
    extension: str = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
) -> AsyncPath:
    """
    Generate a new file path asynchronously with optional timestamp and a random suffix.
    Uses non-blocking I/O (AnyIO).
    """

    # Use AsyncPath for construction and execution
    if "/" in filename:
        sub_dir, filename = filename.split("/")[:-1], filename.split("/")[-1]
        directory = AsyncPath(directory) / "/".join(sub_dir)

    if "\\" in filename:
        raise ValueError("Filename cannot contain directory separators.")

    # Ensure directory is an AsyncPath
    directory = AsyncPath(directory)
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
    else:
        name, ext = filename, extension
    ext = f".{ext.lstrip('.')}" if ext else ""

    if timestamp:
        # datetime.now() is generally non-blocking
        ts_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
        name = f"{ts_str}_{name}" if time_prefix else f"{name}_{ts_str}"

    if random_hash_digits > 0:
        random_suffix = uuid.uuid4().hex[:random_hash_digits]
        name = f"{name}-{random_suffix}"

    full_path = directory / f"{name}{ext}"

    # --- CRITICAL: ASYNC I/O Operations ---
    await full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)

    if await full_path.exists() and not file_exist_ok:
        raise FileExistsError(
            f"File {full_path} already exists and file_exist_ok is False."
        )

    return full_path


def get_bins(input_: list[str], upper: int) -> list[list[int]]:
    """Organizes indices of strings into bins based on a cumulative upper limit.

    Args:
        input_ (List[str]): The list of strings to be binned.
        upper (int): The cumulative length upper limit for each bin.

    Returns:
        List[List[int]]: A list of bins, each bin is a list of indices from the input list.
    """
    current = 0
    bins = []
    current_bin = []
    for idx, item in enumerate(input_):
        if current + len(item) < upper:
            current_bin.append(idx)
            current += len(item)
        else:
            bins.append(current_bin)
            current_bin = [idx]
            current = len(item)
    if current_bin:
        bins.append(current_bin)
    return bins


def import_module(
    package_name: str,
    module_name: str = None,
    import_name: str | list = None,
) -> Any:
    """
    Import a module by its path.

    Args:
        module_path: The path of the module to import.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        full_import_path = (
            f"{package_name}.{module_name}" if module_name else package_name
        )

        if import_name:
            import_name = (
                [import_name]
                if not isinstance(import_name, list)
                else import_name
            )
            a = __import__(
                full_import_path,
                fromlist=import_name,
            )
            if len(import_name) == 1:
                return getattr(a, import_name[0])
            return [getattr(a, name) for name in import_name]
        else:
            return __import__(full_import_path)

    except ImportError as e:
        raise ImportError(
            f"Failed to import module {full_import_path}: {e}"
        ) from e


def is_import_installed(package_name: str) -> bool:
    """
    Check if a package is installed.

    Args:
        package_name: The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    return importlib.util.find_spec(package_name) is not None
