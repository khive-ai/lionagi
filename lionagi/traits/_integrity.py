from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Any

# ---------- optional deps ----------------------------------------------------
try:
    import blake3  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    blake3 = None  # type: ignore


# ---------- supported algorithms --------------------------------------------
class Algo(Enum):
    """Enumeration of supported integrity algorithms."""

    SHA256 = auto()
    BLAKE3 = auto()
    XXH64 = auto()


Hasher = Callable[[bytes], str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blake3(data: bytes) -> str:
    if blake3 is None:
        raise RuntimeError("blake3 package not installed")
    return blake3.blake3(data).hexdigest()


def _xxh64(data: bytes) -> str:
    import xxhash

    return xxhash.xxh64(data).hexdigest()


_IMPL: dict[Algo, Hasher] = {
    Algo.SHA256: _sha256,
    Algo.BLAKE3: _blake3,
    Algo.XXH64: _xxh64,
}


# ---------- object hashing ---------------------------------------------------
def digest(obj: Any, *, algo: Algo = Algo.XXH64) -> str:
    """Return a deterministic digest of a JSON-serialisable object.

    Parameters
    ----------
    obj
        Any JSON-serialisable Python value (dict, list, str, …).
    algo
        Hashing algorithm to use (default SHA-256).

    Returns
    -------
    str
        Hex-encoded digest.
    """
    try:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except TypeError as exc:  # pragma: no cover
        raise TypeError("Object is not JSON-serialisable") from exc

    return _IMPL[algo](data)


# ---------- streaming file hashing ------------------------------------------
def file_digest(
    path: str | Path, *, algo: Algo = Algo.XXH64, block_size: int = 1 << 20
) -> str:
    """Compute the digest of a file incrementally.

    Parameters
    ----------
    path
        Path to the file on disk.
    algo
        Algorithm to use.
    block_size
        Number of bytes per read (default 1 MiB).

    Returns
    -------
    str
        Hex-encoded digest of the file contents.
    """
    import xxhash

    if algo is Algo.SHA256:
        h = hashlib.sha256()
    elif algo is Algo.BLAKE3:
        if blake3 is None:
            raise RuntimeError("blake3 package not installed")
        h = blake3.blake3()
    elif algo is Algo.XXH64:
        if xxhash is None:
            raise RuntimeError("xxhash package not installed")
        h = xxhash.xxh64()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported algorithm {algo}")

    path = Path(path)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()
