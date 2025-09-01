from __future__ import annotations

import os.path
from fnmatch import fnmatch

from .morphism import Morphism
from .types import Branch, rights_view


def _normalize_path(path: str) -> str:
    """Normalize path to prevent directory traversal attacks."""
    if not path or not path.startswith("/"):
        return path
    # Use os.path.normpath to resolve .. and . components
    normalized = os.path.normpath(path)
    # Ensure it's still an absolute path after normalization
    return normalized if normalized.startswith("/") else path


def _split(s: str) -> tuple[str, str]:
    """Split 'domain.action[:resource]' -> ('domain.action', 'resource-or-empty')."""
    d, _, r = s.partition(":")
    # Normalize resource path if it looks like a file path
    if r and r.startswith("/"):
        r = _normalize_path(r)
    return d, r


def _is_prefix_star(p: str) -> bool:
    """True iff pattern is exactly 'prefix*' (single star at the end)."""
    return p.endswith("*") and p.count("*") == 1


def _covers_resource(have_res: str, req_res: str) -> bool:
    """Conservative capability pattern matching for resource coverage.

    Implements secure wildcard matching based on capability-based security principles.
    Pattern matching uses conservative semantics to prevent privilege escalation.

    References:
    - Capability-based security: https://en.wikipedia.org/wiki/Capability-based_security
    - fnmatch algorithm: https://docs.python.org/3/library/fnmatch.html

    - If have has NO resource -> covers all in that domain/action.
    - If req has NO resource  -> only have with NO resource covers it.
    - Both concrete -> exact equality.
    - have wildcard + req concrete -> fnmatch(req, have).
    - req wildcard + have concrete -> deny (too narrow to cover).
    - both wildcard -> allow only simple prefix-star patterns, and only if
      have-prefix is a prefix of req-prefix. Otherwise deny.
    """
    if have_res == "":
        return True
    if req_res == "":
        return have_res == ""

    have_wc = "*" in have_res
    req_wc = "*" in req_res

    if not have_wc and not req_wc:
        return have_res == req_res

    if have_wc and not req_wc:
        return fnmatch(req_res, have_res)

    if req_wc and not have_wc:
        return False

    if _is_prefix_star(have_res) and _is_prefix_star(req_res):
        have_pref = have_res[:-1]
        req_pref = req_res[:-1]
        return req_pref.startswith(have_pref)

    return False


def _covers(have: str, req: str) -> bool:
    hd, hr = _split(have)
    rd, rr = _split(req)
    if hd != rd:
        return False
    return _covers_resource(hr, rr)


def policy_check(
    branch: Branch,
    morphism: Morphism,
    override_reqs: set[str] | None = None,
) -> bool:
    """
    True iff every required right R is covered by some capability H in the branch.

    If 'override_reqs' is provided (e.g., derived from morphism params via
    Morphism.required_rights(**kwargs)), it is used instead of 'morphism.requires'.
    """
    reqs = (
        set(override_reqs)
        if override_reqs is not None
        else (getattr(morphism, "requires", set()) or set())
    )
    if not reqs:
        return True
    # Use the live capability view so runtime changes are honored
    have = rights_view(branch)
    return all(any(_covers(h, r) for h in have) for r in reqs)
