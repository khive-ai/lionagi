# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Capability-based policy enforcement for morphism execution.

Coverage is segment-aware: "net.out" covers "net.out:any.host" (no resource = wildcard),
"fs.read:/data/*" covers "fs.read:/data/x" but not "fs.read:/data/../etc/passwd".
Resources are canonicalized before matching — fnmatch is not used to prevent traversal attacks.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import Principal

__all__ = ("covers", "policy_check")


def _split(s: str) -> tuple[str, str]:
    d, _, r = s.partition(":")
    return d, r


def _canonicalize_resource(res: str) -> str:
    if not res:
        return res
    normalized = posixpath.normpath(res)
    if res.endswith("/*") and not normalized.endswith("/*"):
        normalized += "/*"
    return normalized


def _segments_match(have_segs: list[str], req_segs: list[str]) -> bool:
    if len(have_segs) != len(req_segs):
        if have_segs and have_segs[-1] == "*":
            return len(req_segs) >= len(have_segs) - 1
        return False
    for h, r in zip(have_segs, req_segs):
        if h == "*":
            continue
        if h != r:
            return False
    return True


def _covers_resource(have_res: str, req_res: str) -> bool:
    if have_res == "":
        return True
    if req_res == "":
        return have_res == ""

    have_res = _canonicalize_resource(have_res)
    req_res = _canonicalize_resource(req_res)

    have_wc = "*" in have_res
    req_wc = "*" in req_res

    if not have_wc and not req_wc:
        return have_res == req_res

    if have_wc and not req_wc:
        if "/" in have_res or "/" in req_res:
            return _segments_match(have_res.split("/"), req_res.split("/"))
        return have_res.rstrip("*") == "" or req_res.startswith(have_res.rstrip("*"))

    if req_wc and not have_wc:
        return False

    # Both wildcards: have must be at least as broad
    have_segs = have_res.rstrip("*").rstrip("/").split("/")
    req_segs = req_res.rstrip("*").rstrip("/").split("/")
    if len(have_segs) > len(req_segs):
        return False
    return all(h == "*" or h == r for h, r in zip(have_segs, req_segs))


def covers(have: str, req: str) -> bool:
    """True if capability string 'have' covers requirement 'req'."""
    hd, hr = _split(have)
    rd, rr = _split(req)
    if hd != rd:
        return False
    return _covers_resource(hr, rr)


def policy_check(
    principal: Principal,
    morphism: Any,
    override_reqs: set[str] | None = None,
    extra_rights: frozenset[str] | None = None,
) -> bool:
    """Return True iff every required right is covered by a capability held by principal.

    extra_rights extends principal.rights() with accumulated predecessor provides.
    """
    reqs: set[str]
    if override_reqs is not None:
        reqs = set(override_reqs)
    else:
        raw = getattr(morphism, "requires", None)
        reqs = set(raw) if raw else set()

    if not reqs:
        return True

    have: frozenset[str] = principal.rights()
    if extra_rights:
        have = have | extra_rights
    return all(any(covers(h, r) for h in have) for r in reqs)
