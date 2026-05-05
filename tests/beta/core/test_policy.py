# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.beta.core.policy: covers, policy_check, resource matching."""

from __future__ import annotations

import pytest

from lionagi.beta.core.policy import (
    _canonicalize_resource,
    _covers_resource,
    _split,
    covers,
    policy_check,
)
from lionagi.beta.core.types import Capability, Principal

# ---------------------------------------------------------------------------
# _split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_simple_domain_only(self):
        assert _split("fs.read") == ("fs.read", "")

    def test_domain_with_resource(self):
        assert _split("fs.read:/data/x") == ("fs.read", "/data/x")

    def test_first_colon_only(self):
        assert _split("net.out:host:port") == ("net.out", "host:port")

    def test_empty_string(self):
        assert _split("") == ("", "")

    def test_only_colon(self):
        assert _split(":") == ("", "")


# ---------------------------------------------------------------------------
# _canonicalize_resource
# ---------------------------------------------------------------------------


class TestCanonicalizeResource:
    def test_empty_passthrough(self):
        assert _canonicalize_resource("") == ""

    def test_simple_path(self):
        assert _canonicalize_resource("/data/x") == "/data/x"

    def test_path_traversal_normalized(self):
        result = _canonicalize_resource("/data/../etc/passwd")
        assert result == "/etc/passwd"

    def test_trailing_wildcard_preserved(self):
        result = _canonicalize_resource("/data/*")
        assert result.endswith("/*")

    def test_path_traversal_in_wildcard(self):
        result = _canonicalize_resource("/data/../*")
        # after normpath, should not contain traversal
        assert ".." not in result


# ---------------------------------------------------------------------------
# _covers_resource
# ---------------------------------------------------------------------------


class TestCoversResource:
    def test_empty_have_covers_any(self):
        assert _covers_resource("", "/anything") is True

    def test_empty_have_covers_empty_req(self):
        assert _covers_resource("", "") is True

    def test_exact_match(self):
        assert _covers_resource("/data/x", "/data/x") is True

    def test_exact_mismatch(self):
        assert _covers_resource("/data/x", "/data/y") is False

    def test_wildcard_covers_child(self):
        assert _covers_resource("/data/*", "/data/x") is True

    def test_wildcard_does_not_cover_parent(self):
        assert _covers_resource("/data/*", "/other/x") is False

    def test_req_wildcard_not_escalated_by_exact_have(self):
        # req with wildcard should not be covered by exact have (cannot escalate)
        assert _covers_resource("/data/x", "/data/*") is False

    def test_nonempty_req_empty_have(self):
        # empty have_res → True (wildcard)
        assert _covers_resource("", "/specific/path") is True

    def test_path_traversal_blocked(self):
        # "/data/../etc/passwd" canonicalizes to "/etc/passwd" — not under /data/*
        assert _covers_resource("/data/*", "/data/../etc/passwd") is False


# ---------------------------------------------------------------------------
# covers
# ---------------------------------------------------------------------------


class TestCovers:
    def test_identical_strings(self):
        assert covers("fs.read", "fs.read") is True

    def test_different_domains(self):
        assert covers("fs.read", "net.out") is False

    def test_same_domain_empty_res_covers_any(self):
        # "fs.read" (no colon, empty resource) covers "fs.read:/data/x"
        assert covers("fs.read", "fs.read:/data/x") is True

    def test_explicit_empty_resource_covers_any(self):
        # "fs.read:" with empty resource is still wildcard
        assert covers("fs.read:", "fs.read:/data/x") is True

    def test_wildcard_path_covers_child(self):
        assert covers("fs.read:/data/*", "fs.read:/data/x") is True

    def test_wildcard_path_does_not_cover_traversal(self):
        assert covers("fs.read:/data/*", "fs.read:/data/../etc/passwd") is False

    def test_net_out_covers_specific_host(self):
        assert covers("net.out", "net.out:any.host") is True

    def test_narrower_does_not_cover_broader(self):
        assert covers("fs.read:/data/x", "fs.read:/data/*") is False

    def test_req_wildcard_not_covered_by_narrow_have(self):
        assert covers("fs.read:/data/a", "fs.read:/data/*") is False

    def test_both_wildcards_broader_covers(self):
        # "/data/*" is broader than "/data/sub/*"
        assert covers("fs.read:/data/*", "fs.read:/data/sub/x") is True

    def test_domain_prefix_mismatch(self):
        assert covers("fs", "fs.read") is False


# ---------------------------------------------------------------------------
# policy_check
# ---------------------------------------------------------------------------


class TestPolicyCheck:
    def _make_morphism(self, requires=None, provides=None):
        """Simple namespace object mimicking Morphism attributes."""

        class _M:
            pass

        m = _M()
        m.requires = frozenset(requires or [])
        m.provides = frozenset(provides or [])
        return m

    def test_empty_requires_always_true(self):
        p = Principal()
        m = self._make_morphism(requires=[])
        assert policy_check(p, m) is True

    def test_right_present_in_caps(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"fs.read"}))
        p2 = Principal(id=p.id, caps=[cap])
        m = self._make_morphism(requires=["fs.read"])
        assert policy_check(p2, m) is True

    def test_missing_right_returns_false(self):
        p = Principal()
        m = self._make_morphism(requires=["fs.write"])
        assert policy_check(p, m) is False

    def test_partial_rights_returns_false(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"fs.read"}))
        p2 = Principal(id=p.id, caps=[cap])
        m = self._make_morphism(requires=["fs.read", "fs.write"])
        assert policy_check(p2, m) is False

    def test_override_reqs_replaces_morphism_requires(self):
        p = Principal()
        # morphism requires something, but override says nothing needed
        m = self._make_morphism(requires=["admin"])
        assert policy_check(p, m, override_reqs=set()) is True

    def test_override_reqs_stricter(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"fs.read"}))
        p2 = Principal(id=p.id, caps=[cap])
        m = self._make_morphism(requires=[])
        # override adds a requirement not met
        assert policy_check(p2, m, override_reqs={"net.out"}) is False

    def test_extra_rights_extend_principal(self):
        p = Principal()
        m = self._make_morphism(requires=["dynamic.right"])
        assert policy_check(p, m, extra_rights=frozenset({"dynamic.right"})) is True

    def test_covers_wildcard_satisfies_specific(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"fs.read"}))
        p2 = Principal(id=p.id, caps=[cap])
        m = self._make_morphism(requires=["fs.read:/data/secret.txt"])
        assert policy_check(p2, m) is True

    def test_morphism_with_no_requires_attr(self):
        p = Principal()

        class _NoReqs:
            pass

        assert policy_check(p, _NoReqs()) is True

    def test_other_principal_cap_not_counted(self):
        # Cap minted for a different principal.id should not count
        other_id = Principal()
        cap = Capability(subject=other_id.id, rights=frozenset({"fs.read"}))
        p = Principal(caps=[cap])
        m = self._make_morphism(requires=["fs.read"])
        assert policy_check(p, m) is False
