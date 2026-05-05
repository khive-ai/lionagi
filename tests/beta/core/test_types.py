# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.beta.core.types: now_utc, Capability, Principal, Observation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from lionagi.beta.core.types import Capability, Observation, Principal, now_utc

# ---------------------------------------------------------------------------
# now_utc
# ---------------------------------------------------------------------------


class TestNowUtc:
    def test_returns_datetime(self):
        result = now_utc()
        assert isinstance(result, datetime)

    def test_has_utc_timezone(self):
        result = now_utc()
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0

    def test_returns_current_time(self):
        import time

        before = datetime.now(timezone.utc)
        result = now_utc()
        after = datetime.now(timezone.utc)
        assert before <= result <= after

    def test_successive_calls_monotone(self):
        t1 = now_utc()
        t2 = now_utc()
        assert t2 >= t1


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class TestCapability:
    def test_basic_construction(self):
        subj = uuid4()
        cap = Capability(subject=subj, rights=frozenset({"read", "write"}))
        assert cap.subject == subj
        assert "read" in cap.rights
        assert "write" in cap.rights

    def test_frozen_struct(self):
        cap = Capability(subject=uuid4(), rights=frozenset({"r"}))
        with pytest.raises(Exception):
            cap.rights = frozenset({"x"})  # type: ignore[misc]

    def test_hashable(self):
        subj = uuid4()
        cap = Capability(subject=subj, rights=frozenset({"a"}))
        s = {cap}
        assert cap in s

    def test_equality_same_values(self):
        subj = uuid4()
        c1 = Capability(subject=subj, rights=frozenset({"a"}))
        c2 = Capability(subject=subj, rights=frozenset({"a"}))
        assert c1 == c2

    def test_equality_different_subject(self):
        r = frozenset({"a"})
        c1 = Capability(subject=uuid4(), rights=r)
        c2 = Capability(subject=uuid4(), rights=r)
        assert c1 != c2

    def test_empty_rights(self):
        cap = Capability(subject=uuid4(), rights=frozenset())
        assert cap.rights == frozenset()

    def test_hash_consistent(self):
        subj = uuid4()
        cap = Capability(subject=subj, rights=frozenset({"x", "y"}))
        assert hash(cap) == hash(cap)

    def test_usable_in_dict_key(self):
        cap = Capability(subject=uuid4(), rights=frozenset({"r"}))
        d = {cap: "value"}
        assert d[cap] == "value"


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


class TestPrincipalBasic:
    def test_default_construction(self):
        p = Principal()
        assert p.name == "default"
        assert p.ctx == {}
        assert p.caps == ()
        assert isinstance(p.id, UUID)

    def test_custom_name(self):
        p = Principal(name="alice")
        assert p.name == "alice"

    def test_id_is_uuid(self):
        p = Principal()
        assert isinstance(p.id, UUID)

    def test_unique_ids(self):
        p1 = Principal()
        p2 = Principal()
        assert p1.id != p2.id

    def test_lineage_default_empty(self):
        p = Principal()
        assert p.lineage == ()

    def test_tags_default_empty(self):
        p = Principal()
        assert p.tags == ()


class TestPrincipalValidateCaps:
    def test_caps_from_capability_objects(self):
        subj = uuid4()
        cap = Capability(subject=subj, rights=frozenset({"read"}))
        p = Principal(caps=[cap])
        assert len(p.caps) == 1
        assert p.caps[0] == cap

    def test_caps_from_dict(self):
        subj = uuid4()
        p = Principal(caps=[{"subject": subj, "rights": frozenset({"read"})}])
        assert len(p.caps) == 1
        assert p.caps[0].subject == subj
        assert "read" in p.caps[0].rights

    def test_caps_from_dict_string_subject(self):
        subj = uuid4()
        p = Principal(caps=[{"subject": str(subj), "rights": frozenset({"write"})}])
        assert p.caps[0].subject == subj

    def test_caps_from_dict_list_rights(self):
        subj = uuid4()
        p = Principal(caps=[{"subject": subj, "rights": ["read", "write"]}])
        assert p.caps[0].rights == frozenset({"read", "write"})

    def test_multiple_caps(self):
        subj1, subj2 = uuid4(), uuid4()
        cap1 = Capability(subject=subj1, rights=frozenset({"r"}))
        cap2 = Capability(subject=subj2, rights=frozenset({"w"}))
        p = Principal(caps=[cap1, cap2])
        assert len(p.caps) == 2


class TestPrincipalRights:
    def test_rights_empty_when_no_caps(self):
        p = Principal()
        assert p.rights() == frozenset()

    def test_rights_returns_own_rights(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"read", "write"}))
        p2 = Principal(id=p.id, caps=[cap])
        assert p2.rights() == frozenset({"read", "write"})

    def test_rights_excludes_other_subject_caps(self):
        other_subj = uuid4()
        cap = Capability(subject=other_subj, rights=frozenset({"admin"}))
        p = Principal(caps=[cap])
        assert p.rights() == frozenset()

    def test_has_right_true(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"read"}))
        p2 = Principal(id=p.id, caps=[cap])
        assert p2.has_right("read") is True

    def test_has_right_false(self):
        p = Principal()
        assert p.has_right("write") is False

    def test_rights_union_over_multiple_caps(self):
        p = Principal()
        cap1 = Capability(subject=p.id, rights=frozenset({"read"}))
        cap2 = Capability(subject=p.id, rights=frozenset({"write"}))
        p2 = Principal(id=p.id, caps=[cap1, cap2])
        assert p2.rights() == frozenset({"read", "write"})


class TestPrincipalGrant:
    def test_grant_returns_new_principal(self):
        p = Principal()
        p2 = p.grant("read")
        assert p2 is not p

    def test_grant_preserves_id(self):
        p = Principal()
        p2 = p.grant("read")
        assert p2.id == p.id

    def test_grant_adds_right(self):
        p = Principal()
        p2 = p.grant("read")
        assert p2.has_right("read")

    def test_grant_multiple_rights(self):
        p = Principal()
        p2 = p.grant("read", "write", "execute")
        assert p2.has_right("read")
        assert p2.has_right("write")
        assert p2.has_right("execute")

    def test_grant_original_unchanged(self):
        p = Principal()
        _ = p.grant("read")
        assert not p.has_right("read")

    def test_grant_stacks(self):
        p = Principal()
        p2 = p.grant("r")
        p3 = p2.grant("w")
        assert p3.has_right("r")
        assert p3.has_right("w")


class TestPrincipalSerializeCaps:
    def test_serialize_caps_returns_list_of_dicts(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"read"}))
        p2 = Principal(id=p.id, caps=[cap])
        dumped = p2.model_dump()
        caps_data = dumped["caps"]
        assert isinstance(caps_data, list)
        assert len(caps_data) == 1
        assert "subject" in caps_data[0]
        assert "rights" in caps_data[0]

    def test_serialize_rights_sorted(self):
        p = Principal()
        cap = Capability(subject=p.id, rights=frozenset({"z", "a", "m"}))
        p2 = Principal(id=p.id, caps=[cap])
        dumped = p2.model_dump()
        rights = dumped["caps"][0]["rights"]
        assert rights == sorted(rights)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_default_construction(self):
        obs = Observation()
        assert isinstance(obs.id, UUID)
        assert isinstance(obs.ts, datetime)
        assert isinstance(obs.who, UUID)
        assert obs.what == ""
        assert obs.payload == {}
        assert obs.lineage == ()
        assert obs.tags == ()

    def test_custom_fields(self):
        uid = uuid4()
        obs = Observation(who=uid, what="test_event", payload={"x": 1})
        assert obs.who == uid
        assert obs.what == "test_event"
        assert obs.payload == {"x": 1}

    def test_unique_ids(self):
        o1 = Observation()
        o2 = Observation()
        assert o1.id != o2.id

    def test_ts_is_utc(self):
        obs = Observation()
        assert obs.ts.tzinfo is not None
