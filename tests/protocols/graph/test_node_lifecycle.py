# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for PR8: Enhanced Node with NodeConfig, create_node
factory, and lifecycle methods (touch, soft_delete, restore, rehash).

Covers:
- NodeConfig frozen dataclass (defaults, properties, immutability)
- compute_content_hash() determinism and input variants
- create_node() factory (class generation, config wiring, extra fields)
- Node lifecycle methods with various config combinations
- Backwards compatibility (base Node has node_config=None)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from lionagi.protocols.graph.node import Node
from lionagi.protocols.graph.node_factory import (
    NodeConfig,
    compute_content_hash,
    create_node,
)


# ===================================================================
# 1. NodeConfig
# ===================================================================


class TestNodeConfigDefaults:
    """NodeConfig default values and basic construction."""

    def test_all_defaults(self):
        cfg = NodeConfig()
        assert cfg.table_name is None
        assert cfg.schema == "public"
        assert cfg.soft_delete is False
        assert cfg.versioning is False
        assert cfg.content_hashing is False
        assert cfg.track_updated_at is False

    def test_explicit_table_name(self):
        cfg = NodeConfig(table_name="jobs")
        assert cfg.table_name == "jobs"

    def test_explicit_schema(self):
        cfg = NodeConfig(schema="analytics")
        assert cfg.schema == "analytics"

    def test_all_flags_enabled(self):
        cfg = NodeConfig(
            table_name="items",
            schema="warehouse",
            soft_delete=True,
            versioning=True,
            content_hashing=True,
            track_updated_at=True,
        )
        assert cfg.table_name == "items"
        assert cfg.schema == "warehouse"
        assert cfg.soft_delete is True
        assert cfg.versioning is True
        assert cfg.content_hashing is True
        assert cfg.track_updated_at is True


class TestNodeConfigProperties:
    """NodeConfig computed properties."""

    def test_is_persisted_when_table_name_set(self):
        cfg = NodeConfig(table_name="events")
        assert cfg.is_persisted is True

    def test_is_persisted_when_table_name_none(self):
        cfg = NodeConfig()
        assert cfg.is_persisted is False

    def test_has_audit_fields_false_when_all_disabled(self):
        cfg = NodeConfig()
        assert cfg.has_audit_fields is False

    def test_has_audit_fields_true_when_content_hashing(self):
        cfg = NodeConfig(content_hashing=True)
        assert cfg.has_audit_fields is True

    def test_has_audit_fields_true_when_soft_delete(self):
        cfg = NodeConfig(soft_delete=True)
        assert cfg.has_audit_fields is True

    def test_has_audit_fields_true_when_versioning(self):
        cfg = NodeConfig(versioning=True)
        assert cfg.has_audit_fields is True

    def test_has_audit_fields_true_when_track_updated_at(self):
        cfg = NodeConfig(track_updated_at=True)
        assert cfg.has_audit_fields is True

    def test_has_audit_fields_true_when_multiple_flags(self):
        cfg = NodeConfig(soft_delete=True, versioning=True)
        assert cfg.has_audit_fields is True


class TestNodeConfigImmutability:
    """NodeConfig frozen=True enforcement."""

    def test_cannot_set_table_name(self):
        cfg = NodeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.table_name = "oops"

    def test_cannot_set_schema(self):
        cfg = NodeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.schema = "oops"

    def test_cannot_set_soft_delete(self):
        cfg = NodeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.soft_delete = True

    def test_cannot_set_versioning(self):
        cfg = NodeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.versioning = True

    def test_cannot_set_content_hashing(self):
        cfg = NodeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.content_hashing = True

    def test_cannot_set_track_updated_at(self):
        cfg = NodeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.track_updated_at = True


# ===================================================================
# 2. compute_content_hash
# ===================================================================


class TestComputeContentHash:
    """compute_content_hash input variants and determinism."""

    def test_none_content(self):
        result = compute_content_hash(None)
        expected = hashlib.sha256(b"null").hexdigest()
        assert result == expected

    def test_string_content(self):
        result = compute_content_hash("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_empty_string_content(self):
        result = compute_content_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_bytes_content(self):
        data = b"\x00\x01\x02"
        result = compute_content_hash(data)
        expected = hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_dict_content_json_serialized(self):
        content = {"b": 2, "a": 1}
        result = compute_content_hash(content)
        expected = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_dict_content_sort_keys(self):
        """Dict ordering does not affect the hash due to sort_keys=True."""
        h1 = compute_content_hash({"z": 1, "a": 2})
        h2 = compute_content_hash({"a": 2, "z": 1})
        assert h1 == h2

    def test_list_content(self):
        content = [1, 2, 3]
        result = compute_content_hash(content)
        expected = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_nested_dict_content(self):
        content = {"outer": {"inner": [1, 2]}}
        result = compute_content_hash(content)
        expected = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_integer_content(self):
        """Integers are JSON-serializable so they go through dumps."""
        result = compute_content_hash(42)
        expected = hashlib.sha256(
            json.dumps(42, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_custom_object_goes_through_json_dumps_default_str(self):
        """Custom objects are handled by json.dumps(default=str)."""

        class Weird:
            def __str__(self):
                return "weird-object"

        result = compute_content_hash(Weird())
        # json.dumps with default=str serializes as '"weird-object"'
        expected = hashlib.sha256(
            json.dumps("weird-object", sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_determinism_same_content(self):
        """Same content always produces the same hash."""
        a = compute_content_hash({"key": "value"})
        b = compute_content_hash({"key": "value"})
        assert a == b

    def test_different_content_different_hash(self):
        a = compute_content_hash("alpha")
        b = compute_content_hash("beta")
        assert a != b

    def test_hash_is_64_char_hex(self):
        """SHA-256 hex digest is always 64 characters."""
        result = compute_content_hash("anything")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_dict_with_non_serializable_values(self):
        """Dict with datetime values uses default=str fallback in dumps."""
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        content = {"ts": dt}
        result = compute_content_hash(content)
        expected = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected


# ===================================================================
# 3. create_node factory
# ===================================================================


class TestCreateNodeBasic:
    """create_node factory: class creation, naming, config wiring."""

    def test_returns_a_type(self):
        cls = create_node("Widget")
        assert isinstance(cls, type)

    def test_class_name(self):
        cls = create_node("Widget")
        assert cls.__name__ == "Widget"

    def test_is_node_subclass(self):
        cls = create_node("Widget")
        assert issubclass(cls, Node)

    def test_node_config_set_on_class(self):
        cls = create_node("Widget", table_name="widgets", versioning=True)
        assert isinstance(cls.node_config, NodeConfig)
        assert cls.node_config.table_name == "widgets"
        assert cls.node_config.versioning is True

    def test_node_config_defaults(self):
        cls = create_node("Widget")
        cfg = cls.node_config
        assert cfg.table_name is None
        assert cfg.schema == "public"
        assert cfg.soft_delete is False
        assert cfg.versioning is False
        assert cfg.content_hashing is False
        assert cfg.track_updated_at is False

    def test_instances_are_nodes(self):
        cls = create_node("Widget")
        w = cls(content="hello")
        assert isinstance(w, Node)
        assert w.content == "hello"

    def test_doc_string_set(self):
        cls = create_node("Widget", doc="A widget node.")
        assert cls.__doc__ == "A widget node."

    def test_doc_string_not_set(self):
        cls = create_node("Widget")
        # When doc is not provided, __doc__ may be None or inherited; either
        # way it should not be "A widget node."
        assert cls.__doc__ != "A widget node."


class TestCreateNodeExtraFields:
    """create_node with extra_fields parameter."""

    def test_extra_field_available_on_instance(self):
        cls = create_node(
            "Job",
            extra_fields={"priority": (int, 0)},
        )
        job = cls(content="build")
        assert job.priority == 0

    def test_extra_field_custom_value(self):
        cls = create_node(
            "Job",
            extra_fields={"priority": (int, 0)},
        )
        job = cls(content="build", priority=5)
        assert job.priority == 5

    def test_multiple_extra_fields(self):
        cls = create_node(
            "Task",
            extra_fields={
                "priority": (int, 0),
                "label": (str, ""),
            },
        )
        t = cls(priority=3, label="urgent")
        assert t.priority == 3
        assert t.label == "urgent"

    def test_extra_field_optional_type(self):
        cls = create_node(
            "Memo",
            extra_fields={"note": (str | None, None)},
        )
        m = cls()
        assert m.note is None

    def test_extra_fields_none_means_no_extras(self):
        cls = create_node("Plain")
        p = cls(content="x")
        assert not hasattr(p, "priority")


class TestCreateNodeConfigPropagation:
    """create_node: config flags propagate to lifecycle methods."""

    def test_touch_increments_version(self):
        cls = create_node("Versioned", versioning=True)
        v = cls(content="data")
        v.touch()
        assert v.metadata.get("version") == 1
        v.touch()
        assert v.metadata.get("version") == 2

    def test_soft_delete_enabled(self):
        cls = create_node("Deletable", soft_delete=True)
        d = cls(content="bye")
        d.soft_delete()
        assert d.metadata["is_deleted"] is True

    def test_content_hashing_on_touch(self):
        cls = create_node("Hashed", content_hashing=True)
        h = cls(content="payload")
        h.touch()
        assert "content_hash" in h.metadata
        assert len(h.metadata["content_hash"]) == 64

    def test_full_config(self):
        cls = create_node(
            "FullAudit",
            table_name="audits",
            soft_delete=True,
            versioning=True,
            content_hashing=True,
            track_updated_at=True,
        )
        cfg = cls.node_config
        assert cfg.table_name == "audits"
        assert cfg.soft_delete is True
        assert cfg.versioning is True
        assert cfg.content_hashing is True
        assert cfg.track_updated_at is True
        assert cfg.is_persisted is True
        assert cfg.has_audit_fields is True


# ===================================================================
# 4. Node lifecycle methods
# ===================================================================


class TestNodeTouch:
    """Node.touch() behaviour with various configs."""

    def test_touch_noop_on_base_node(self):
        """Base Node has node_config=None; touch is a no-op."""
        n = Node(content="hi")
        n.touch()
        assert "version" not in n.metadata
        assert "updated_at" not in n.metadata

    def test_touch_noop_when_config_none(self):
        """Subclass without config should also be a no-op."""

        class Bare(Node):
            pass

        b = Bare()
        b.touch()
        assert "version" not in b.metadata

    def test_touch_track_updated_at(self):
        cls = create_node("Tracked", track_updated_at=True)
        t = cls(content="x")
        before = datetime.now(timezone.utc).isoformat()
        t.touch()
        after = datetime.now(timezone.utc).isoformat()
        ts = t.metadata["updated_at"]
        assert before <= ts <= after

    def test_touch_versioning_starts_at_one(self):
        cls = create_node("V", versioning=True)
        v = cls()
        v.touch()
        assert v.metadata["version"] == 1

    def test_touch_versioning_increments(self):
        cls = create_node("V", versioning=True)
        v = cls()
        for expected in range(1, 6):
            v.touch()
            assert v.metadata["version"] == expected

    def test_touch_content_hashing_calls_rehash(self):
        cls = create_node("H", content_hashing=True)
        h = cls(content="data")
        h.touch()
        assert "content_hash" in h.metadata
        expected_hash = compute_content_hash("data")
        assert h.metadata["content_hash"] == expected_hash

    def test_touch_by_param_sets_updated_by(self):
        cls = create_node("ByUser", track_updated_at=True)
        b = cls()
        b.touch(by="alice")
        assert b.metadata["updated_by"] == "alice"

    def test_touch_by_none_does_not_set_updated_by(self):
        cls = create_node("ByUser", track_updated_at=True)
        b = cls()
        b.touch()
        assert "updated_by" not in b.metadata

    def test_touch_by_non_string_coerced(self):
        cls = create_node("ByUser", track_updated_at=True)
        b = cls()
        b.touch(by=42)
        assert b.metadata["updated_by"] == "42"

    def test_touch_all_features_combined(self):
        cls = create_node(
            "All",
            track_updated_at=True,
            versioning=True,
            content_hashing=True,
        )
        a = cls(content="payload")
        a.touch(by="system")
        assert a.metadata["version"] == 1
        assert "updated_at" in a.metadata
        assert a.metadata["updated_by"] == "system"
        assert "content_hash" in a.metadata


class TestNodeSoftDelete:
    """Node.soft_delete() behaviour."""

    def test_soft_delete_raises_on_base_node(self):
        n = Node()
        with pytest.raises(RuntimeError, match="does not support soft_delete"):
            n.soft_delete()

    def test_soft_delete_raises_when_flag_disabled(self):
        cls = create_node("NoDel", soft_delete=False)
        nd = cls()
        with pytest.raises(RuntimeError, match="does not support soft_delete"):
            nd.soft_delete()

    def test_soft_delete_sets_is_deleted(self):
        cls = create_node("Del", soft_delete=True)
        d = cls(content="bye")
        d.soft_delete()
        assert d.metadata["is_deleted"] is True

    def test_soft_delete_sets_deleted_at(self):
        cls = create_node("Del", soft_delete=True)
        d = cls()
        before = datetime.now(timezone.utc).isoformat()
        d.soft_delete()
        after = datetime.now(timezone.utc).isoformat()
        ts = d.metadata["deleted_at"]
        assert before <= ts <= after

    def test_soft_delete_with_by_param(self):
        cls = create_node("Del", soft_delete=True)
        d = cls()
        d.soft_delete(by="admin")
        assert d.metadata["deleted_by"] == "admin"

    def test_soft_delete_without_by_param(self):
        cls = create_node("Del", soft_delete=True)
        d = cls()
        d.soft_delete()
        assert "deleted_by" not in d.metadata

    def test_soft_delete_calls_touch(self):
        cls = create_node("Del", soft_delete=True, versioning=True)
        d = cls()
        d.soft_delete()
        # touch() should have incremented version
        assert d.metadata.get("version") == 1

    def test_soft_delete_with_track_updated_at(self):
        cls = create_node(
            "Del",
            soft_delete=True,
            track_updated_at=True,
        )
        d = cls()
        d.soft_delete(by="admin")
        assert d.metadata["is_deleted"] is True
        assert "updated_at" in d.metadata
        assert d.metadata.get("updated_by") == "admin"

    def test_soft_delete_raises_on_subclass_without_config(self):
        """Subclass inheriting node_config=None from base Node."""

        class PlainNode(Node):
            pass

        pn = PlainNode()
        with pytest.raises(RuntimeError, match="does not support soft_delete"):
            pn.soft_delete()


class TestNodeRestore:
    """Node.restore() behaviour."""

    def test_restore_raises_on_base_node(self):
        n = Node()
        with pytest.raises(RuntimeError, match="does not support restore"):
            n.restore()

    def test_restore_raises_when_flag_disabled(self):
        cls = create_node("NoRestore", soft_delete=False)
        nr = cls()
        with pytest.raises(RuntimeError, match="does not support restore"):
            nr.restore()

    def test_restore_sets_is_deleted_false(self):
        cls = create_node("Rest", soft_delete=True)
        r = cls()
        r.soft_delete()
        assert r.metadata["is_deleted"] is True
        r.restore()
        assert r.metadata["is_deleted"] is False

    def test_restore_removes_deleted_at(self):
        cls = create_node("Rest", soft_delete=True)
        r = cls()
        r.soft_delete()
        assert "deleted_at" in r.metadata
        r.restore()
        assert "deleted_at" not in r.metadata

    def test_restore_removes_deleted_by(self):
        cls = create_node("Rest", soft_delete=True)
        r = cls()
        r.soft_delete(by="admin")
        assert "deleted_by" in r.metadata
        r.restore()
        assert "deleted_by" not in r.metadata

    def test_restore_calls_touch(self):
        cls = create_node("Rest", soft_delete=True, versioning=True)
        r = cls()
        r.soft_delete()
        assert r.metadata["version"] == 1
        r.restore()
        # touch was called again during restore
        assert r.metadata["version"] == 2

    def test_restore_with_by_param(self):
        cls = create_node(
            "Rest", soft_delete=True, track_updated_at=True
        )
        r = cls()
        r.soft_delete(by="admin")
        r.restore(by="manager")
        assert r.metadata.get("updated_by") == "manager"
        assert r.metadata["is_deleted"] is False

    def test_restore_without_prior_delete(self):
        """Calling restore on a never-deleted node still works (sets is_deleted=False)."""
        cls = create_node("Rest", soft_delete=True)
        r = cls()
        r.restore()
        assert r.metadata["is_deleted"] is False

    def test_restore_raises_on_subclass_without_config(self):
        class PlainNode(Node):
            pass

        pn = PlainNode()
        with pytest.raises(RuntimeError, match="does not support restore"):
            pn.restore()


class TestNodeRehash:
    """Node.rehash() behaviour."""

    def test_rehash_returns_none_on_base_node(self):
        n = Node(content="hi")
        assert n.rehash() is None

    def test_rehash_returns_none_when_hashing_disabled(self):
        cls = create_node("NoHash", content_hashing=False)
        nh = cls(content="hi")
        assert nh.rehash() is None

    def test_rehash_returns_hex_string(self):
        cls = create_node("Hash", content_hashing=True)
        h = cls(content="payload")
        result = h.rehash()
        assert isinstance(result, str)
        assert len(result) == 64

    def test_rehash_stores_in_metadata(self):
        cls = create_node("Hash", content_hashing=True)
        h = cls(content="payload")
        h.rehash()
        assert "content_hash" in h.metadata

    def test_rehash_matches_compute_content_hash(self):
        cls = create_node("Hash", content_hashing=True)
        content = {"key": "value"}
        h = cls(content=content)
        result = h.rehash()
        assert result == compute_content_hash(content)

    def test_rehash_updates_on_content_change(self):
        cls = create_node("Hash", content_hashing=True)
        h = cls(content="before")
        hash1 = h.rehash()
        h.content = "after"
        hash2 = h.rehash()
        assert hash1 != hash2
        assert hash2 == compute_content_hash("after")

    def test_rehash_none_content(self):
        cls = create_node("Hash", content_hashing=True)
        h = cls(content=None)
        result = h.rehash()
        assert result == compute_content_hash(None)

    def test_rehash_returns_none_on_subclass_without_config(self):
        class PlainNode(Node):
            pass

        pn = PlainNode(content="data")
        assert pn.rehash() is None


# ===================================================================
# 5. Full delete-restore cycle
# ===================================================================


class TestDeleteRestoreCycle:
    """End-to-end soft_delete -> restore cycle with full audit trail."""

    def test_full_cycle(self):
        cls = create_node(
            "Audited",
            soft_delete=True,
            versioning=True,
            track_updated_at=True,
            content_hashing=True,
        )
        a = cls(content={"status": "active"})

        # --- initial state ---
        assert a.metadata == {}

        # --- touch once ---
        a.touch(by="alice")
        assert a.metadata["version"] == 1
        assert "updated_at" in a.metadata
        assert a.metadata["updated_by"] == "alice"
        assert "content_hash" in a.metadata
        hash_v1 = a.metadata["content_hash"]

        # --- soft_delete ---
        a.soft_delete(by="bob")
        assert a.metadata["is_deleted"] is True
        assert "deleted_at" in a.metadata
        assert a.metadata["deleted_by"] == "bob"
        assert a.metadata["version"] == 2  # touch was called inside soft_delete

        # --- restore ---
        a.restore(by="carol")
        assert a.metadata["is_deleted"] is False
        assert "deleted_at" not in a.metadata
        assert "deleted_by" not in a.metadata
        assert a.metadata["version"] == 3  # touch was called inside restore
        assert a.metadata["updated_by"] == "carol"

        # --- content change + rehash ---
        a.content = {"status": "updated"}
        a.touch(by="dave")
        assert a.metadata["version"] == 4
        hash_v4 = a.metadata["content_hash"]
        assert hash_v4 != hash_v1

    def test_double_delete_is_idempotent(self):
        cls = create_node("Del", soft_delete=True, versioning=True)
        d = cls()
        d.soft_delete()
        assert d.metadata["is_deleted"] is True
        assert d.metadata["version"] == 1
        d.soft_delete()
        assert d.metadata["is_deleted"] is True
        assert d.metadata["version"] == 2  # version still increments

    def test_double_restore_is_idempotent(self):
        cls = create_node("Rst", soft_delete=True, versioning=True)
        r = cls()
        r.soft_delete()
        r.restore()
        assert r.metadata["is_deleted"] is False
        assert r.metadata["version"] == 2
        r.restore()
        assert r.metadata["is_deleted"] is False
        assert r.metadata["version"] == 3


# ===================================================================
# 6. Backwards compatibility
# ===================================================================


class TestBackwardsCompatibility:
    """Ensure base Node and unaware subclasses remain unaffected."""

    def test_base_node_has_no_config(self):
        assert Node.node_config is None

    def test_base_node_touch_is_noop(self):
        n = Node(content="safe")
        n.touch()
        assert n.metadata == {}

    def test_base_node_rehash_returns_none(self):
        n = Node(content="safe")
        assert n.rehash() is None
        assert "content_hash" not in n.metadata

    def test_base_node_soft_delete_raises(self):
        n = Node()
        with pytest.raises(RuntimeError):
            n.soft_delete()

    def test_base_node_restore_raises(self):
        n = Node()
        with pytest.raises(RuntimeError):
            n.restore()

    def test_unaware_subclass_has_no_config(self):
        class LegacyNode(Node):
            extra: str = "legacy"

        assert LegacyNode.node_config is None
        ln_ = LegacyNode(extra="hello")
        ln_.touch()  # no-op
        assert ln_.metadata == {}

    def test_unaware_subclass_soft_delete_raises(self):
        class LegacyNode(Node):
            pass

        with pytest.raises(RuntimeError):
            LegacyNode().soft_delete()

    def test_unaware_subclass_restore_raises(self):
        class LegacyNode(Node):
            pass

        with pytest.raises(RuntimeError):
            LegacyNode().restore()

    def test_unaware_subclass_rehash_returns_none(self):
        class LegacyNode(Node):
            pass

        assert LegacyNode(content="x").rehash() is None

    def test_node_config_classvar_does_not_bleed_across_subclasses(self):
        """A config on one subclass must not affect another."""
        Configured = create_node("ConfiguredNode", versioning=True)
        assert Configured.node_config is not None
        assert Configured.node_config.versioning is True
        # Base Node is still unaffected
        assert Node.node_config is None


# ===================================================================
# 7. Edge cases and error conditions
# ===================================================================


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_create_node_empty_name(self):
        cls = create_node("")
        assert cls.__name__ == ""
        inst = cls(content="works")
        assert inst.content == "works"

    def test_create_node_with_only_extra_fields(self):
        cls = create_node(
            "FieldsOnly",
            extra_fields={"score": (float, 0.0)},
        )
        f = cls(score=9.5)
        assert f.score == 9.5
        assert f.node_config.table_name is None

    def test_touch_on_freshly_created_node_no_prior_version(self):
        cls = create_node("Fresh", versioning=True)
        f = cls()
        assert "version" not in f.metadata
        f.touch()
        assert f.metadata["version"] == 1

    def test_rehash_with_large_content(self):
        cls = create_node("Big", content_hashing=True)
        big_content = "x" * 1_000_000
        b = cls(content=big_content)
        result = b.rehash()
        assert result == compute_content_hash(big_content)

    def test_create_node_schema_override(self):
        cls = create_node("Custom", schema="private")
        assert cls.node_config.schema == "private"

    def test_multiple_create_node_calls_independent(self):
        A = create_node("A", versioning=True)
        B = create_node("B", soft_delete=True)
        assert A.node_config.versioning is True
        assert A.node_config.soft_delete is False
        assert B.node_config.soft_delete is True
        assert B.node_config.versioning is False

    def test_touch_updates_timestamp_each_call(self):
        cls = create_node("TS", track_updated_at=True)
        t = cls()
        t.touch()
        ts1 = t.metadata["updated_at"]
        # Ensure a tiny time gap so timestamps differ
        time.sleep(0.01)
        t.touch()
        ts2 = t.metadata["updated_at"]
        assert ts2 >= ts1

    def test_soft_delete_error_message_includes_class_name(self):
        cls = create_node("MySpecialNode", soft_delete=False)
        inst = cls()
        with pytest.raises(RuntimeError, match="MySpecialNode"):
            inst.soft_delete()

    def test_restore_error_message_includes_class_name(self):
        cls = create_node("MySpecialNode", soft_delete=False)
        inst = cls()
        with pytest.raises(RuntimeError, match="MySpecialNode"):
            inst.restore()

    def test_compute_content_hash_empty_dict(self):
        result = compute_content_hash({})
        expected = hashlib.sha256(
            json.dumps({}, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_compute_content_hash_empty_list(self):
        result = compute_content_hash([])
        expected = hashlib.sha256(
            json.dumps([], sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected

    def test_compute_content_hash_boolean(self):
        h_true = compute_content_hash(True)
        h_false = compute_content_hash(False)
        assert h_true != h_false

    def test_compute_content_hash_float(self):
        result = compute_content_hash(3.14)
        expected = hashlib.sha256(
            json.dumps(3.14, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert result == expected


# ===================================================================
# 8. Manual subclass with node_config ClassVar
# ===================================================================


class TestManualSubclassWithConfig:
    """Test setting node_config as a ClassVar on a hand-written subclass."""

    def test_manual_subclass_lifecycle(self):
        class Article(Node):
            node_config = NodeConfig(
                table_name="articles",
                soft_delete=True,
                versioning=True,
                content_hashing=True,
                track_updated_at=True,
            )

        a = Article(content={"title": "Test"})
        assert Article.node_config.table_name == "articles"

        a.touch(by="editor")
        assert a.metadata["version"] == 1
        assert "updated_at" in a.metadata
        assert a.metadata["updated_by"] == "editor"
        assert "content_hash" in a.metadata

        a.soft_delete(by="admin")
        assert a.metadata["is_deleted"] is True
        assert a.metadata["version"] == 2

        a.restore(by="admin")
        assert a.metadata["is_deleted"] is False
        assert a.metadata["version"] == 3

    def test_manual_subclass_rehash(self):
        class Doc(Node):
            node_config = NodeConfig(content_hashing=True)

        d = Doc(content="hello world")
        h = d.rehash()
        assert h == compute_content_hash("hello world")
        assert d.metadata["content_hash"] == h

    def test_manual_subclass_no_soft_delete(self):
        class ReadOnly(Node):
            node_config = NodeConfig(versioning=True)

        ro = ReadOnly()
        with pytest.raises(RuntimeError):
            ro.soft_delete()

    def test_manual_subclass_no_restore(self):
        class ReadOnly(Node):
            node_config = NodeConfig(versioning=True)

        ro = ReadOnly()
        with pytest.raises(RuntimeError):
            ro.restore()
