# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for V0 compatibility with V1 Observable Protocol.

Validates that existing V0 components satisfy the new V1 protocol and
maintain their V0 inheritance structure without breaking changes.
"""

from uuid import UUID

import pytest

# Import V0 Components (covering the core ecosystem)
from lionagi.protocols.generic.element import Element, IDType
from lionagi.protocols.generic.event import Event
from lionagi.protocols.generic.log import Log
from lionagi.protocols.generic.pile import Pile
from lionagi.protocols.generic.progression import Progression
from lionagi.protocols.ids import canonical_id, to_uuid

# Import via the bridge (V1 preferred path)
from lionagi.protocols.types import LegacyObservable, Observable

# Additional V0 components from ecosystem analysis
try:
    from lionagi.protocols.action.tool import Tool
    from lionagi.protocols.forms.report import BaseForm
    from lionagi.protocols.graph.edge import Edge
    from lionagi.protocols.graph.node import Node
    from lionagi.protocols.mail.mail import Mail

    EXTENDED_COMPONENTS = [BaseForm, Mail, Node, Edge, Tool]
except ImportError:
    # Some components might not be available in all environments
    EXTENDED_COMPONENTS = []


class TestV0StructuralCompatibility:
    """Test V0 components structurally satisfy V1 Observable Protocol."""

    @pytest.mark.parametrize("v0_class", [Element, Event, Pile, Progression])
    def test_v0_components_satisfy_v1_observable(self, v0_class):
        """Verify core V0 components structurally satisfy the V1 Observable Protocol."""
        instance = v0_class()

        # The core structural compatibility check (V1 Protocol)
        assert isinstance(
            instance, Observable
        ), f"{v0_class.__name__} should satisfy V1 Observable Protocol"
        assert hasattr(
            instance, "id"
        ), f"{v0_class.__name__} should have 'id' attribute"
        assert (
            instance.id is not None
        ), f"{v0_class.__name__}.id should not be None"

        # Verify the ID is an IDType (V0 convention)
        assert isinstance(
            instance.id, IDType
        ), f"{v0_class.__name__}.id should be IDType"

    def test_v0_log_satisfies_v1_observable(self):
        """Specific test for Log which requires content for instantiation."""
        instance = Log(content={"test": "data"})

        assert isinstance(instance, Observable)
        assert hasattr(instance, "id")
        assert instance.id is not None
        assert isinstance(instance.id, IDType)

    @pytest.mark.parametrize("v0_class", EXTENDED_COMPONENTS)
    def test_extended_v0_components_satisfy_v1_observable(self, v0_class):
        """Test extended V0 ecosystem components satisfy V1 Protocol."""
        try:
            if v0_class.__name__ == "Tool":
                # Tool might need specific initialization
                instance = v0_class(function=lambda: None, name="test_tool")
            else:
                instance = v0_class()
        except Exception:
            pytest.skip(f"Could not instantiate {v0_class.__name__}")

        assert isinstance(
            instance, Observable
        ), f"{v0_class.__name__} should satisfy V1 Observable Protocol"
        assert hasattr(instance, "id")
        assert instance.id is not None

    def test_all_v0_elements_have_consistent_id_structure(self):
        """Test that all V0 Element subclasses have consistent ID structure."""
        instances = [
            Element(),
            Event(),
            Log(content={"test": "value"}),
            Pile(),
            Progression(),
        ]

        for instance in instances:
            # All should satisfy V1 Protocol
            assert isinstance(instance, Observable)

            # All should have IDType with internal UUID
            assert isinstance(instance.id, IDType)
            assert hasattr(instance.id, "_id")
            assert isinstance(instance.id._id, UUID)


class TestV0NominalTypingRegression:
    """Ensure V0 components still inherit from V0 ABC (critical for V0 logic)."""

    def test_v0_element_nominal_typing_regression(self):
        """Ensure V0 Element still inherits from the V0 ABC (LegacyObservable)."""
        # Check inheritance (nominal typing) - critical for Pile and other V0 logic
        assert issubclass(
            Element, LegacyObservable
        ), "Element must still inherit from V0 LegacyObservable ABC"
        assert isinstance(
            Element(), LegacyObservable
        ), "Element instances must still be instances of LegacyObservable"

    def test_all_element_subclasses_maintain_nominal_typing(self):
        """All Element subclasses should maintain V0 nominal typing."""
        v0_classes = [Event, Log, Pile, Progression]

        for v0_class in v0_classes:
            assert issubclass(
                v0_class, LegacyObservable
            ), f"{v0_class.__name__} must still inherit from LegacyObservable"

            # Test instance creation and nominal typing
            if v0_class == Log:
                instance = v0_class(content={"test": "data"})
            else:
                instance = v0_class()

            assert isinstance(
                instance, LegacyObservable
            ), f"{v0_class.__name__} instances must still be LegacyObservable"

    def test_dual_typing_compatibility(self):
        """Test that V0 components satisfy both V1 Protocol and V0 ABC."""
        element = Element()

        # Should satisfy both typing systems simultaneously
        assert isinstance(element, Observable), "Should satisfy V1 Protocol"
        assert isinstance(element, LegacyObservable), "Should satisfy V0 ABC"

        # They should be different objects
        assert (
            Observable is not LegacyObservable
        ), "Should be distinct typing systems"


class TestConsumerSideIdNormalization:
    """Test normalization strategies for V0 IDType to UUID (V1 consumer pattern)."""

    def test_optimized_id_normalization_element(self):
        """Test optimized normalization for Element via direct _id access."""
        element = Element()
        observable: Observable = element  # Treat as V1 Observable

        obs_id = observable.id
        assert isinstance(obs_id, IDType)

        # Strategy 1: Optimized direct access (duck typing IDType._id)
        if hasattr(obs_id, "_id") and isinstance(obs_id._id, UUID):
            canonical_uuid_direct = obs_id._id
            assert isinstance(canonical_uuid_direct, UUID)
        else:
            pytest.fail(
                "V0 IDType structure changed; optimized normalization failed."
            )

        # Strategy 2: to_uuid utility (should use optimized path)
        canonical_via_utility = to_uuid(obs_id)
        assert canonical_uuid_direct == canonical_via_utility

    def test_canonical_id_utility_consistency(self):
        """Test canonical_id utility works consistently across V0 components."""
        instances = [
            Element(),
            Event(),
            Log(content={"test": "data"}),
            Pile(),
            Progression(),
        ]

        for instance in instances:
            canonical_uuid = canonical_id(instance)
            assert isinstance(
                canonical_uuid, UUID
            ), f"canonical_id({type(instance).__name__}) should return UUID"

            # Should be consistent with direct access
            direct_uuid = instance.id._id
            assert (
                canonical_uuid == direct_uuid
            ), f"canonical_id should match direct _id access for {type(instance).__name__}"

    def test_to_uuid_handles_all_v0_id_types(self):
        """Test to_uuid utility handles various V0 ID representations."""
        element = Element()

        # Test different input types
        test_cases = [
            element,  # Element with .id
            element.id,  # IDType directly
            element.id._id,  # UUID directly
            str(element.id),  # String representation
        ]

        expected_uuid = element.id._id

        for test_case in test_cases:
            result = to_uuid(test_case)
            assert isinstance(
                result, UUID
            ), f"to_uuid should return UUID for {type(test_case)}"
            assert (
                result == expected_uuid
            ), f"to_uuid should return consistent UUID for {type(test_case)}"

    def test_roundtrip_consistency(self):
        """Test roundtrip Element -> canonical_id -> canonical_id is consistent."""
        elements = [Element(), Event(), Log(content={"test": "value"})]

        for element in elements:
            canonical = canonical_id(element)
            # Converting UUID again should yield same result
            roundtrip = canonical_id(canonical)
            assert (
                canonical == roundtrip
            ), f"Roundtrip should be consistent for {type(element).__name__}"
