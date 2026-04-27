# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Security regression tests for FlowAgent.id / FlowOp.id path containment.

PR review 2026-04-24 flagged a HIGH finding: model-controlled agent ids
became filesystem path segments via ``RunDir.agent_artifact_dir``. An id
like ``../../tmp/evil`` or ``/etc`` would escape the artifact root.

Two layers of defense are now in place:
  1. Pydantic `field_validator` on FlowAgent.id / FlowOp.id rejects any
     identifier that doesn't match ``^[A-Za-z0-9_-]{1,64}$``.
  2. RunDir.agent_artifact_dir rejects components containing path
     separators, leading dots, or resolved paths outside the root.

Both layers are tested independently so either alone provides protection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lionagi.cli._runs import RunDir
from lionagi.cli.orchestrate.flow import FlowAgent, FlowOp

# ── Layer 1: Pydantic field_validator ────────────────────────────────


class TestFlowAgentIdValidation:
    def test_valid_id_accepted(self):
        a = FlowAgent(id="r1", role="researcher")
        assert a.id == "r1"

    def test_valid_id_with_dash_and_underscore(self):
        a = FlowAgent(id="impl-1_v2", role="implementer")
        assert a.id == "impl-1_v2"

    def test_rejects_path_separator(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="evil/nested", role="researcher")

    def test_rejects_backslash(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="evil\\win", role="researcher")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="/etc/passwd", role="researcher")

    def test_rejects_parent_directory(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="..", role="researcher")

    def test_rejects_traversal(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="../../tmp/evil", role="researcher")

    def test_rejects_leading_dot(self):
        with pytest.raises(ValidationError):
            FlowAgent(id=".hidden", role="researcher")

    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="", role="researcher")

    def test_rejects_too_long(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="a" * 65, role="researcher")

    def test_accepts_max_length(self):
        a = FlowAgent(id="a" * 64, role="researcher")
        assert len(a.id) == 64

    def test_rejects_space(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="has space", role="researcher")

    def test_rejects_unicode(self):
        with pytest.raises(ValidationError):
            FlowAgent(id="café", role="researcher")


class TestFlowOpIdValidation:
    def test_valid_ids_accepted(self):
        op = FlowOp(id="o1", agent_id="r1", instruction="do thing")
        assert op.id == "o1"
        assert op.agent_id == "r1"

    def test_rejects_bad_op_id(self):
        with pytest.raises(ValidationError):
            FlowOp(id="../escape", agent_id="r1", instruction="x")

    def test_rejects_bad_agent_id(self):
        with pytest.raises(ValidationError):
            FlowOp(id="o1", agent_id="/abs/path", instruction="x")


# ── Layer 2: RunDir.agent_artifact_dir defense-in-depth ─────────────


class TestAgentArtifactDirContainment:
    @pytest.fixture
    def rundir(self, tmp_path: Path) -> RunDir:
        # Minimal RunDir — only artifact_root is used by agent_artifact_dir
        artifact_root = tmp_path / "runs" / "r1" / "artifacts"
        artifact_root.mkdir(parents=True)
        state_root = tmp_path / "runs" / "r1"
        return RunDir(
            run_id="r1",
            state_root=state_root,
            artifact_root=artifact_root,
        )

    def test_safe_id_resolves(self, rundir: RunDir):
        path = rundir.agent_artifact_dir("impl1")
        assert path == rundir.artifact_root / "impl1"

    def test_rejects_traversal(self, rundir: RunDir):
        with pytest.raises(ValueError, match="safe path component"):
            rundir.agent_artifact_dir("../evil")

    def test_rejects_absolute(self, rundir: RunDir):
        with pytest.raises(ValueError, match="safe path component"):
            rundir.agent_artifact_dir("/etc/passwd")

    def test_rejects_backslash(self, rundir: RunDir):
        with pytest.raises(ValueError, match="safe path component"):
            rundir.agent_artifact_dir("a\\b")

    def test_rejects_empty(self, rundir: RunDir):
        with pytest.raises(ValueError, match="safe path component"):
            rundir.agent_artifact_dir("")

    def test_rejects_dot(self, rundir: RunDir):
        with pytest.raises(ValueError, match="safe path component"):
            rundir.agent_artifact_dir(".")

    def test_rejects_non_string(self, rundir: RunDir):
        with pytest.raises(ValueError, match="safe path component"):
            rundir.agent_artifact_dir(None)  # type: ignore[arg-type]
