"""Cloud contract: offline-first behaviour and guarded AWS construction."""

from __future__ import annotations

import numpy as np
import pytest

from src.cloud import AwsCloud, OfflineCloud, build_cloud, upload_evidence
from src.config import VisionConfig


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_offline_cloud_is_deterministic_and_shape_compatible(cfg: VisionConfig, hit_factory) -> None:
    cloud = OfflineCloud(cfg)
    faces = [hit_factory(identity="alice"), hit_factory(face_id=2)]
    frame = _frame()
    first = cloud(frame, faces, "unit-test")
    second = cloud(frame, faces, "unit-test")
    assert cloud.calls == 2
    assert first.provider == "offline-mock"
    assert first.incident_id == second.incident_id
    assert first.rekognition_faces == 2
    assert first.matched is True
    assert "No biometric payload" in (first.bedrock_summary or "")


def test_offline_cloud_without_faces(cfg: VisionConfig) -> None:
    record = OfflineCloud(cfg)(_frame(), [], "empty")
    assert record.matched is None
    assert record.incident_id is not None


def test_build_cloud_falls_back_when_aws_disabled(cfg: VisionConfig) -> None:
    assert isinstance(build_cloud(cfg), OfflineCloud)


def test_aws_cloud_requires_explicit_enablement(cfg: VisionConfig) -> None:
    with pytest.raises(ValueError):
        AwsCloud(cfg)


def test_upload_evidence_is_offline_by_default(cfg: VisionConfig, tmp_path) -> None:
    path = tmp_path / "incident.json"
    path.write_text("{}")
    assert upload_evidence(path, cfg) is None


def test_upload_evidence_needs_a_bucket(cfg: VisionConfig, tmp_path) -> None:
    enabled = cfg.model_copy(update={"aws_enabled": True, "aws_bucket": None})
    path = tmp_path / "incident.json"
    path.write_text("{}")
    assert upload_evidence(path, enabled) is None
