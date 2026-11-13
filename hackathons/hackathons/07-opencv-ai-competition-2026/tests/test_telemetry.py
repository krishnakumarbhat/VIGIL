"""Contract tests for the versioned telemetry schema."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from src.telemetry import (
    BBox,
    FaceRecord,
    FrameTelemetry,
    TelemetryWriter,
    embedding_digest,
    summarise_telemetry,
    utc_now,
)


def _record(run_id: str = "run-test", **overrides) -> FrameTelemetry:
    payload = {
        "run_id": run_id,
        "frame_index": 0,
        "timestamp_utc": utc_now(),
        "width": 640,
        "height": 360,
        "processing_ms": 12.5,
        "cpu_ms": 11.0,
        "fps_instant": 80.0,
        "backend": "yunet+sface",
    }
    payload.update(overrides)
    return FrameTelemetry(**payload)


def test_minimal_record_has_stable_schema_version() -> None:
    record = _record()
    assert record.schema_version == "vigil.telemetry/1.0"
    assert record.degraded is False
    assert record.faces == []
    assert record.agent.state_path == []
    assert record.verification.passed is True


def test_face_scores_are_bounded() -> None:
    with pytest.raises(ValidationError):
        FaceRecord(face_id=1, bbox=BBox(x=0, y=0, w=10, h=10), det_score=1.5)
    with pytest.raises(ValidationError):
        _record(processing_ms=-1.0)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(surprise="not in contract")


def test_bbox_iou_and_area() -> None:
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=0, w=10, h=10)
    assert a.area == 100
    assert a.iou(a) == 1.0
    assert a.iou(b) == pytest.approx(50 / 150)
    assert a.iou(BBox(x=100, y=100, w=5, h=5)) == 0.0


def test_embedding_digest_is_deterministic_and_privacy_preserving() -> None:
    vec = np.linspace(-1, 1, 128, dtype=np.float32)
    first = embedding_digest(vec)
    second = embedding_digest(vec.copy())
    assert first == second
    assert first is not None and len(first) == 16
    assert embedding_digest(vec * 3.0) == first  # scale invariant after L2 norm
    assert embedding_digest(np.zeros(128, dtype=np.float32)) is None
    assert embedding_digest(None) is None


def test_writer_roundtrip_and_mismatch(tmp_path) -> None:
    path = tmp_path / "telemetry.jsonl"
    with TelemetryWriter(path, "run-a") as writer:
        writer.write(_record("run-a", frame_index=0))
        writer.write(_record("run-a", frame_index=1))
        assert writer.count == 2
        with pytest.raises(ValueError):
            writer.write(_record("run-b"))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["frame_index"] for line in lines] == [0, 1]

    summary = summarise_telemetry(path)
    assert summary["frames"] == 2
    assert summary["p50_ms"] == 12.5
    assert summary["backends"] == ["yunet+sface"]


def test_summary_of_missing_or_empty_file(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert summarise_telemetry(empty) == {"frames": 0}
