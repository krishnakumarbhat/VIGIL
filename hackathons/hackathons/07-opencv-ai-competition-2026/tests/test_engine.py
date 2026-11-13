"""Engine integration: gallery consent, frame processing, annotation, run summary."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.cloud import OfflineCloud
from src.config import VisionConfig
from src.engine import VisionEngine
from src.telemetry import FrameTelemetry


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> VisionEngine:
    cfg = VisionConfig(telemetry_dir=tmp_path_factory.mktemp("telemetry"), threads=1)
    return VisionEngine(cfg, cloud=OfflineCloud(cfg), run_id="run-engine")


def test_process_frame_and_annotate(engine: VisionEngine, lena: np.ndarray) -> None:
    outcome = engine.process_frame(lena, source="unit-test", index=3)
    assert isinstance(outcome.telemetry, FrameTelemetry)
    assert outcome.telemetry.source == "unit-test"
    assert outcome.telemetry.frame_index == 3
    canvas = engine.annotate(lena, outcome)
    assert canvas.shape == lena.shape
    assert canvas is not lena


def test_gallery_requires_consent(tmp_path, engine: VisionEngine, lena: np.ndarray) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    cv2.imwrite(str(gallery / "alice.jpg"), lena)

    assert engine.enroll_gallery(gallery) == []
    (gallery / "consent.json").write_text(json.dumps({"labels": ["alice"]}))
    assert engine.enroll_gallery(gallery) == ["alice"]
    assert engine.pipeline.gallery_size >= 1


def test_run_image_dir_writes_valid_telemetry(tmp_path, engine: VisionEngine, lena: np.ndarray) -> None:
    images = tmp_path / "images"
    images.mkdir()
    assert cv2.imwrite(str(images / "frame.jpg"), lena)
    telemetry = tmp_path / "run.jsonl"
    summary = engine.run_image_dir(images, telemetry_path=telemetry)
    assert summary["frames"] == 1
    assert summary["faces"] >= 1
    lines = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert len(lines) == 1
    FrameTelemetry(**lines[0])


def test_missing_video_raises(tmp_path, engine: VisionEngine) -> None:
    with pytest.raises(FileNotFoundError):
        engine.run_video(tmp_path / "nope.mp4")
