"""Perception tests: model path, fallback path, embeddings and consent."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import VisionConfig
from src.perception.face import FaceHit, FacePipeline


def test_model_backends_and_degradation(pipeline: FacePipeline) -> None:
    assert pipeline.detector_backend == "yunet"
    assert pipeline.recognizer_backend == "sface"
    assert pipeline.degraded is False
    assert pipeline.descriptor_dim == 128


def test_detects_faces_with_in_bounds_boxes_and_landmarks(pipeline: FacePipeline, lena: np.ndarray) -> None:
    hits = pipeline.detect(lena, target_side=640)
    assert hits, "expected at least one face"
    box, score, landmarks = max(hits, key=lambda t: t[1])
    x, y, w, h = box
    assert score >= 0.6
    assert 0 <= x < lena.shape[1] and 0 <= y < lena.shape[0]
    assert x + w <= lena.shape[1] + 2 and y + h <= lena.shape[0] + 2
    assert landmarks is not None and landmarks.shape == (5, 2)


def test_roi_boxes_are_mapped_back_to_full_frame(pipeline: FacePipeline, lena: np.ndarray) -> None:
    roi = (20, 15, 180, 200)
    hits = pipeline.detect(lena, roi=roi, target_side=640)
    assert hits
    for (x, y, w, h), _score, _ in hits:
        cx, cy = x + w / 2, y + h / 2
        assert roi[0] - 2 <= cx <= roi[0] + roi[2] + 2
        assert roi[1] - 2 <= cy <= roi[1] + roi[3] + 2


def test_embedding_is_unit_norm_and_repeatable(pipeline: FacePipeline, lena: np.ndarray) -> None:
    pipeline.reset_tracking()
    first = pipeline.analyze(lena, embed=True, target_side=640)[0]
    assert first.embedding is not None
    assert float(np.linalg.norm(first.embedding)) == pytest.approx(1.0, abs=1e-5)
    pipeline.reset_tracking()
    second = pipeline.analyze(lena, embed=True, target_side=640)[0]
    assert float(np.dot(first.embedding, second.embedding)) > 0.99


def test_different_identities_are_separable(pipeline: FacePipeline, lena: np.ndarray, messi: np.ndarray) -> None:
    pipeline.reset_tracking()
    a = pipeline.analyze(lena, embed=True, target_side=640)[0].embedding
    pipeline.reset_tracking()
    b = pipeline.analyze(messi, embed=True, target_side=640)[0].embedding
    assert a is not None and b is not None
    assert float(np.dot(a, b)) < 0.95


def test_tracker_keeps_stable_ids(pipeline: FacePipeline, lena: np.ndarray) -> None:
    pipeline.reset_tracking()
    first = pipeline.analyze(lena, embed=False, target_side=640)
    second = pipeline.analyze(lena, embed=False, target_side=640)
    assert [h.face_id for h in first] == [h.face_id for h in second]


def test_gallery_requires_consent_and_matches(pipeline: FacePipeline, lena: np.ndarray) -> None:
    ref = FacePipeline(VisionConfig())
    ref.reset_tracking()
    vec = ref.analyze(lena, embed=True, target_side=640)[0].embedding
    assert vec is not None
    assert ref.enroll("refused", vec, consent=False) is False
    assert ref.gallery_size == 0
    assert ref.enroll("lena", vec, consent=True) is True
    label, score = ref.match(vec)
    assert label == "lena"
    assert score is not None and score > 0.99
    assert ref.enroll("zero", np.zeros(128, dtype=np.float32), consent=True) is False


def test_fallback_chain_without_models(tmp_path) -> None:
    cfg = VisionConfig(
        detector_model=tmp_path / "missing_detector.onnx",
        recognizer_model=tmp_path / "missing_recognizer.onnx",
    )
    fallback = FacePipeline(cfg)
    assert fallback.detector_backend == "skin"
    assert fallback.recognizer_backend == "lbp"
    assert fallback.degraded is True
    assert fallback.descriptor_dim == 1024
    assert any("missing" in note for note in fallback.notes)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert fallback.analyze(frame, embed=True, target_side=320) == []

    hit = FaceHit(face_id=1, box=(80, 40, 120, 120), score=0.5)
    vec = fallback.embed(frame, hit)
    assert vec is not None
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_rejects_non_bgr_frame(pipeline: FacePipeline) -> None:
    with pytest.raises(ValueError):
        pipeline.detect(np.zeros((100, 100), dtype=np.uint8))


def test_analyze_returns_empty_on_blank_frame(pipeline: FacePipeline) -> None:
    pipeline.reset_tracking()
    hits = pipeline.analyze(np.zeros((240, 320, 3), dtype=np.uint8), target_side=320)
    assert hits == []
