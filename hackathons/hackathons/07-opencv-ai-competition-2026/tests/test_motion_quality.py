"""DSP tests for optical flow, anomaly scoring, quality and liveness."""

from __future__ import annotations

import cv2
import numpy as np

from src.perception.motion import MotionAnalyzer
from src.perception.quality import LivenessEstimator, assess_face


def _textured_scene(width: int = 320, height: int = 240, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 110, dtype=np.int16)
    noise = rng.integers(-10, 11, (height, width, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_static_scene_has_no_motion() -> None:
    frame = _textured_scene()
    analyzer = MotionAnalyzer(scale=0.5)
    reports = [analyzer.update(frame, idx) for idx in range(20)]
    assert reports[0].warmup is True
    assert all(not report.moving for report in reports)
    assert max(report.energy for report in reports) < 0.5


def test_moving_object_produces_roi_and_flow() -> None:
    analyzer = MotionAnalyzer(scale=0.5, anomaly_z=3.0)
    reports = []
    for idx in range(40):
        frame = _textured_scene()
        if idx >= 15:  # establish a static baseline first, then move
            x = 20 + (idx - 15) * 6
            cv2.rectangle(frame, (x, 70), (x + 70, 140), (240, 240, 240), -1)
            cv2.putText(frame, "MOVE", (x + 5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        reports.append(analyzer.update(frame, idx))
    assert any(report.moving for report in reports[18:]), "moving square was never flagged"
    assert reports[-1].roi is not None
    assert reports[-1].foreground_ratio > 0.0
    assert reports[-1].energy > 0.5
    flow = analyzer.flow_in((20, 70, 70, 70), (240, 320, 3))
    assert flow is not None and flow.ndim == 3 and flow.shape[-1] == 2


def test_reset_clears_state() -> None:
    analyzer = MotionAnalyzer(scale=0.5)
    frame = _textured_scene()
    for idx in range(8):
        analyzer.update(frame, idx)
    analyzer.reset()
    assert analyzer.warm is False
    assert analyzer.update(frame, 9).warmup is True


def test_quality_blank_is_low() -> None:
    blank = np.zeros((160, 160, 3), dtype=np.uint8)
    quality = assess_face(blank, (20, 20, 110, 110), None, min_face_px=56)
    assert quality.score < 0.35
    assert quality.sharpness == 0.0
    assert quality.brightness == 0.0
    assert 0.0 <= quality.occlusion <= 1.0
    assert "laplacian_var" in quality.components


def test_quality_prefers_sharp_texture() -> None:
    sharp = np.zeros((160, 160, 3), dtype=np.uint8)
    for y in range(0, 160, 8):
        for x in range(0, 160, 8):
            if (x // 8 + y // 8) % 2 == 0:
                sharp[y : y + 8, x : x + 8] = 230
    blurred = cv2.GaussianBlur(sharp, (21, 21), 6.0)
    box = (20, 20, 120, 120)
    sharp_q = assess_face(sharp, box, None, min_face_px=56)
    blurred_q = assess_face(blurred, box, None, min_face_px=56)
    assert sharp_q.sharpness > blurred_q.sharpness
    assert sharp_q.score > blurred_q.score


def test_quality_pose_uses_landmarks() -> None:
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    centered = np.array(
        [[70, 70], [130, 70], [100, 100], [80, 130], [120, 130]], dtype=np.float32
    )
    turned = centered.copy()
    turned[2, 0] += 25
    base = assess_face(frame, (50, 50, 100, 100), centered, min_face_px=56)
    rotated = assess_face(frame, (50, 50, 100, 100), turned, min_face_px=56)
    assert abs(base.yaw) < abs(rotated.yaw)
    assert rotated.pose < base.pose


def test_empty_face_crop_is_safe() -> None:
    quality = assess_face(np.zeros((16, 16, 3), np.uint8), (30, 30, 5, 5), None)
    assert quality.score == 0.0


def test_liveness_static_and_moving() -> None:
    estimator = LivenessEstimator()
    empty = estimator.update(np.zeros((0, 0, 3), np.uint8), None)
    assert empty.reason == "empty" and empty.score == 0.0

    frame = _textured_scene()
    crop = frame[40:140, 40:140]
    static = estimator.update(crop, np.zeros((50, 50, 2), np.float32))
    assert 0.0 <= static.score <= 1.0
    assert static.reason in {"no_micro_motion", "uncertain", "partial"}

    flow = np.ones((50, 50, 2), np.float32)
    moving = estimator.update(crop, flow)
    assert moving.micro_motion > static.micro_motion
    assert moving.score >= static.score
