"""Face quality and liveness proxies.

These are deliberately *cheap* signals that run on every frame.  They gate
forensic escalation; they are not a certified presentation-attack detector and
the report says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FaceQuality:
    score: float
    sharpness: float
    brightness: float
    contrast: float
    size: float
    pose: float
    yaw: float
    pitch: float
    occlusion: float
    components: dict[str, float] = field(default_factory=dict)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def assess_face(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    landmarks: np.ndarray | None,
    min_face_px: int = 56,
) -> FaceQuality:
    """Score blur, exposure, pose, scale and an occlusion heuristic."""
    x, y, w, h = (int(v) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return FaceQuality(0.0, 0, 0, 0, 0, 0, 0, 0, 1.0, {})

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _clamp01(np.log1p(lap_var) / np.log1p(900.0))
    mean = float(gray.mean())
    brightness = _clamp01(1.0 - abs(mean - 128.0) / 128.0)
    contrast = _clamp01(float(gray.std()) / 64.0)
    size = _clamp01(min(w, h) / (min_face_px * 2.4))

    yaw = pitch = 0.0
    pose = 0.5
    if landmarks is not None and len(landmarks) == 5:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        re, le, nose, rm, lm = pts
        eye_dist = float(np.linalg.norm(le - re)) + 1e-6
        yaw = float((nose[0] - (re[0] + le[0]) / 2.0) / eye_dist)
        pitch = float((nose[1] - (re[1] + le[1]) / 2.0) / eye_dist)
        pose = _clamp01(1.0 - (abs(yaw) / 0.55 + abs(pitch) / 0.65) / 2.0)

    # Occlusion heuristic: asymmetric edge energy between the two face halves
    # plus a collapse of texture in the lower half (masks, hands, scarves).
    left = gray[:, : gray.shape[1] // 2]
    right = gray[:, gray.shape[1] // 2 :]
    edge_l = float(cv2.Sobel(left, cv2.CV_32F, 1, 0).var())
    edge_r = float(cv2.Sobel(right, cv2.CV_32F, 1, 0).var())
    asym = abs(edge_l - edge_r) / (edge_l + edge_r + 1e-6)
    lower = gray[gray.shape[0] // 2 :, :]
    lower_energy = float(cv2.Laplacian(lower, cv2.CV_64F).var()) / (lap_var + 1e-6)
    occlusion = _clamp01(0.65 * asym + 0.35 * (1.0 - _clamp01(lower_energy)))

    score = _clamp01(
        0.34 * sharpness
        + 0.14 * brightness
        + 0.14 * contrast
        + 0.20 * size
        + 0.12 * pose
        + 0.06 * (1.0 - occlusion)
    )
    return FaceQuality(
        score=score,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        size=size,
        pose=pose,
        yaw=yaw,
        pitch=pitch,
        occlusion=occlusion,
        components={
            "laplacian_var": round(lap_var, 3),
            "mean_luma": round(mean, 2),
            "edge_asymmetry": round(asym, 4),
            "lower_texture_ratio": round(lower_energy, 4),
        },
    )


@dataclass
class LivenessResult:
    score: float
    texture: float
    micro_motion: float
    screen_artifact: float
    reason: str


class LivenessEstimator:
    """Temporal texture + micro-motion liveness proxy.

    A live face acquires small non-rigid motion and never sits at exactly zero
    optical flow for long.  A replayed screen often shows a static, very sharp
    patch (or a high-frequency moiré grid).  The score is intentionally
    conservative and only one input to escalation.
    """

    def __init__(self, history: int = 8) -> None:
        self._scores: list[float] = []
        self._history = history

    def update(self, crop: np.ndarray, flow_roi: np.ndarray | None) -> LivenessResult:
        if crop.size == 0:
            return LivenessResult(0.0, 0.0, 0.0, 0.0, "empty")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.size == 0:
            return LivenessResult(0.0, 0.0, 0.0, 0.0, "empty")
        gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
        lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        texture = _clamp01(np.log1p(lap) / np.log1p(1200.0))

        micro = 0.0
        if flow_roi is not None and flow_roi.size:
            mag = np.linalg.norm(flow_roi, axis=-1)
            micro = _clamp01(float(np.percentile(mag, 75)) / 1.8)

        # Moiré / screen grid: a strong isolated high-frequency FFT peak.
        spec = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
        cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
        spec[cy - 4 : cy + 5, cx - 4 : cx + 5] = 0
        total = float(spec.sum()) + 1e-6
        peak = float(spec.max()) / total
        screen = _clamp01((peak - 0.0025) / 0.0075)

        raw = _clamp01(0.34 * texture + 0.46 * micro + 0.20 * (1.0 - screen))
        self._scores.append(raw)
        if len(self._scores) > self._history:
            self._scores.pop(0)
        stable = float(np.median(self._scores))
        if screen > 0.75:
            reason = "possible_screen_replay"
        elif micro < 0.05:
            reason = "no_micro_motion"
        elif stable > 0.55:
            reason = "live_motion"
        else:
            reason = "uncertain"
        return LivenessResult(round(stable, 3), round(texture, 3), round(micro, 3), round(screen, 3), reason)
