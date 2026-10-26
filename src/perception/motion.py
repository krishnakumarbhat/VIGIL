"""Optical-flow motion analysis and anomaly scoring.

OpenCV 5 pieces used here: Farneback dense flow, phase correlation for global
camera motion, MOG2 background subtraction for ROI proposals.  The analyser
runs at half resolution and only upsamples a small ROI hint, keeping the
per-frame cost low enough for the agent's latency budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MotionReport:
    energy: float = 0.0
    residual: float = 0.0
    global_dx: float = 0.0
    global_dy: float = 0.0
    coherent_ratio: float = 0.0
    anomaly_z: float = 0.0
    moving: bool = False
    roi: tuple[int, int, int, int] | None = None
    foreground_ratio: float = 0.0
    warmup: bool = True
    debug: dict[str, float] = field(default_factory=dict)


class MotionAnalyzer:
    """Dense-flow motion analysis with an EMA anomaly baseline."""

    def __init__(
        self,
        scale: float = 0.5,
        moving_threshold: float = 0.55,
        anomaly_z: float = 3.0,
        ema_alpha: float = 0.08,
    ) -> None:
        self.scale = float(scale)
        self.moving_threshold = float(moving_threshold)
        self.anomaly_z = float(anomaly_z)
        self.alpha = float(ema_alpha)
        self._prev: np.ndarray | None = None
        self._flow: np.ndarray | None = None
        self._mean = 0.0
        self._var = 1.0
        self._warm = 0
        self._hist: deque[float] = deque(maxlen=90)
        self._mog = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=24, detectShadows=False)

    def reset(self) -> None:
        self._prev = None
        self._flow = None
        self._mean = 0.0
        self._var = 1.0
        self._warm = 0
        self._hist: deque[float] = deque(maxlen=90)
        self._mog = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=24, detectShadows=False)

    def update(self, frame: np.ndarray, frame_index: int = 0) -> MotionReport:
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self._mog.apply(gray)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_ratio = float(np.count_nonzero(fg)) / fg.size
        roi = self._foreground_roi(fg)

        if self._prev is None:
            self._prev = gray
            self._warm = 1
            return MotionReport(roi=roi, foreground_ratio=fg_ratio, warmup=True)

        flow = cv2.calcOpticalFlowFarneback(
            self._prev, gray, None, pyr_scale=0.5, levels=3, winsize=17,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._flow = flow
        median = np.median(flow.reshape(-1, 2), axis=0)
        residual_vec = flow - median.reshape(1, 1, 2)
        residual_mag = np.linalg.norm(residual_vec, axis=-1)
        residual = float(np.mean(residual_mag))
        energy = float(np.percentile(residual_mag, 95))

        mag = np.linalg.norm(flow, axis=-1)
        active = mag > 1.2
        coherent = 0.0
        if np.count_nonzero(active) > 16:
            angles = np.arctan2(flow[..., 1][active], flow[..., 0][active])
            hist, _ = np.histogram(angles, bins=12, range=(-np.pi, np.pi))
            coherent = float(hist.max() / max(1, hist.sum()))

        dx = dy = 0.0
        try:
            h, w = gray.shape
            win = cv2.createHanningWindow((w, h), cv2.CV_32F)
            (sx, sy), _response = cv2.phaseCorrelate(self._prev.astype(np.float32), gray.astype(np.float32), win)
            dx, dy = float(sx) / self.scale, float(sy) / self.scale
        except cv2.error:
            pass

        self._warm += 1
        warm = len(self._hist) < 12
        if not warm:
            hist = np.asarray(self._hist, dtype=np.float32)
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med)))
            z = (energy - med) / (1.4826 * mad + 0.08)
            moving = energy > med * 1.55 + 0.25 and (z > 0.8 or coherent > 0.55)
        else:
            med, z = 0.0, 0.0
            moving = False
        self._mean = med if not warm else energy
        self._hist.append(energy)

        self._prev = gray
        return MotionReport(
            energy=round(energy, 4),
            residual=round(residual, 4),
            global_dx=round(dx, 3),
            global_dy=round(dy, 3),
            coherent_ratio=round(coherent, 3),
            anomaly_z=round(float(z), 3),
            moving=bool(moving),
            roi=roi,
            foreground_ratio=round(fg_ratio, 4),
            warmup=bool(warm),
            debug={"median_dx": round(float(median[0]), 3), "median_dy": round(float(median[1]), 3)},
        )

    def _foreground_roi(self, fg: np.ndarray) -> tuple[int, int, int, int] | None:
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 90:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if area > best_area:
                best_area, best = area, (x, y, w, h)
        if best is None:
            return None
        inv = 1.0 / self.scale
        x, y, w, h = best
        return (int(x * inv), int(y * inv), int(w * inv), int(h * inv))

    def flow_in(self, box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> np.ndarray | None:
        """Return the flow field inside a full-resolution box (work-scale crop)."""
        if self._flow is None:
            return None
        x, y, w, h = (int(v) for v in box)
        s = self.scale
        x0, y0 = max(0, int(x * s)), max(0, int(y * s))
        x1 = min(self._flow.shape[1], int((x + w) * s))
        y1 = min(self._flow.shape[0], int((y + h) * s))
        if x1 <= x0 or y1 <= y0:
            return None
        return self._flow[y0:y1, x0:x1]

    @property
    def warm(self) -> bool:
        return self._warm >= 6
