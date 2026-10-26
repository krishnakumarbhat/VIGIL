"""Shared fixtures for the VIGIL-5 test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VisionConfig  # noqa: E402
from src.perception.face import FaceHit, FacePipeline  # noqa: E402
from src.perception.quality import FaceQuality  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture()
def cfg(tmp_path: Path) -> VisionConfig:
    return VisionConfig(telemetry_dir=tmp_path / "telemetry")


@pytest.fixture(scope="session")
def pipeline() -> FacePipeline:
    return FacePipeline(VisionConfig())


@pytest.fixture(scope="session")
def lena(root: Path) -> np.ndarray:
    image = cv2.imread(str(root / "data" / "samples" / "faces" / "src_lena_36fe10b1.jpg"))
    assert image is not None, "sample portrait missing"
    return image


@pytest.fixture(scope="session")
def messi(root: Path) -> np.ndarray:
    image = cv2.imread(str(root / "data" / "samples" / "faces" / "src_messi5_ee5e7209.jpg"))
    assert image is not None, "sample portrait missing"
    return image


def make_hit(
    box: tuple[int, int, int, int] = (40, 30, 96, 96),
    score: float = 0.9,
    identity: str | None = None,
    quality: float = 0.8,
    liveness: float = 0.9,
    face_id: int = 1,
) -> FaceHit:
    """Build a deterministic FaceHit without touching the models."""
    q = FaceQuality(
        score=quality,
        sharpness=quality,
        brightness=0.8,
        contrast=0.7,
        size=0.9,
        pose=0.9,
        yaw=0.0,
        pitch=0.0,
        occlusion=0.1,
    )
    return FaceHit(
        face_id=face_id,
        box=box,
        score=score,
        quality=q,
        liveness=liveness,
        identity=identity,
        identity_score=0.8 if identity else None,
    )


@pytest.fixture()
def hit_factory():
    return make_hit
