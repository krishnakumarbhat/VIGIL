"""Typed configuration for the VIGIL-5 vision engine.

All knobs live in one validated object so benchmarks, the CLI, the Lambda
handler and the tests share exactly the same defaults.  Values can be
overridden by a JSON file (``VIGIL_CONFIG``) or by ``VIGIL_*`` environment
variables; secrets are never read here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TRACK_ROOT = Path(__file__).resolve().parents[1]


class VisionConfig(BaseModel):
    """Runtime configuration with conservative, CPU-friendly defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- models -----------------------------------------------------------
    detector_model: Path = TRACK_ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
    recognizer_model: Path = TRACK_ROOT / "data" / "models" / "face_recognition_sface_2021dec.onnx"
    allow_model_download: bool = False

    # --- perception -------------------------------------------------------
    det_threshold: float = Field(0.60, ge=0.05, le=0.99)
    nms_threshold: float = Field(0.30, ge=0.05, le=0.95)
    max_faces: int = Field(12, ge=1, le=128)
    min_face_px: int = Field(56, ge=16, le=512)
    recognition_threshold: float = Field(0.363, ge=0.0, le=1.0)
    enable_liveness: bool = True
    embed_stride: int = Field(3, ge=1, le=60, description="reuse embeddings for N frames per track")
    flow_scale: float = Field(0.4, gt=0.1, le=1.0)
    motion_anomaly_z: float = Field(3.0, ge=1.0, le=20.0)

    # --- agent ------------------------------------------------------------
    base_resolution: int = Field(640, ge=128, le=1920)
    resolution_ladder: tuple[int, ...] = (320, 480, 640, 768)
    latency_budget_ms: float = Field(100.0, ge=5.0, le=5000.0)
    max_adaptive_steps: int = Field(2, ge=0, le=4)
    forensic_max_side: int = Field(640, ge=224, le=2048)
    enable_forensics: bool = True
    anomaly_trigger_z: float = Field(2.5, ge=0.5, le=20.0)
    quality_floor: float = Field(0.42, ge=0.0, le=1.0)
    det_confidence_floor: float = Field(0.72, ge=0.0, le=1.0)
    unknown_identity_escalation: bool = True
    require_consent_for_enrollment: bool = True
    seed: int = 20261026

    # --- output / privacy -------------------------------------------------
    telemetry_dir: Path = TRACK_ROOT / "outputs" / "telemetry"
    store_embeddings: bool = False
    redact_identity_labels: bool = False
    jpeg_quality: int = Field(90, ge=30, le=100)
    draw_overlay: bool = True

    # --- AWS --------------------------------------------------------------
    aws_enabled: bool = False
    aws_region: str = "us-east-1"
    aws_bucket: str | None = None
    aws_mock: bool = True
    rekognition_confidence: float = Field(90.0, ge=50.0, le=100.0)
    bedrock_model_id: str = "amazon.nova-lite-v1:0"

    # --- runtime ----------------------------------------------------------
    threads: int = Field(1, ge=0, le=64, description="0 keeps the OpenCV default; 1 minimises CPU time per frame and keeps the CPU budget deterministic; 2 maximises wall throughput at ~13% more CPU per frame")
    backend: Literal["auto", "cpu", "opencl"] = "auto"

    @field_validator("resolution_ladder")
    @classmethod
    def _ladder_sorted(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or list(value) != sorted(set(value)):
            raise ValueError("resolution_ladder must be strictly increasing")
        return value

    @model_validator(mode="after")
    def _base_in_ladder(self) -> "VisionConfig":
        if self.base_resolution not in self.resolution_ladder:
            object.__setattr__(
                self,
                "resolution_ladder",
                tuple(sorted({*self.resolution_ladder, self.base_resolution})),
            )
        return self


def load_config(path: str | Path | None = None) -> VisionConfig:
    """Load config from defaults, then JSON, then ``VIGIL_*`` env overrides."""
    payload: dict = {}
    cfg_path = path or os.environ.get("VIGIL_CONFIG")
    if cfg_path:
        payload.update(json.loads(Path(cfg_path).read_text()))

    env_map = {
        "VIGIL_BASE_RESOLUTION": ("base_resolution", int),
        "VIGIL_LATENCY_BUDGET_MS": ("latency_budget_ms", float),
        "VIGIL_ENABLE_FORENSICS": ("enable_forensics", lambda v: v.lower() in {"1", "true", "yes"}),
        "VIGIL_AWS_ENABLED": ("aws_enabled", lambda v: v.lower() in {"1", "true", "yes"}),
        "VIGIL_AWS_MOCK": ("aws_mock", lambda v: v.lower() in {"1", "true", "yes"}),
        "VIGIL_AWS_BUCKET": ("aws_bucket", str),
        "VIGIL_AWS_REGION": ("aws_region", str),
        "VIGIL_TELEMETRY_DIR": ("telemetry_dir", Path),
        "VIGIL_DETECTOR_MODEL": ("detector_model", Path),
        "VIGIL_RECOGNIZER_MODEL": ("recognizer_model", Path),
        "VIGIL_THREADS": ("threads", int),
        "VIGIL_STORE_EMBEDDINGS": ("store_embeddings", lambda v: v.lower() in {"1", "true", "yes"}),
    }
    for env_key, (field, caster) in env_map.items():
        if env_key in os.environ:
            payload[field] = caster(os.environ[env_key])
    return VisionConfig(**payload)
