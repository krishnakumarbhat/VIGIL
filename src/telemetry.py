"""Structured telemetry for the autonomous vision loop.

One JSON object is emitted per frame (JSON Lines).  The schema is validated by
Pydantic before it is written so downstream AWS consumers, the benchmark and
the tests all agree on the contract.  Raw embeddings are never serialised by
default: only a truncated digest is stored.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "vigil.telemetry/1.0"


class AgentState(str, Enum):
    SEE = "SEE"
    REASON = "REASON"
    ACT = "ACT"
    VERIFY = "VERIFY"
    ESCALATE = "ESCALATE"
    DONE = "DONE"


class ActionKind(str, Enum):
    NOOP = "noop"
    INCREASE_RESOLUTION = "increase_resolution"
    ZOOM_ROI = "zoom_roi"
    FORENSIC_INSPECTION = "forensic_inspection"
    ESCALATE_HUMAN = "escalate_human"
    ROLLBACK = "rollback"


class ReasonCode(str, Enum):
    BASELINE = "baseline"
    NO_FACE_WITH_MOTION = "no_face_with_motion"
    LOW_RESOLUTION = "low_resolution"
    LOW_CONFIDENCE = "low_confidence"
    LOW_QUALITY = "low_quality"
    MOTION_ANOMALY = "motion_anomaly"
    UNKNOWN_IDENTITY = "unknown_identity"
    SPOOF_SUSPECT = "spoof_suspect"
    BUDGET_PRESSURE = "budget_pressure"
    FORENSIC_CONFIRMED = "forensic_confirmed"
    FORENSIC_REJECTED = "forensic_rejected"
    HUMAN_REVIEW = "human_review"


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def iou(self, other: "BBox") -> float:
        x0, y0 = max(self.x, other.x), max(self.y, other.y)
        x1 = min(self.x + self.w, other.x + other.w)
        y1 = min(self.y + self.h, other.y + other.h)
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        union = self.area + other.area - inter
        return float(inter / union) if union else 0.0


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: float
    y: float


class FaceRecord(BaseModel):
    face_id: int
    bbox: BBox
    det_score: float = Field(ge=0.0, le=1.0)
    landmarks: list[Point] = Field(default_factory=list, max_length=5)
    identity: str | None = None
    identity_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    quality: float = Field(0.0, ge=0.0, le=1.0)
    liveness: float = Field(0.0, ge=0.0, le=1.0)
    embedding_digest: str | None = Field(default=None, max_length=32)
    forensic: bool = False


class MotionRecord(BaseModel):
    energy: float = Field(0.0, ge=0.0)
    residual: float = Field(0.0, ge=0.0)
    global_dx: float = 0.0
    global_dy: float = 0.0
    coherent_ratio: float = Field(0.0, ge=0.0, le=1.0)
    anomaly_z: float = Field(0.0, ge=-50.0, le=50.0)
    moving: bool = False
    roi: BBox | None = None


class ActionRecord(BaseModel):
    kind: ActionKind
    detail: str = ""
    resolution: int | None = None
    accepted: bool = True


class ReasonRecord(BaseModel):
    code: ReasonCode
    detail: str = ""
    severity: Literal["info", "warn", "critical"] = "info"


class AgentTrace(BaseModel):
    state_path: list[AgentState] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    reasons: list[ReasonRecord] = Field(default_factory=list)
    adaptive_steps: int = Field(0, ge=0, le=8)
    forensic: bool = False
    escalated: bool = False
    budget_exhausted: bool = False
    decision_ms: float = Field(0.0, ge=0.0)


class Verification(BaseModel):
    passed: bool = True
    detections_before: int = Field(0, ge=0)
    detections_after: int = Field(0, ge=0)
    score_delta: float = 0.0
    iou: float = Field(0.0, ge=0.0, le=1.0)
    notes: str = ""


class CloudRecord(BaseModel):
    provider: str = "offline"
    latency_ms: float = Field(0.0, ge=0.0)
    rekognition_faces: int = Field(0, ge=0)
    matched: bool | None = None
    incident_id: str | None = None
    bedrock_summary: str | None = Field(default=None, max_length=2000)


class FrameTelemetry(BaseModel):
    """Versioned per-frame contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    frame_index: int = Field(ge=0)
    timestamp_utc: str
    source: str = "local"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    processing_ms: float = Field(ge=0.0)
    cpu_ms: float = Field(0.0, ge=0.0)
    fps_instant: float = Field(ge=0.0)
    backend: str
    degraded: bool = False
    frames_dropped: int = Field(0, ge=0)
    faces: list[FaceRecord] = Field(default_factory=list)
    motion: MotionRecord = Field(default_factory=MotionRecord)
    agent: AgentTrace = Field(default_factory=AgentTrace)
    verification: Verification = Field(default_factory=Verification)
    cloud: CloudRecord = Field(default_factory=CloudRecord)

    @field_validator("timestamp_utc")
    @classmethod
    def _iso_utc(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def embedding_digest(embedding: np.ndarray | None) -> str | None:
    """Privacy-preserving digest of an L2-normalised embedding."""
    if embedding is None:
        return None
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    quant = np.clip(np.round(vec / norm * 127.0), -127, 127).astype(np.int8)
    return hashlib.sha256(quant.tobytes()).hexdigest()[:16]


class TelemetryWriter:
    """Append-only JSONL writer with schema validation on every record."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.count = 0
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: FrameTelemetry) -> None:
        if record.run_id != self.run_id:
            raise ValueError("telemetry run_id mismatch")
        self._handle.write(record.model_dump_json() + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def summarise_telemetry(path: str | Path) -> dict:
    """Aggregate a telemetry JSONL file (used by reports and benchmarks)."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        return {"frames": 0}
    latencies = sorted(r["processing_ms"] for r in records)
    n = len(latencies)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((p / 100.0) * n + 0.5)) - 1))
        return round(latencies[idx], 3)

    return {
        "frames": n,
        "mean_ms": round(sum(latencies) / n, 3),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "max_ms": round(max(latencies), 3),
        "faces": sum(len(r["faces"]) for r in records),
        "forensic_frames": sum(1 for r in records if r["agent"]["forensic"]),
        "escalations": sum(1 for r in records if r["agent"]["escalated"]),
        "degraded_frames": sum(1 for r in records if r["degraded"]),
        "budget_exhausted": sum(1 for r in records if r["agent"]["budget_exhausted"]),
        "backends": sorted({r["backend"] for r in records}),
        "generated_utc": utc_now(),
        "wall_clock_s": None,
    }


def timed() -> float:
    return time.perf_counter()
