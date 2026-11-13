"""Agent loop tests with deterministic fakes (no model dependency)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.agent.agent import AutonomousVisionAgent
from src.cloud import OfflineCloud
from src.config import VisionConfig
from src.perception.face import FaceHit
from src.perception.motion import MotionReport
from src.telemetry import ActionKind, AgentState, FrameTelemetry


class FakeMotion:
    def __init__(self, report: MotionReport | None = None) -> None:
        self.report = report or MotionReport()
        self.calls = 0

    def update(self, frame: np.ndarray, frame_index: int = 0) -> MotionReport:
        self.calls += 1
        return replace(self.report)


class FakePipeline:
    detector_backend = "fake"
    recognizer_backend = "fake"
    degraded = False
    notes: list[str] = []
    descriptor_dim = 128

    def __init__(self, faces: list[FaceHit] | None = None) -> None:
        self.faces = list(faces or [])
        self.calls: list[dict] = []
        self.detect_calls = 0

    def reset_tracking(self) -> None:
        return None

    def analyze(self, frame, motion=None, roi=None, target_side=None, embed=True, score_threshold=None, forensic=False):
        self.calls.append({"roi": roi, "target_side": target_side, "embed": embed})
        return [replace(face) for face in self.faces]

    def detect(self, frame, roi=None, target_side=None, score_threshold=None):
        self.detect_calls += 1
        return []

    def embed(self, frame, hit):
        return None

    def match(self, embedding):
        return None, None

    def enroll(self, label, embedding, consent=True):
        return True


def _frame(width: int = 320, height: int = 240) -> np.ndarray:
    rng = np.random.default_rng(3)
    frame = np.full((height, width, 3), 120, dtype=np.int16)
    frame += rng.integers(-12, 13, frame.shape, dtype=np.int16)
    # Sharp edges so the scene quality gate is not triggered accidentally.
    frame[40:200, 20:300:12] = 235
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_frame_emits_valid_telemetry_and_state_path(cfg: VisionConfig, hit_factory) -> None:
    agent = AutonomousVisionAgent(cfg, FakePipeline([hit_factory(identity="alice")]), motion=FakeMotion(), run_id="run-unit")
    outcome = agent.process(_frame(), frame_index=0)
    assert isinstance(outcome.telemetry, FrameTelemetry)
    assert outcome.telemetry.run_id == "run-unit"
    assert outcome.telemetry.frame_index == 0
    assert outcome.state_path[:3] == [AgentState.SEE, AgentState.REASON, AgentState.VERIFY]
    assert outcome.telemetry.cpu_ms >= 0.0
    assert outcome.telemetry.processing_ms >= 0.0
    assert outcome.telemetry.backend == "fake+fake"
    assert agent.stats["frames"] == 1


def test_low_resolution_face_triggers_resolution_step(cfg: VisionConfig, hit_factory) -> None:
    face = hit_factory(box=(60, 50, 40, 40), score=0.9, identity=None)
    pipeline = FakePipeline([face])
    agent = AutonomousVisionAgent(cfg, pipeline, motion=FakeMotion(MotionReport()), run_id="run-res")
    outcome = agent.process(_frame(), frame_index=0)
    kinds = [action.kind for action in outcome.actions]
    assert ActionKind.INCREASE_RESOLUTION in kinds
    assert outcome.telemetry.agent.adaptive_steps >= 1
    assert max(call["target_side"] for call in pipeline.calls) > cfg.base_resolution


def test_low_quality_face_triggers_forensic_inspection(cfg: VisionConfig, hit_factory) -> None:
    face = hit_factory(box=(70, 50, 110, 110), score=0.9, quality=0.15, liveness=0.9)
    agent = AutonomousVisionAgent(cfg, FakePipeline([face]), motion=FakeMotion(), run_id="run-forensic")
    outcome = agent.process(_frame(), frame_index=0)
    assert outcome.telemetry.agent.forensic is True
    assert any(action.kind == ActionKind.FORENSIC_INSPECTION for action in outcome.actions)
    assert outcome.telemetry.verification.detections_before == 1


def test_persistent_unknown_near_anomaly_escalates_and_survives_cloud_failure(
    cfg: VisionConfig, hit_factory
) -> None:
    face = hit_factory(box=(70, 50, 110, 110), score=0.9, identity=None, liveness=0.05)
    motion = FakeMotion(MotionReport(moving=True, anomaly_z=5.0, roi=(40, 30, 160, 160)))

    def broken_cloud(frame, faces, reason):
        raise RuntimeError("network down")

    agent = AutonomousVisionAgent(cfg, FakePipeline([face]), motion=motion, cloud=broken_cloud, run_id="run-escalate")
    agent._unknown_streak = 10
    outcome = agent.process(_frame(), frame_index=10)
    assert outcome.telemetry.agent.escalated is True
    assert AgentState.ESCALATE in outcome.state_path
    assert any(action.kind == ActionKind.ESCALATE_HUMAN for action in outcome.actions)
    assert outcome.telemetry.cloud.provider == "offline"  # cloud failure leaves the default record


def test_offline_cloud_records_incident_id(cfg: VisionConfig, hit_factory) -> None:
    face = hit_factory(box=(70, 50, 110, 110), score=0.9, identity=None, liveness=0.05)
    motion = FakeMotion(MotionReport(moving=True, anomaly_z=5.0, roi=(40, 30, 160, 160)))
    cloud = OfflineCloud(cfg)
    agent = AutonomousVisionAgent(cfg, FakePipeline([face]), motion=motion, cloud=cloud, run_id="run-cloud")
    agent._unknown_streak = 10
    outcome = agent.process(_frame(), frame_index=10)
    assert cloud.calls == 1
    assert outcome.telemetry.cloud.incident_id is not None
    assert outcome.telemetry.cloud.incident_id.startswith("inc-")
    assert "No biometric payload" in (outcome.telemetry.cloud.bedrock_summary or "")


def test_verification_rejects_confidence_regression(cfg: VisionConfig, hit_factory) -> None:
    agent = AutonomousVisionAgent(cfg, FakePipeline(), motion=FakeMotion(), run_id="run-verify")
    before = hit_factory(score=0.9)
    after = hit_factory(score=0.5)
    verification = agent._verify([before], [after], initial_best=0.9, forensic=False, evidence={})
    assert verification.passed is False
    assert verification.score_delta < 0
    assert "confidence regressed" in verification.notes


def test_rescan_may_not_lose_faces(cfg: VisionConfig, hit_factory) -> None:
    agent = AutonomousVisionAgent(cfg, FakePipeline(), motion=FakeMotion(), run_id="run-improved")
    before = [hit_factory(face_id=1), hit_factory(box=(150, 30, 90, 90), face_id=2)]
    after = [hit_factory(face_id=1, score=0.95)]
    assert agent._improved(before, after) is False
    assert agent._improved(before, after, allow_count_drop=True) is True


def test_unknown_streak_advances_once_per_frame(cfg: VisionConfig, hit_factory) -> None:
    face = hit_factory(box=(70, 50, 100, 100), score=0.9, identity=None)
    agent = AutonomousVisionAgent(cfg, FakePipeline([face]), motion=FakeMotion(), run_id="run-streak")
    agent.process(_frame(), frame_index=0)
    assert agent._unknown_streak == 1
    agent.process(_frame(), frame_index=1)
    assert agent._unknown_streak == 2


def test_verification_rejects_inconsistent_forensic_embedding(cfg: VisionConfig, hit_factory) -> None:
    agent = AutonomousVisionAgent(cfg, FakePipeline(), motion=FakeMotion(), run_id="run-verify2")
    face = hit_factory()
    verification = agent._verify(
        [face], [face], initial_best=face.score, forensic=True, evidence={"consistency": [0.2], "confirmed": 1}
    )
    assert verification.passed is False
    assert "inconsistent" in verification.notes
