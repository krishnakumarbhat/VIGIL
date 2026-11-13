"""The autonomous See -> Reason -> Act -> Verify loop.

The agent owns a latency budget, a resolution ladder and a forensic escalation
policy.  It can zoom a motion proposal, re-run perception at a higher
resolution, perform a multi-scale + flip-TTA forensic inspection on a
suspicious face, and roll an action back when verification does not confirm the
expected gain.  Every decision is emitted as structured telemetry; uncertain
cases escalate to a human instead of auto-enrolling an identity.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from src.config import VisionConfig
from src.perception.face import FaceHit, FacePipeline
from src.perception.motion import MotionAnalyzer, MotionReport
from src.telemetry import (
    ActionKind,
    ActionRecord,
    AgentState,
    AgentTrace,
    BBox,
    FrameTelemetry,
    MotionRecord,
    ReasonCode,
    ReasonRecord,
    Verification,
    embedding_digest,
    utc_now,
)

CloudHook = Callable[[np.ndarray, list[FaceHit], str], object]


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class AgentConfig:
    """Behavioural knobs layered on top of :class:`VisionConfig`."""

    resolution_ladder: tuple[int, ...] = (256, 384, 512, 768)
    base_resolution: int = 384
    latency_budget_ms: float = 55.0
    max_adaptive_steps: int = 2
    anomaly_z: float = 2.5
    quality_floor: float = 0.42
    det_floor: float = 0.72
    forensic_max_side: int = 640
    enable_forensics: bool = True
    coop_seconds: float = 3.0

    @classmethod
    def from_config(cls, cfg: VisionConfig) -> "AgentConfig":
        return cls(
            resolution_ladder=tuple(cfg.resolution_ladder),
            base_resolution=cfg.base_resolution,
            latency_budget_ms=cfg.latency_budget_ms,
            max_adaptive_steps=cfg.max_adaptive_steps,
            anomaly_z=cfg.anomaly_trigger_z,
            quality_floor=cfg.quality_floor,
            det_floor=cfg.det_confidence_floor,
            forensic_max_side=cfg.forensic_max_side,
            enable_forensics=cfg.enable_forensics,
        )


@dataclass
class FrameOutcome:
    telemetry: FrameTelemetry
    faces: list[FaceHit]
    motion: MotionReport
    state_path: list[AgentState]
    actions: list[ActionRecord] = field(default_factory=list)


class AutonomousVisionAgent:
    """Budget-aware perception policy with forensic escalation and rollback."""

    def __init__(
        self,
        cfg: VisionConfig,
        pipeline: FacePipeline,
        motion: MotionAnalyzer | None = None,
        cloud: CloudHook | None = None,
        run_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.acfg = AgentConfig.from_config(cfg)
        self.pipeline = pipeline
        self.motion = motion or MotionAnalyzer(scale=cfg.flow_scale, anomaly_z=cfg.motion_anomaly_z)
        self.cloud = cloud
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        self._incident_cooldown: dict[int, float] = {}
        self._frame_index = 0
        self._actions_total = 0
        self._forensic_total = 0
        self._escalations = 0
        self._last_cloud = None
        self._recent_anomaly_until = -1
        self._unknown_counts: dict[int, int] = {}
        self._ever_known: set[int] = set()
        self._escalated_at: dict[int, int] = {}
        self._scene_quality = 1.0
        self._scene_degraded = False
        self._luma_baseline: float | None = None
        self._unknown_streak = 0
        self._unknown_missing = 0
        self._last_escalation_frame = -10**6

    # ------------------------------------------------------------------ public
    def process(
        self,
        frame: np.ndarray,
        source: str = "local",
        frame_index: int | None = None,
        ground_truth: dict | None = None,
    ) -> FrameOutcome:
        t_start = time.perf_counter()
        t_cpu = time.process_time()
        self._last_cloud = None
        idx = self._frame_index if frame_index is None else int(frame_index)
        self._frame_index = idx + 1
        h, w = frame.shape[:2]

        state_path: list[AgentState] = [AgentState.SEE]
        motion = self.motion.update(frame, idx)

        # ------------------------------- SEE
        roi = self._expand(motion.roi, 0.25, frame.shape) if motion.moving else None
        resolution = self.acfg.base_resolution
        faces = self.pipeline.analyze(frame, motion=self.motion, roi=roi, target_side=resolution)
        initial_faces = list(faces)
        initial_best = max((f.score for f in faces), default=0.0)
        self._update_identity_streaks(faces)

        # ---------------------------- REASON
        state_path.append(AgentState.REASON)
        self._scene_quality, luma = self._scene_stats(frame)
        if self._luma_baseline is None:
            self._luma_baseline = luma
        self._scene_degraded = luma < 0.78 * self._luma_baseline and self._luma_baseline > 20.0
        if not self._scene_degraded:
            self._luma_baseline = 0.95 * self._luma_baseline + 0.05 * luma
        if motion.anomaly_z >= self.acfg.anomaly_z and motion.moving:
            self._recent_anomaly_until = idx + 45
        reasons = self._reasons(faces, motion)
        actions: list[ActionRecord] = []
        attempted: set[ActionKind] = set()
        forensic = False
        forensic_evidence: dict = {}
        escalated = False
        budget_exhausted = False
        steps = 0

        while steps < self.acfg.max_adaptive_steps:
            elapsed_ms = (time.process_time() - t_cpu) * 1000.0
            if elapsed_ms > self.acfg.latency_budget_ms * 0.72:
                if any(r.code in {ReasonCode.LOW_RESOLUTION, ReasonCode.LOW_CONFIDENCE, ReasonCode.MOTION_ANOMALY} for r in reasons):
                    budget_exhausted = True
                    reasons.append(
                        ReasonRecord(code=ReasonCode.BUDGET_PRESSURE, detail=f"{elapsed_ms:.1f}ms > 72% budget", severity="warn")
                    )
                break

            action = self._choose_action(faces, motion, reasons, forensic, escalated, attempted, idx)
            if action is None:
                break
            state_path.append(AgentState.ACT)
            steps += 1
            kind, detail = action
            attempted.add(kind)
            accepted = True

            if kind == ActionKind.INCREASE_RESOLUTION:
                nxt = self._next_resolution(resolution)
                if nxt == resolution:
                    break
                before = list(faces)
                resolution = nxt
                new_roi = self._focus_roi(before, motion, frame.shape)
                faces = self.pipeline.analyze(frame, motion=self.motion, roi=new_roi, target_side=resolution)
                accepted = self._improved(before, faces, allow_count_drop=False)
                actions.append(ActionRecord(kind=kind, detail=detail, resolution=resolution, accepted=accepted))
                if not accepted:
                    faces = before
                    resolution = self._prev_resolution(resolution)

            elif kind == ActionKind.ZOOM_ROI:
                before = list(faces)
                new_roi = self._focus_roi(before, motion, frame.shape)
                if new_roi is None:
                    break
                resolution = min(max(resolution, self._next_resolution(resolution)), self.acfg.forensic_max_side)
                faces = self.pipeline.analyze(frame, motion=self.motion, roi=new_roi, target_side=resolution)
                accepted = self._improved(before, faces, allow_count_drop=True)
                actions.append(ActionRecord(kind=kind, detail=detail, resolution=resolution, accepted=accepted))
                if not accepted:
                    faces = before

            elif kind == ActionKind.FORENSIC_INSPECTION:
                faces, forensic_evidence = self._forensic(frame, faces, motion, idx)
                forensic = True
                self._actions_total += 0
                self._forensic_total += 1
                actions.append(ActionRecord(kind=kind, detail=detail, accepted=True))

            elif kind == ActionKind.ESCALATE_HUMAN:
                escalated = True
                self._escalations += 1
                actions.append(ActionRecord(kind=kind, detail=detail, accepted=True))
                reasons.append(ReasonRecord(code=ReasonCode.HUMAN_REVIEW, detail=detail, severity="critical"))
                if self.cloud is not None:
                    self._call_cloud(frame, faces, detail)

            reasons = self._merge_reasons(reasons, self._reasons(faces, motion))

        self._actions_total += len(actions)

        # ----------------------------- VERIFY
        state_path.append(AgentState.VERIFY)
        verification = self._verify(initial_faces, faces, initial_best, forensic, forensic_evidence)
        if any(not action.accepted for action in actions):
            verification.passed = False
            verification.notes = "; ".join(filter(None, [verification.notes, "rejected action rolled back"]))
            actions.append(ActionRecord(kind=ActionKind.ROLLBACK, detail="verification rejected an action", accepted=True))
        if escalated:
            state_path.append(AgentState.ESCALATE)

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        cpu_ms = (time.process_time() - t_cpu) * 1000.0

        trace = AgentTrace(
            state_path=state_path,
            actions=actions,
            reasons=reasons or [ReasonRecord(code=ReasonCode.BASELINE, detail="no trigger")],
            adaptive_steps=steps,
            forensic=forensic,
            escalated=escalated,
            budget_exhausted=budget_exhausted,
            decision_ms=round(cpu_ms, 3),
        )

        telemetry = FrameTelemetry(
            run_id=self.run_id,
            frame_index=idx,
            timestamp_utc=utc_now(),
            source=source,
            width=w,
            height=h,
            processing_ms=round(wall_ms, 3),
            cpu_ms=round(cpu_ms, 3),
            fps_instant=round(1000.0 / max(wall_ms, 1e-3), 2),
            backend=f"{self.pipeline.detector_backend}+{self.pipeline.recognizer_backend}",
            degraded=self.pipeline.degraded,
            faces=[
                self._face_record(f)
                for f in faces
            ],
            motion=MotionRecord(
                energy=motion.energy,
                residual=motion.residual,
                global_dx=motion.global_dx,
                global_dy=motion.global_dy,
                coherent_ratio=motion.coherent_ratio,
                anomaly_z=motion.anomaly_z,
                moving=motion.moving,
                roi=BBox(x=max(0, motion.roi[0]), y=max(0, motion.roi[1]), w=max(0, motion.roi[2]), h=max(0, motion.roi[3])) if motion.roi else None,
            ),
            agent=trace,
            verification=verification,
            cloud=self._last_cloud if self._last_cloud is not None else self._empty_cloud(),
        )
        return FrameOutcome(telemetry=telemetry, faces=faces, motion=motion, state_path=state_path, actions=actions)

    @property
    def stats(self) -> dict:
        return {
            "frames": self._frame_index,
            "actions": self._actions_total,
            "forensic_runs": self._forensic_total,
            "escalations": self._escalations,
        }

    # ----------------------------------------------------------------- helpers
    def _face_record(self, face: FaceHit):
        from src.telemetry import FaceRecord

        payload = face.as_record(store_embedding=self.cfg.store_embeddings)
        if self.cfg.redact_identity_labels and payload.get("identity"):
            payload["identity"] = hashlib.sha256(str(payload["identity"]).encode()).hexdigest()[:8]
        return FaceRecord(**payload)

    def _empty_cloud(self):
        from src.telemetry import CloudRecord

        return CloudRecord(provider="offline")

    def _update_identity_streaks(self, faces: list[FaceHit]) -> None:
        """Advance temporal identity state exactly once per frame.

        This used to live inside :meth:`_reasons`, which is called again after
        each action; that double-counted a single unknown frame and escalated
        too early.  Keeping the mutation separate makes the policy idempotent
        within a frame.
        """
        active_ids = {f.face_id for f in faces}
        for face_id in list(self._unknown_counts):
            if face_id not in active_ids:
                self._unknown_counts.pop(face_id, None)
        unseen_unknown = False
        for face in faces:
            if face.identity is None:
                if face.face_id in self._ever_known:
                    self._unknown_counts[face.face_id] = 0
                else:
                    self._unknown_counts[face.face_id] = self._unknown_counts.get(face.face_id, 0) + 1
                    if face.quality is None or face.quality.score >= 0.40:
                        unseen_unknown = True
            else:
                self._ever_known.add(face.face_id)
                self._unknown_counts[face.face_id] = 0
        if unseen_unknown:
            self._unknown_streak += 1
            self._unknown_missing = 0
        else:
            self._unknown_missing += 1
            if self._unknown_missing > 2:
                self._unknown_streak = 0

    def _reasons(self, faces: list[FaceHit], motion: MotionReport) -> list[ReasonRecord]:
        out: list[ReasonRecord] = []
        if self._scene_quality < self.acfg.quality_floor or self._scene_degraded:
            detail = "scene_low_light" if self._scene_degraded else f"scene_q={self._scene_quality:.2f}"
            out.append(ReasonRecord(code=ReasonCode.LOW_QUALITY, detail=detail, severity="warn"))
        if motion.anomaly_z >= self.acfg.anomaly_z and motion.moving:
            out.append(
                ReasonRecord(code=ReasonCode.MOTION_ANOMALY, detail=f"z={motion.anomaly_z:.2f}", severity="critical")
            )
        if not faces and motion.moving:
            out.append(ReasonRecord(code=ReasonCode.NO_FACE_WITH_MOTION, detail="motion without detection", severity="warn"))
        for face in faces:
            size = min(face.box[2], face.box[3])
            if size < self.cfg.min_face_px:
                out.append(
                    ReasonRecord(code=ReasonCode.LOW_RESOLUTION, detail=f"{size}px < {self.cfg.min_face_px}px", severity="warn")
                )
            if face.score < self.acfg.det_floor:
                out.append(
                    ReasonRecord(code=ReasonCode.LOW_CONFIDENCE, detail=f"det={face.score:.3f}", severity="warn")
                )
            if face.quality and face.quality.score < self.acfg.quality_floor:
                out.append(
                    ReasonRecord(code=ReasonCode.LOW_QUALITY, detail=f"q={face.quality.score:.3f}", severity="warn")
                )
            if self.cfg.enable_liveness and face.liveness < 0.32:
                out.append(
                    ReasonRecord(code=ReasonCode.SPOOF_SUSPECT, detail=face.liveness_reason, severity="critical")
                )
            if face.identity is None and size >= self.cfg.min_face_px:
                out.append(ReasonRecord(code=ReasonCode.UNKNOWN_IDENTITY, detail="not in consenting gallery", severity="warn"))
        return out

    @staticmethod
    def _scene_stats(frame: np.ndarray) -> tuple[float, float]:
        """Cheap whole-scene quality score and mean luma."""
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        luma = float(gray.mean())
        sharp = float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / np.log1p(900.0))
        return float(min(1.0, max(0.0, sharp))), luma

    @staticmethod
    def _merge_reasons(existing: list[ReasonRecord], new: list[ReasonRecord]) -> list[ReasonRecord]:
        seen = {(r.code, r.detail) for r in existing}
        for reason in new:
            if (reason.code, reason.detail) not in seen:
                existing.append(reason)
                seen.add((reason.code, reason.detail))
        return existing

    def _choose_action(
        self,
        faces: list[FaceHit],
        motion: MotionReport,
        reasons: list[ReasonRecord],
        forensic_done: bool,
        escalated: bool,
        attempted: set[ActionKind],
        frame_idx: int,
    ) -> tuple[ActionKind, str] | None:
        codes = {r.code for r in reasons}
        if (
            self.acfg.enable_forensics
            and not forensic_done
            and ActionKind.FORENSIC_INSPECTION not in attempted
            and (
                ReasonCode.LOW_QUALITY in codes
                or ReasonCode.SPOOF_SUSPECT in codes
                or ReasonCode.MOTION_ANOMALY in codes
                or ReasonCode.NO_FACE_WITH_MOTION in codes
            )
        ):
            return ActionKind.FORENSIC_INSPECTION, "anomaly requires deep inspection"
        if ReasonCode.NO_FACE_WITH_MOTION in codes and motion.roi is not None and ActionKind.ZOOM_ROI not in attempted:
            return ActionKind.ZOOM_ROI, "motion proposal without a face"
        if ReasonCode.LOW_RESOLUTION in codes and ActionKind.INCREASE_RESOLUTION not in attempted:
            return ActionKind.INCREASE_RESOLUTION, "face below minimum pixel size"
        if ReasonCode.LOW_CONFIDENCE in codes and not forensic_done and ActionKind.INCREASE_RESOLUTION not in attempted:
            return ActionKind.INCREASE_RESOLUTION, "low detector confidence"
        recent_anomaly = frame_idx <= self._recent_anomaly_until
        if (
            self.cfg.unknown_identity_escalation
            and not escalated
            and ActionKind.ESCALATE_HUMAN not in attempted
            and self._unknown_streak >= 10
            and frame_idx - self._last_escalation_frame > 120
            and (
                ReasonCode.SPOOF_SUSPECT in codes
                or recent_anomaly
                or self._scene_degraded
            )
        ):
            self._last_escalation_frame = frame_idx
            return ActionKind.ESCALATE_HUMAN, "persistent unknown identity near a recent anomaly"
        return None

    def _forensic(
        self, frame: np.ndarray, faces: list[FaceHit], motion: MotionReport, frame_idx: int = 0
    ) -> tuple[list[FaceHit], dict]:
        """Deep inspection: high-res recovery scan + upscale + flip-TTA.

        The recovery scan runs when the scene is degraded or motion suggests a
        face the base pass missed.  It deliberately uses a lower detector
        threshold and the forensic resolution, then requires identity/quality
        evidence before the face is accepted.
        """
        from src.perception.quality import assess_face

        evidence: dict = {"inspected": 0, "consistency": [], "multi_scale": [], "recovered": 0}
        if self.acfg.enable_forensics and (
            not faces
            or (motion.anomaly_z >= self.acfg.anomaly_z and motion.moving)
            or self._scene_degraded
            or self._scene_quality < self.acfg.quality_floor
        ):
            roi = self._expand(motion.roi, 0.45, frame.shape) if motion.roi else None
            found = self.pipeline.detect(
                frame, roi=roi, target_side=self.acfg.forensic_max_side, score_threshold=0.25
            )
            for box, score, landmarks in found:
                if any(_box_iou(box, face.box) > 0.3 for face in faces):
                    continue
                quality = assess_face(frame, box, landmarks, self.cfg.min_face_px)
                probe = FaceHit(face_id=9000 + len(faces), box=box, score=score, landmarks=landmarks, quality=quality)
                emb = self.pipeline.embed(frame, probe)
                identity, id_score = self.pipeline.match(emb) if emb is not None else (None, None)
                faces.append(
                    FaceHit(
                        face_id=probe.face_id,
                        box=box,
                        score=score,
                        landmarks=landmarks,
                        embedding=emb,
                        identity=identity,
                        identity_score=id_score,
                        quality=quality,
                        liveness=0.5,
                        liveness_reason="forensic_recovery",
                        digest=embedding_digest(emb),
                        forensic=True,
                    )
                )
                evidence["recovered"] += 1

        if not faces:
            return faces, evidence
        suspicion = sorted(
            faces,
            key=lambda f: (
                (1.0 - (f.quality.score if f.quality else 0.0))
                + (1.0 - f.liveness)
                + (0.3 if f.identity is None else 0.0)
            ),
            reverse=True,
        )[:3]
        inspected_ids = set()
        for face in suspicion:
            box = self._expand(face.box, 0.35, frame.shape)
            side = int(max(224, min(self.acfg.forensic_max_side, max(face.box[2], face.box[3]) * 3)))
            found = self.pipeline.detect(
                frame, roi=box, target_side=side, score_threshold=max(0.3, self.cfg.det_threshold * 0.75)
            )
            if found:
                best = max(found, key=lambda t: t[1])
                consistency = None
                if face.embedding is not None:
                    fh = FaceHit(face_id=face.face_id, box=best[0], score=best[1], landmarks=best[2])
                    high_res = self.pipeline.embed(frame, fh)
                    if high_res is not None:
                        consistency = float(np.dot(high_res, face.embedding))
                        evidence["consistency"].append(round(consistency, 4))
                    x, y, w, h = face.box
                    crop = frame[max(0, y) : y + h, max(0, x) : x + w]
                    if crop.size:
                        flipped = cv2.flip(crop, 1)
                        fh2 = FaceHit(face_id=face.face_id, box=(0, 0, flipped.shape[1], flipped.shape[0]), score=best[1])
                        femb = self.pipeline.embed(flipped, fh2)
                        if femb is not None:
                            evidence["multi_scale"].append(round(float(np.dot(femb, face.embedding)), 4))
                fuse = 0.45 * best[1] + 0.25 * (face.quality.score if face.quality else 0.0) + 0.30 * (consistency if consistency is not None else 0.5)
                if fuse >= self.cfg.det_threshold and best[1] >= face.score * 0.9:
                    face.box = best[0]
                    face.score = max(face.score, best[1])
                    face.forensic = True
                    inspected_ids.add(face.face_id)
                evidence["inspected"] += 1
        evidence["confirmed"] = len(inspected_ids)
        return faces, evidence

    def _verify(
        self,
        before: list[FaceHit],
        after: list[FaceHit],
        initial_best: float,
        forensic: bool,
        evidence: dict,
    ) -> Verification:
        best_after = max((f.score for f in after), default=0.0)
        iou = 0.0
        if before and after:
            iou = max(
                BBox(x=a.box[0], y=a.box[1], w=a.box[2], h=a.box[3]).iou(
                    BBox(x=b.box[0], y=b.box[1], w=b.box[2], h=b.box[3])
                )
                for a in before
                for b in after
            )
        delta = best_after - initial_best
        passed = True
        notes = []
        if len(after) < len(before) and not forensic:
            passed = False
            notes.append("detection count regressed")
        if delta < -0.08:
            passed = False
            notes.append(f"confidence regressed by {-delta:.3f}")
        if forensic:
            cons = evidence.get("consistency") or []
            if cons and min(cons) < 0.45:
                passed = False
                notes.append("embedding inconsistent under upscale")
            notes.append(f"forensic confirmed={evidence.get('confirmed', 0)}")
        if not notes:
            notes.append("stable" if not forensic else "forensic evidence consistent")
        return Verification(
            passed=passed,
            detections_before=len(before),
            detections_after=len(after),
            score_delta=round(delta, 4),
            iou=round(iou, 4),
            notes="; ".join(notes),
        )

    def _improved(self, before: list[FaceHit], after: list[FaceHit], allow_count_drop: bool = False) -> bool:
        """Accept a re-scan only if it preserves the scene's detections.

        A higher-resolution pass is allowed to improve confidence, never to
        silently lose faces: count regression causes a rollback.  A deliberate
        ROI zoom may narrow the scene, so it is allowed to drop faces outside
        the requested region as long as the best target improves.
        """
        if not before:
            return bool(after)
        if not after:
            return False
        if not allow_count_drop and len(after) < len(before):
            return False
        best_before = max(f.score for f in before)
        best_after = max(f.score for f in after)
        return best_after >= best_before - 0.05

    def _focus_roi(
        self, faces: list[FaceHit], motion: MotionReport, shape: tuple[int, ...]
    ) -> tuple[int, int, int, int] | None:
        if faces:
            face = max(faces, key=lambda f: f.box[2] * f.box[3])
            return self._expand(face.box, 0.55, shape)
        if motion.roi:
            return self._expand(motion.roi, 0.45, shape)
        return None

    @staticmethod
    def _expand(box: tuple[int, int, int, int] | None, margin: float, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        if box is None:
            return None
        h, w = shape[:2]
        x, y, bw, bh = (int(v) for v in box)
        mx, my = int(bw * margin), int(bh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _next_resolution(self, current: int) -> int:
        for value in self.acfg.resolution_ladder:
            if value > current:
                return value
        return current

    def _prev_resolution(self, current: int) -> int:
        lower = [v for v in self.acfg.resolution_ladder if v < current]
        return max(lower) if lower else current

    def _call_cloud(self, frame: np.ndarray, faces: list[FaceHit], reason: str) -> None:
        if self.cloud is None:
            return
        try:
            record = self.cloud(frame, faces, reason)
            if record is not None:
                self._last_cloud = record
        except Exception:  # cloud must never break the local loop
            self._last_cloud = None
