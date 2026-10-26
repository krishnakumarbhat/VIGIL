"""VIGIL-5 engine: video/image ingestion, offline gallery, annotated output.

Usage
-----
    python3 -m src.engine --video data/demo/demo.mp4 \
        --out outputs/demo_annotated.mp4 \
        --telemetry outputs/telemetry/demo.jsonl \
        --gallery data/demo/gallery

The engine is deliberately thin: perception, policy and telemetry live in
their own modules; here we only wire capture, drawing, gallery consent and
artifact writing.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from src.agent.agent import AutonomousVisionAgent, FrameOutcome
from src.cloud import OfflineCloud, build_cloud
from src.config import VisionConfig, load_config
from src.perception.face import FacePipeline
from src.perception.motion import MotionAnalyzer
from src.telemetry import FrameTelemetry, TelemetryWriter, summarise_telemetry

COLORS = {
    "known": (80, 220, 120),
    "unknown": (40, 190, 255),
    "forensic": (60, 60, 255),
}


class VisionEngine:
    """Single-process vision runtime shared by local runs and Lambda."""

    def __init__(self, cfg: VisionConfig | None = None, cloud=None, run_id: str | None = None) -> None:
        self.cfg = cfg or load_config()
        if self.cfg.threads:
            cv2.setNumThreads(self.cfg.threads)
        cv2.setUseOptimized(True)
        if self.cfg.backend != "opencl":
            cv2.ocl.setUseOpenCL(False)
        self.run_id = run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        self.pipeline = FacePipeline(self.cfg)
        self.cloud = cloud if cloud is not None else build_cloud(self.cfg)
        self.agent = AutonomousVisionAgent(
            self.cfg,
            self.pipeline,
            motion=MotionAnalyzer(scale=self.cfg.flow_scale, anomaly_z=self.cfg.motion_anomaly_z),
            cloud=self.cloud.__call__ if callable(self.cloud) else None,
            run_id=self.run_id,
        )
        self.incidents: list[dict] = []
        self.enrolled: list[str] = []

    # --------------------------------------------------------------- gallery
    def enroll_gallery(self, directory: str | Path) -> list[str]:
        """Enroll reference faces that have explicit consent.

        ``consent.json`` must contain ``{"labels": ["alice", ...]}``.  Without
        that file nothing is enrolled — consent is not inferred from presence.
        """
        directory = Path(directory)
        consent_path = directory / "consent.json"
        if not consent_path.exists():
            return []
        consent = json.loads(consent_path.read_text()).get("labels", [])
        enrolled = []
        for image_path in sorted(directory.glob("*.jpg")):
            label = image_path.stem
            if label not in consent:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            self.pipeline.reset_tracking()
            hits = self.pipeline.analyze(image, embed=True, target_side=640)
            if not hits:
                continue
            best = max(hits, key=lambda h: h.score)
            if best.embedding is not None and self.pipeline.enroll(label, best.embedding, consent=True):
                enrolled.append(label)
        self.enrolled = enrolled
        return enrolled

    # ---------------------------------------------------------------- frames
    def process_frame(self, frame: np.ndarray, source: str = "local", index: int = 0) -> FrameOutcome:
        outcome = self.agent.process(frame, source=source, frame_index=index)
        if outcome.telemetry.agent.escalated:
            self.incidents.append(
                {
                    "frame_index": index,
                    "source": source,
                    "timestamp_utc": outcome.telemetry.timestamp_utc,
                    "faces": len(outcome.faces),
                    "unknown": sum(1 for f in outcome.faces if f.identity is None),
                    "reasons": [r.code.value for r in outcome.telemetry.agent.reasons],
                    "incident_id": outcome.telemetry.cloud.incident_id,
                    "summary": outcome.telemetry.cloud.bedrock_summary,
                }
            )
        return outcome

    def annotate(self, frame: np.ndarray, outcome: FrameOutcome) -> np.ndarray:
        canvas = frame.copy()
        for face in outcome.faces:
            x, y, w, h = face.box
            if face.forensic:
                color = COLORS["forensic"]
            elif face.identity:
                color = COLORS["known"]
            else:
                color = COLORS["unknown"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            if face.landmarks is not None:
                for point in np.asarray(face.landmarks).reshape(-1, 2):
                    cv2.circle(canvas, (int(point[0]), int(point[1])), 2, color, -1)
            label = face.identity or "unknown"
            text = f"{label} {face.identity_score:.2f}" if face.identity_score is not None else label
            text += f" q{face.quality.score:.2f}" if face.quality else ""
            if face.forensic:
                text += " FORENSIC"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = min(max(2, x), max(2, canvas.shape[1] - tw - 4))
            ty = max(16, y - 6)
            cv2.rectangle(canvas, (tx - 2, ty - th - 3), (tx + tw + 2, ty + baseline), (18, 18, 18), -1)
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Sparse flow quiver (motion evidence rendered, not decorative).
        motion = outcome.motion
        if motion.moving and motion.roi:
            x, y, w, h = motion.roi
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 160, 40), 1)
        state = " -> ".join(s.value for s in outcome.state_path)
        banner = (
            f"{state} | {outcome.telemetry.backend} | {outcome.telemetry.processing_ms:.1f} ms "
            f"| z={motion.anomaly_z:.1f} | faces={len(outcome.faces)}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (18, 18, 18), -1)
        cv2.putText(canvas, banner, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        if outcome.telemetry.agent.escalated:
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (0, 0, 235), 4)
            cv2.putText(canvas, "HUMAN REVIEW REQUIRED", (canvas.shape[1] - 250, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 235), 2)
        return canvas

    # ------------------------------------------------------------------- runs
    def run_video(
        self,
        video: str | Path,
        output_video: str | Path | None = None,
        telemetry_path: str | Path | None = None,
        max_frames: int | None = None,
        source_label: str | None = None,
    ) -> dict:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        telemetry_path = Path(telemetry_path or self.cfg.telemetry_dir / f"{self.run_id}.jsonl")
        writer = None
        if output_video:
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps_in, (width, height))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(Path(output_video).with_suffix(".avi")), cv2.VideoWriter_fourcc(*"MJPG"), fps_in, (width, height))
        frames = 0
        wall_start = time.perf_counter()
        result: dict = {"run_id": self.run_id, "source": str(video), "telemetry": str(telemetry_path)}
        with TelemetryWriter(telemetry_path, self.run_id) as tw:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                outcome = self.process_frame(frame, source=source_label or str(video), index=frames)
                tw.write(outcome.telemetry)
                if writer is not None and self.cfg.draw_overlay:
                    writer.write(self.annotate(frame, outcome))
                frames += 1
                if max_frames and frames >= max_frames:
                    break
        cap.release()
        if writer is not None:
            writer.release()
        wall = time.perf_counter() - wall_start
        summary = summarise_telemetry(telemetry_path)
        summary["wall_clock_s"] = round(wall, 3)
        summary["fps_wall"] = round(frames / wall, 2) if wall > 0 else 0.0
        summary["enrolled"] = self.enrolled
        summary["incidents"] = self.incidents
        summary["agent_stats"] = self.agent.stats
        summary["backend"] = f"{self.pipeline.detector_backend}+{self.pipeline.recognizer_backend}"
        summary["degraded"] = self.pipeline.degraded
        summary["notes"] = self.pipeline.notes
        summary["run_id"] = self.run_id
        result.update(summary)
        return result

    def run_image_dir(self, directory: str | Path, telemetry_path: str | Path | None = None) -> dict:
        directory = Path(directory)
        telemetry_path = Path(telemetry_path or self.cfg.telemetry_dir / f"{self.run_id}.jsonl")
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        result = {"run_id": self.run_id, "source": str(directory), "telemetry": str(telemetry_path), "images": len(images)}
        with TelemetryWriter(telemetry_path, self.run_id) as tw:
            for idx, path in enumerate(images):
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                outcome = self.process_frame(frame, source=str(path), index=idx)
                tw.write(outcome.telemetry)
        summary = summarise_telemetry(telemetry_path)
        summary["incidents"] = self.incidents
        result.update(summary)
        return result


def _iter_video(path: str | Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VIGIL-5 autonomous vision engine")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    engine = VisionEngine(load_config(args.config))
    if args.gallery:
        engine.enroll_gallery(args.gallery)
    if args.video:
        summary = engine.run_video(args.video, args.out, args.telemetry, args.max_frames)
    elif args.images:
        summary = engine.run_image_dir(args.images, args.telemetry)
    else:
        parser.error("one of --video or --images is required")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("frames", "fps_wall", "p50_ms", "p95_ms", "faces", "backend") if k in summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
