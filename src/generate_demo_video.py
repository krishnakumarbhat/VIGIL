"""Generate the bundled demo clip, its ground truth and the annotated MP4.

The clip is synthetic by design: licence-clean portrait crops are composited
onto a generated concourse, with scripted camera pan, lighting flicker,
occlusion, fast entry and motion blur.  Because every face position is known,
the same clip doubles as a labelled end-to-end benchmark for the agent loop.
After composing the raw clip the script enrolls the consented gallery and runs
the real agent over it, so `outputs/demo_annotated.mp4` is shipped evidence
rather than a separately edited video.

    python3 src/generate_demo_video.py

Output:
    data/demo/demo.mp4              640x360, 24 fps, 12 s, mp4v
    data/demo/demo_gt.json          per-frame boxes, identity, event flags
    data/demo/gallery/*.jpg         consented staff references
    data/demo/frames/*.jpg          three preview stills
    outputs/demo_annotated.mp4      agent-overlaid clip (default)
    outputs/telemetry/demo.jsonl    288 validated records (default)
    outputs/demo_summary.json       run summary (default)

Use `--skip-annotated` to compose only the raw clip.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python3 src/generate_demo_video.py`
    sys.path.insert(0, str(ROOT))
FACE_DIR = ROOT / "data" / "samples" / "faces"
W, H = 640, 360


@dataclass
class Track:
    identity: str
    ref: np.ndarray
    start: int
    end: int
    x0: float
    x1: float
    y: float
    size: int
    depth: int = 0
    speed: float = 0.0
    tight: tuple[float, float, float, float] = (0.15, 0.15, 0.7, 0.7)

    def position(self, frame: int) -> tuple[float, float, float]:
        if frame < self.start or frame > self.end:
            return (-1000.0, -1000.0, 0.0)
        if self.speed > 2.0:
            # Fast entry: constant velocity for 18 frames, then settle.
            fast = 18
            if frame <= self.start + fast:
                x = self.x0 + (self.x1 + 45.0 - self.x0) * (frame - self.start) / fast
            else:
                t = (frame - self.start - fast) / max(1, self.end - self.start - fast)
                ease = t * t * (3 - 2 * t)
                x = (self.x1 + 45.0) + (self.x1 - (self.x1 + 45.0)) * ease
        else:
            t = (frame - self.start) / max(1, self.end - self.start)
            ease = t * t * (3 - 2 * t)
            x = self.x0 + (self.x1 - self.x0) * ease
        y = self.y + 4.0 * math.sin(frame * 0.11 + self.depth)
        size = self.size * (0.94 + 0.06 * math.sin(frame * 0.07 + self.depth))
        return (x, y, size)


def _tight_box(image: np.ndarray) -> tuple[float, float, float, float]:
    """Normalised tight face box inside a reference crop."""
    det = cv2.FaceDetectorYN_create(str(ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.5, 0.3, 5)
    h, w = image.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(image)
    if faces is None:
        return (0.15, 0.15, 0.7, 0.7)
    x, y, bw, bh = faces[0][:4]
    return (float(x / w), float(y / h), float(bw / w), float(bh / h))


def _ref_for(source_suffix: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    manifest = json.loads((FACE_DIR / "manifest.json").read_text())
    for ref in manifest["references"]:
        if ref["source"].endswith(source_suffix):
            image = cv2.imread(str(FACE_DIR / ref["file"]))
            if image is not None:
                return image, _tight_box(image)
    raise FileNotFoundError(f"reference not found for {source_suffix}")


def background(frame: int, pan: float, flicker: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    top = np.array([44, 48, 58], np.float32)
    bottom = np.array([116, 120, 128], np.float32)
    ramp = (yy / H)[..., None]
    img = top * (1 - ramp) + bottom * ramp
    # Perspective floor.
    horizon = 196
    floor_mask = yy > horizon
    depth = np.clip((yy - horizon) / (H - horizon), 0, 1)
    floor = np.array([58, 62, 70], np.float32) * (1 - depth[..., None] * 0.35)
    img[floor_mask] = floor[floor_mask]
    for k in range(7):
        x = int(((k * 130) - pan * (0.4 + k * 0.06)) % (W + 160)) - 80
        cv2.line(img, (x, horizon), (int(x + (x - 320) * 1.6), H), (78, 82, 90), 2)
    # Wall panels / windows, parallax with pan.
    for k in range(5):
        x = int((k * 190 - pan * 0.55) % (W + 260)) - 130
        cv2.rectangle(img, (x, 40), (x + 130, 168), (60, 72, 92), -1)
        cv2.rectangle(img, (x + 6, 46), (x + 124, 162), (86, 112, 138), -1)
        for gy in range(3):
            cv2.line(img, (x + 8, 70 + gy * 34), (x + 122, 70 + gy * 34), (70, 92, 116), 1)
    # Text drawing requires 8-bit input.
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Signage.
    sign_x = int((430 - pan * 0.7) % (W + 240)) - 120
    cv2.rectangle(img, (sign_x, 212), (sign_x + 150, 246), (24, 30, 44), -1)
    cv2.putText(img, "TERMINAL B  /  GATE 07", (sign_x + 8, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 225, 240), 1, cv2.LINE_AA)
    # Distant figures (blurred, non-face) for depth.
    for k in range(4):
        x = int((90 + k * 160 - pan * 0.25) % (W + 80)) - 40
        cv2.circle(img, (x, 188), 9, (48, 52, 62), -1)
        cv2.line(img, (x, 197), (x, 238), (48, 52, 62), 9)
    img = np.clip(img.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
    return img


def paste_face(
    canvas: np.ndarray,
    face: np.ndarray,
    cx: float,
    cy: float,
    size: int,
    angle: float,
    blur: float,
    depth_scale: float = 1.0,
) -> tuple[int, int, int, int]:
    fh, fw = face.shape[:2]
    side = int(max(24, size))
    crop = cv2.resize(face, (side, side), interpolation=cv2.INTER_AREA)
    # Photometric match to the local background.
    bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    bx1, by1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    patch = canvas[max(0, by0) : max(0, by1), max(0, bx0) : max(0, bx1)]
    if patch.size and crop.size:
        target_mean = patch.reshape(-1, 3).mean(axis=0)
        face_mean = crop.reshape(-1, 3).mean(axis=0) + 1e-6
        crop = np.clip(crop.astype(np.float32) * (0.55 + 0.45 * target_mean / face_mean), 0, 255).astype(np.uint8)
    if blur > 0.1:
        crop = cv2.GaussianBlur(crop, (0, 0), blur)
    if abs(angle) > 0.1:
        mat = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
        crop = cv2.warpAffine(crop, mat, (side, side), borderMode=cv2.BORDER_REFLECT)
    yy, xx = np.mgrid[0:side, 0:side]
    mask = (((xx - side / 2) / (side * 0.45)) ** 2 + ((yy - side / 2) / (side * 0.49)) ** 2) <= 1
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, side * 0.05))
    mask = cv2.merge([mask, mask, mask])
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = x0 + side, y0 + side
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x1), min(H, y1)
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    if dx1 <= dx0 or dy1 <= dy0:
        return (0, 0, 0, 0)
    roi = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = np.clip(
        crop[sy0:sy1, sx0:sx1].astype(np.float32) * mask[sy0:sy1, sx0:sx1] + roi * (1 - mask[sy0:sy1, sx0:sx1]),
        0,
        255,
    ).astype(np.uint8)
    return (dx0, dy0, dx1 - dx0, dy1 - dy0)


def compose(frame_idx: int, tracks: list[Track], fps: float) -> tuple[np.ndarray, list[dict], dict]:
    t = frame_idx / fps
    pan = 26.0 * math.sin(t * 0.55) + 8.0 * math.sin(t * 2.3)
    # Lighting: gentle flicker plus a deeper dip during the low-light event.
    flicker = 1.0 + 0.035 * math.sin(t * 9.0)
    flags: dict[str, bool] = {}
    if 236 <= frame_idx <= 248:
        flicker *= 0.64 + 0.10 * math.sin(frame_idx * 0.9)
        flags["low_light"] = True
    frame = background(frame_idx, pan, flicker)
    gt_faces = []
    for track in sorted(tracks, key=lambda tr: tr.depth):
        x, y, size = track.position(frame_idx)
        if size <= 0:
            continue
        # Fast-entry tracks get directional motion blur and a slight smear.
        speed = track.speed if track.start <= frame_idx <= track.start + 22 else 0.0
        blur = 0.0
        if speed > 2.0:
            blur = min(3.2, speed * 0.22)
        if 252 <= frame_idx <= 262 and track.identity == "visitor_c":
            blur = 2.4
            flags["blur"] = True
        box = paste_face(frame, track.ref, x - pan * 0.85, y, int(size), angle=1.6 * math.sin(frame_idx * 0.05 + track.depth), blur=blur, depth_scale=1.0)
        if box[2] > 0:
            tx, ty, tw, th = track.tight
            gt_box = [
                int(box[0] + tx * box[2]),
                int(box[1] + ty * box[3]),
                int(tw * box[2]),
                int(th * box[3]),
            ]
            gt_faces.append({"id": track.identity, "box": gt_box, "known": track.identity in {"employee_a", "employee_b"}})
        if track.identity == "visitor_c" and 190 <= frame_idx <= 210:
            flags["fast_entry"] = True
        if track.identity == "employee_b" and 150 <= frame_idx <= 172:
            flags["occlusion"] = True

    # Foreground pillar: occludes employee_b for ~0.9 s.
    pillar_x = int(520 - pan * 1.15)
    cv2.rectangle(frame, (pillar_x, 70), (pillar_x + 46, H), (72, 76, 84), -1)
    cv2.rectangle(frame, (pillar_x + 6, 70), (pillar_x + 40, H), (92, 96, 104), -1)
    cv2.line(frame, (pillar_x + 22, 70), (pillar_x + 22, H), (110, 114, 122), 2)

    # Camera jerk associated with the fast entry.
    if 196 <= frame_idx <= 204:
        shift = int(4 * math.sin((frame_idx - 196) * 1.7))
        frame = np.roll(frame, shift, axis=1)
        gt_faces = [{**f, "box": [f["box"][0] + shift, f["box"][1], f["box"][2], f["box"][3]]} for f in gt_faces]

    # Vignette + sensor noise + timestamp, in that order.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1.0 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    frame = np.clip(frame.astype(np.float32) * vig[..., None], 0, 255).astype(np.uint8)
    rng = np.random.default_rng(1000 + frame_idx)
    frame = np.clip(frame.astype(np.int16) + rng.normal(0, 2.2, frame.shape).astype(np.int16), 0, 255).astype(np.uint8)
    stamp = f"2026-10-26 14:0{int(t // 60)}:{int(t % 60):02d}.{int((t % 1) * 100):02d}  CAM-07"
    cv2.putText(frame, stamp, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 235, 245), 1, cv2.LINE_AA)
    return frame, gt_faces, flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "demo" / "demo.mp4")
    parser.add_argument("--meta", type=Path, default=ROOT / "data" / "demo" / "demo_gt.json")
    parser.add_argument("--gallery", type=Path, default=ROOT / "data" / "demo" / "gallery")
    parser.add_argument("--annotated-out", type=Path, default=ROOT / "outputs" / "demo_annotated.mp4")
    parser.add_argument("--telemetry", type=Path, default=ROOT / "outputs" / "telemetry" / "demo.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "demo_summary.json")
    parser.add_argument("--skip-annotated", action="store_true", help="only compose the raw clip")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args()

    tref_a, tight_a = _ref_for("lena.jpg")
    tref_b, tight_b = _ref_for("messi5.jpg")
    tref_c, tight_c = _ref_for("ext_00.jpg")
    tracks = [
        Track("employee_a", tref_a, 5, 165, 60, 140, 158, 172, depth=0, tight=tight_a),
        Track("employee_b", tref_b, 60, 238, 720, 455, 170, 166, depth=1, tight=tight_b),
        Track("visitor_c", tref_c, 182, 287, 700, 610, 182, 182, depth=2, speed=9.0, tight=tight_c),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.gallery.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter")
    total = int(round(args.fps * args.seconds))
    meta = {
        "note": "Synthetic demo: licence-clean portraits composited onto a generated concourse.",
        "width": W,
        "height": H,
        "fps": args.fps,
        "frames": total,
        "gallery_identities": ["employee_a", "employee_b"],
        "unknown_identities": ["visitor_c"],
        "events": {
            "occlusion": [150, 172],
            "fast_entry": [188, 210],
            "low_light": [236, 248],
            "blur": [252, 262],
            "incident_expected": [190, 230],
        },
        "frames_meta": [],
    }
    preview_at = {60, 200, 240}
    for idx in range(total):
        frame, gt_faces, flags = compose(idx, tracks, args.fps)
        writer.write(frame)
        meta["frames_meta"].append({"frame": idx, "faces": gt_faces, "flags": sorted(flags)})
        if idx in preview_at:
            cv2.imwrite(str(frames_dir / f"preview_{idx:03d}.jpg"), frame)
    writer.release()

    for identity, suffix in (("employee_a", "lena.jpg"), ("employee_b", "messi5.jpg")):
        ref, _tight = _ref_for(suffix)
        cv2.imwrite(str(args.gallery / f"{identity}.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 96])
    (args.gallery / "consent.json").write_text(
        json.dumps({"labels": ["employee_a", "employee_b"], "basis": "synthetic demo identities; explicit consent recorded in repo"}, indent=2)
    )
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2))

    result = {"video": str(args.out), "frames": total, "gallery": str(args.gallery)}
    if not args.skip_annotated:
        # Run the real agent over the clip we just composed.  The annotated MP4
        # is therefore evidence of the shipped pipeline, not a hand-drawn reel.
        from src.config import load_config
        from src.engine import VisionEngine

        engine = VisionEngine(load_config())
        enrolled = engine.enroll_gallery(args.gallery)
        annotated = engine.run_video(
            args.out,
            args.annotated_out,
            args.telemetry,
            source_label=str(args.out.relative_to(ROOT)) if args.out.is_absolute() else str(args.out),
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(annotated, indent=2))
        result.update(
            {
                "annotated_video": str(args.annotated_out),
                "annotated_bytes": args.annotated_out.stat().st_size if args.annotated_out.exists() else 0,
                "telemetry": str(args.telemetry),
                "summary": str(args.summary),
                "enrolled": enrolled,
                "annotated_frames": annotated.get("frames"),
                "annotated_p50_ms": annotated.get("p50_ms"),
                "annotated_p95_ms": annotated.get("p95_ms"),
                "annotated_fps": annotated.get("fps_wall"),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
