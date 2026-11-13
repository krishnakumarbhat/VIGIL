"""Architecture probe: one OpenCV workload, run unmodified on x86_64 and arm64.

Purpose
-------
``run_eval.py`` is the full submission benchmark.  This probe is the small,
portable subset that can also run inside an emulated aarch64 userland
(QEMU user-mode) without a Graviton host.  It answers two questions with
*measured* data:

1. Does the same OpenCV 5 code path produce the same detections on aarch64?
2. What does each operator cost on each architecture in this exact runtime?

It does **not** estimate Graviton performance.  QEMU translates instructions
and its timings are not representative of Apple/Ampere/AWS silicon; the JSON
records ``emulated: true`` and the report must label it that way.  The Graviton
model in ``scripts/build_cool_evidence.py`` is a separate, explicitly
simulated artefact.

Usage
-----
    # native x86_64
    python3 benchmarks/arch_probe.py --out outputs/arch_probe_x86_64.json

    # aarch64 under qemu-user (see scripts/run_arm64_emulated.sh)
    PYTHONPATH=/tmp/opencode/armlibs qemu-aarch64-static -L /tmp/opencode/armroot \
        /tmp/opencode/armpy/bin/python3.11 benchmarks/arch_probe.py \
        --out outputs/arch_probe_aarch64.json --emulated
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
POSITIVES = ROOT / "data" / "eval" / "detection" / "positive"


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile with linear interpolation between neighbours."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo), 3)


def timed(fn, reps: int = 5, iters: int = 3) -> dict:
    """Median-of-reps wall time per single call, in milliseconds."""
    fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - start) / iters * 1000.0)
    return {"p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95), "reps": reps}


def opencv_environment(emulated: bool) -> dict:
    build = cv2.getBuildInformation()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "opencv_module": cv2.__file__,
        "numpy": np.__version__,
        "emulated": emulated,
        "emulation_note": (
            "QEMU user-mode aarch64; timings reflect translation overhead and are "
            "not Graviton performance. Functional equivalence only."
            if emulated
            else "Native execution."
        ),
        "build_info_sha256": hashlib.sha256(build.encode()).hexdigest()[:16],
        "baseline": [l.strip() for l in build.splitlines() if "Baseline:" in l],
        "dispatch": [l.strip() for l in build.splitlines() if "Dispatched code generation" in l],
        "ipp": [l.strip() for l in build.splitlines() if "Intel IPP:" in l],
        "parallel": [l.strip() for l in build.splitlines() if "Parallel framework" in l],
        "opencl_available": bool(cv2.ocl.haveOpenCL()),
        "threads": cv2.getNumThreads(),
        "load_average_1m": round(os.getloadavg()[0], 2),
    }


def operator_benchmark() -> dict:
    """Five representative operators, runtime dispatch on vs off."""
    frame = cv2.imread(str(next(POSITIVES.glob("*.jpg"))))
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ops = {
        "resize_2x": lambda: cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR),
        "gaussian_blur_9": lambda: cv2.GaussianBlur(frame, (9, 9), 2.0),
        "canny": lambda: cv2.Canny(gray, 80, 160),
        "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
        "farneback": lambda: cv2.calcOpticalFlowFarneback(gray, gray, None, 0.5, 3, 17, 3, 5, 1.2, 0),
    }
    results: dict[str, dict] = {}
    for optimized in (True, False):
        cv2.setUseOptimized(optimized)
        for name, fn in ops.items():
            entry = results.setdefault(name, {})
            entry["optimized_ms" if optimized else "baseline_ms"] = timed(fn)["p50_ms"]
    cv2.setUseOptimized(True)
    for entry in results.values():
        entry["dispatch_speedup"] = (
            round(entry["baseline_ms"] / entry["optimized_ms"], 3) if entry["optimized_ms"] else None
        )
    return results


def detection_benchmark(limit: int = 8) -> dict:
    """YuNet at 640 px on labelled positives; latency plus detection counts."""
    detector = cv2.FaceDetectorYN_create(str(MODEL), "", (640, 360), 0.6, 0.3, 12)
    images = sorted(POSITIVES.glob("*.jpg"))[:limit]
    latencies, counts = [], []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = 640.0 / max(h, w)
        resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        detector.setInputSize((resized.shape[1], resized.shape[0]))

        def run() -> None:
            _, faces = detector.detect(resized)
            counts.append(0 if faces is None else int(len(faces)))

        start = time.perf_counter()
        run()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "model": MODEL.name,
        "images": len(latencies),
        "input_side": 640,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "mean_ms": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "fps": round(1000.0 / max(1e-6, float(np.mean(latencies))), 2) if latencies else 0.0,
        "detections_total": int(sum(counts)),
        "detections_per_image": round(float(np.mean(counts)), 3) if counts else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portable OpenCV architecture probe")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emulated", action="store_true", help="mark results as qemu-user emulated")
    parser.add_argument("--detection-images", type=int, default=8)
    args = parser.parse_args(argv)

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    machine = platform.machine() or "unknown"
    out = args.out or ROOT / "outputs" / f"arch_probe_{machine}.json"
    payload = {
        "schema": "vigil.arch_probe/1.0",
        "environment": opencv_environment(args.emulated),
        "operators": operator_benchmark(),
        "detection_yunet_640": detection_benchmark(args.detection_images),
        "commands": {
            "native": "python3 benchmarks/arch_probe.py --out outputs/arch_probe_<machine>.json",
            "arm64_emulated": "bash scripts/run_arm64_emulated.sh",
        },
    }
    payload["environment"]["threads"] = 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "machine": machine,
        "emulated": args.emulated,
        "opencv": payload["environment"]["opencv"],
        "detection_p50_ms": payload["detection_yunet_640"]["p50_ms"],
        "detection_fps": payload["detection_yunet_640"]["fps"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
