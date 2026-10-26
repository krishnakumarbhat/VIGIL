"""Measure the optimisation levers VIGIL-5 can actually pull on this host.

Everything here is a local, reproducible measurement on the machine that runs
the submission.  It is deliberately *not* a Graviton benchmark and it does not
pretend to be one: see ``outputs/cool_evidence.json`` for the explicit
``status``/``reason`` pair and ``scripts/build_cool_evidence.py`` for the
arm64 model, which is labelled as a model.

Levers measured
---------------
1. Runtime SIMD dispatch: ``cv2.setUseOptimized(False/True)``.
2. OpenCV thread count: 1/2/4/8 on the same detector workload.
3. OpenCL: available device, and the cost delta if a device exists.
4. Spatial attention (ROI): full 640 px detection vs a 320 px crop.  The crop
   is derived from ground truth, so this is an *oracle upper bound* on the
   saving, not a deployment claim.
5. Kernel ordering: grayscale-before-resize vs resize-before-grayscale, and a
   cropped fused chain.  This is the local analogue of COOL-style kernel
   fusion: fewer pixels and fewer passes over them.
6. Energy proxy: CPU-time per frame (RAPL is not readable in this container),
   recorded explicitly as a proxy rather than joules.

Usage
-----
    python3 benchmarks/cool_local.py --out outputs/cool_local.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.arch_probe import MODEL, POSITIVES, operator_benchmark, percentile  # noqa: E402

OUT = ROOT / "outputs" / "cool_local.json"


def cpu_time_ms(fn, reps: int = 5) -> dict:
    """Return wall and CPU milliseconds for one call, median of repetitions."""
    fn()
    wall, cpu = [], []
    for _ in range(reps):
        c0 = time.process_time()
        t0 = time.perf_counter()
        fn()
        wall.append((time.perf_counter() - t0) * 1000.0)
        cpu.append((time.process_time() - c0) * 1000.0)
    return {"wall_ms": percentile(wall, 50), "cpu_ms": percentile(cpu, 50)}


def dispatch_interleaved(reps: int = 9) -> dict:
    """A/B/A/B dispatch measurement to cancel frequency and load drift.

    A naive "all optimized, then all baseline" loop confounds dispatch with
    CPU-frequency drift; the shared host makes that worse.  Here every
    repetition toggles both modes on the same input and the per-op medians are
    compared.  Results are reported per operator because the sign is not
    uniform: IPP-backed resize/Canny can be slower than the universal-intrinsics
    path on this build.
    """
    frame = cv2.resize(cv2.imread(str(next(POSITIVES.glob("*.jpg")))), (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ops = {
        "resize_2x": lambda: cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR),
        "gaussian_blur_9": lambda: cv2.GaussianBlur(frame, (9, 9), 2.0),
        "canny": lambda: cv2.Canny(gray, 80, 160),
        "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
        "farneback": lambda: cv2.calcOpticalFlowFarneback(gray, gray, None, 0.5, 3, 17, 3, 5, 1.2, 0),
    }
    results = {}
    for name, fn in ops.items():
        fn()
        on, off = [], []
        for _ in range(reps):
            cv2.setUseOptimized(True)
            t0 = time.perf_counter()
            fn()
            on.append((time.perf_counter() - t0) * 1000.0)
            cv2.setUseOptimized(False)
            t0 = time.perf_counter()
            fn()
            off.append((time.perf_counter() - t0) * 1000.0)
        cv2.setUseOptimized(True)
        results[name] = {
            "optimized_ms": percentile(on, 50),
            "baseline_ms": percentile(off, 50),
            "dispatch_speedup": round(percentile(off, 50) / max(1e-6, percentile(on, 50)), 3),
        }
    return {
        "ops": results,
        "mean_speedup": round(float(np.mean([v["dispatch_speedup"] for v in results.values()])), 3),
        "median_speedup": round(float(np.median([v["dispatch_speedup"] for v in results.values()])), 3),
        "note": "Interleaved A/B, 9 repetitions per operator, median; x86 SIMD/IPP dispatch, not Graviton.",
    }


def thread_scaling(limit: int = 12) -> dict:
    """Detection throughput as OpenCV's thread pool grows."""
    detector = cv2.FaceDetectorYN_create(str(MODEL), "", (640, 360), 0.6, 0.3, 12)
    frames = []
    for path in sorted(POSITIVES.glob("*.jpg"))[:limit]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = 640.0 / max(h, w)
        frames.append(cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA))
    results = {}
    for threads in (1, 2, 4, 8):
        cv2.setNumThreads(threads)
        latencies = []
        for frame in frames:
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            t0 = time.perf_counter()
            detector.detect(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        results[str(threads)] = {
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "fps": round(1000.0 / max(1e-6, float(np.mean(latencies))), 2),
            "frames": len(latencies),
        }
    cv2.setNumThreads(1)
    return results


def opencl_check() -> dict:
    """Report whether an OpenCL device exists; measure only if it does."""
    available = bool(cv2.ocl.haveOpenCL())
    enabled = False
    if available:
        try:
            cv2.ocl.setUseOpenCL(True)
            enabled = bool(cv2.ocl.useOpenCL())
        except Exception:
            enabled = False
    frame = cv2.resize(cv2.imread(str(next(POSITIVES.glob("*.jpg")))), (640, 360))
    cpu_ms = cpu_time_ms(lambda: cv2.GaussianBlur(frame, (9, 9), 2.0))
    gpu_ms = None
    if enabled:
        umat = cv2.UMat(frame)
        gpu_ms = cpu_time_ms(lambda: cv2.GaussianBlur(umat, (9, 9), 2.0))
        cv2.ocl.setUseOpenCL(False)
    return {
        "available": available,
        "enabled": enabled,
        "device": cv2.ocl.Device_getDefault().name() if enabled else None,
        "cpu_blur_ms": cpu_ms,
        "opencl_blur_ms": gpu_ms,
        "note": "No OpenCL device is exposed in this runtime; no GPU claim is made."
        if not enabled
        else "OpenCL device present; measured on the same 640x360 frame.",
    }


def _match(box: list[int], gt: list[int], threshold: float = 0.5) -> bool:
    ax, ay, aw, ah = box
    bx, by, bw, bh = gt
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return union > 0 and inter / union >= threshold


def spoof_roi_benchmark(limit: int = 40) -> dict:
    """Full-frame 640 detection vs oracle 320 crop, with measured recall.

    The crop is centred on the labelled face with 2.2x padding.  It shows what
    spatial attention can save; it is *not* free, because a production system
    has to propose that crop from motion/quality evidence (which VIGIL-5 does).
    """
    meta = json.loads((ROOT / "data" / "eval" / "detection" / "labels.json").read_text())
    items = [i for i in meta["items"] if i["has_face"]][:limit]
    detector = cv2.FaceDetectorYN_create(str(MODEL), "", (640, 360), 0.6, 0.3, 12)
    full_lat, roi_lat = [], []
    full_hits = roi_hits = 0
    for item in items:
        frame = cv2.imread(str(Path(meta["root"]) / item["file"]))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = 640.0 / max(h, w)
        full = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        detector.setInputSize((full.shape[1], full.shape[0]))
        t0 = time.perf_counter()
        _, faces = detector.detect(full)
        full_lat.append((time.perf_counter() - t0) * 1000.0)
        if faces is not None:
            boxes = [[int(v) for v in f[:4]] for f in faces]
            full_hits += any(_match([int(b[0] / scale), int(b[1] / scale), int(b[2] / scale), int(b[3] / scale)], item["box"]) for b in boxes)

        gx, gy, gw, gh = item["box"]
        pad = 1.1
        x0 = max(0, int(gx - gw * pad))
        y0 = max(0, int(gy - gh * pad))
        x1 = min(w, int(gx + gw * (1 + pad)))
        y1 = min(h, int(gy + gh * (1 + pad)))
        crop = frame[y0:y1, x0:x1]
        cscale = 320.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=cscale, fy=cscale, interpolation=cv2.INTER_AREA)
        detector.setInputSize((crop.shape[1], crop.shape[0]))
        t0 = time.perf_counter()
        _, faces = detector.detect(crop)
        roi_lat.append((time.perf_counter() - t0) * 1000.0)
        if faces is not None:
            boxes = [
                [int(f[0] / cscale) + x0, int(f[1] / cscale) + y0, int(f[2] / cscale), int(f[3] / cscale)]
                for f in faces
            ]
            roi_hits += any(_match(b, item["box"]) for b in boxes)

    n = len(full_lat)
    return {
        "n_images": n,
        "full_640": {
            "p50_ms": percentile(full_lat, 50),
            "p95_ms": percentile(full_lat, 95),
            "fps": round(1000.0 / max(1e-6, float(np.mean(full_lat))), 2),
            "recall": round(full_hits / n, 4) if n else None,
        },
        "roi_320_oracle": {
            "p50_ms": percentile(roi_lat, 50),
            "p95_ms": percentile(roi_lat, 95),
            "fps": round(1000.0 / max(1e-6, float(np.mean(roi_lat))), 2),
            "recall": round(roi_hits / n, 4) if n else None,
        },
        "latency_speedup": round(float(np.mean(full_lat)) / max(1e-6, float(np.mean(roi_lat))), 3),
        "caveat": "Oracle crop from ground truth; an upper bound on the saving, not a deployment claim.",
    }


def kernel_chain_benchmark() -> dict:
    """Same pixels, different pass order / working-set size."""
    frame = cv2.resize(cv2.imread(str(next(POSITIVES.glob("*.jpg")))), (640, 360), interpolation=cv2.INTER_AREA)

    def gray_first():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 1.2)

    def resize_first():
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 1.2)

    def cropped_chain():
        crop = frame[90:270, 160:480]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 1.2)

    gray = cpu_time_ms(gray_first)
    order = cpu_time_ms(resize_first)
    crop = cpu_time_ms(cropped_chain)
    ref = gray_first()
    return {
        "gray_then_resize": gray,
        "resize_then_gray": order,
        "roi_bgr_to_gray_resize_blur": crop,
        "resize_first_speedup": round(gray["wall_ms"] / max(1e-6, order["wall_ms"]), 3),
        "roi_speedup": round(gray["wall_ms"] / max(1e-6, crop["wall_ms"]), 3),
        "identical_output": bool(np.array_equal(ref, resize_first())),
        "note": "Resizing BGR before grayscale is not bit-identical in general; the crop variant is a different ROI, not a different result for the same ROI.",
    }


def energy_proxy(wall_cpu: dict) -> dict:
    """Report CPU-time as an energy proxy; never invent joules."""
    rapl = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    readable = rapl.exists()
    if readable:
        try:
            rapl.read_text()
        except OSError:
            readable = False
    return {
        "rapl_energy_uj_readable": readable,
        "cpu_ms_per_gaussian_blur_640x360": wall_cpu["cpu_ms"],
        "note": "RAPL counters are not readable in this container, so energy is reported as CPU-time, not joules.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local OpenCV optimisation measurements")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    frame = cv2.resize(cv2.imread(str(next(POSITIVES.glob("*.jpg")))), (640, 360))
    blur = cpu_time_ms(lambda: cv2.GaussianBlur(frame, (9, 9), 2.0))

    payload = {
        "schema": "vigil.cool_local/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "opencv_module": cv2.__file__,
            "numpy": np.__version__,
            "cpu_count_logical": os.cpu_count(),
            "load_average_1m": round(os.getloadavg()[0], 2),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
        "dispatch": dispatch_interleaved(),
        "threads": thread_scaling(),
        "opencl": opencl_check(),
        "spatial_attention": spoof_roi_benchmark(),
        "kernel_order": kernel_chain_benchmark(),
        "energy": energy_proxy(blur),
        "commands": {
            "local": "python3 benchmarks/cool_local.py --out outputs/cool_local.json",
            "full": "python3 benchmarks/run_eval.py",
            "arm64_emulated": "bash scripts/run_arm64_emulated.sh",
            "arm64_native": "COOL_WHEEL_URL=<url> bash aws/run_cool_bench.sh arm64",
        },
        "caveats": [
            "All timings are x86_64 local measurements on a shared host; load average is recorded.",
            "No COOL binary was available; no COOL speedup is claimed.",
            "The ROI result is an oracle upper bound derived from ground-truth boxes.",
            "Energy is CPU-time, not joules: RAPL is not readable here.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "out": str(args.out),
        "dispatch_mean_speedup": payload["dispatch"]["mean_speedup"],
        "threads_4_fps": payload["threads"]["4"]["fps"],
        "roi_latency_speedup": payload["spatial_attention"]["latency_speedup"],
        "roi_recall_full": payload["spatial_attention"]["full_640"]["recall"],
        "roi_recall_crop": payload["spatial_attention"]["roi_320_oracle"]["recall"],
        "kernel_resize_first_speedup": payload["kernel_order"]["resize_first_speedup"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
