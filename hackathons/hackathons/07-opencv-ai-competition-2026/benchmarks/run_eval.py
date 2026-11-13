"""Reproducible local evaluation for VIGIL-5.

What is measured (all on this machine, no fabricated numbers):

1. Face detection on a labelled synthetic corpus (100 images, exact boxes):
   precision/recall/F1, ROC-AUC of confidence, recall by face size.
2. Recognition on 30 genuine / 30 impostor pairs: ROC-AUC, TAR@FAR=1e-2 and
   accuracy at the configured SFace cosine threshold.
3. End-to-end agent on the 288-frame demo clip in three configurations:
   static 320px detector / adaptive agent without forensics / full agent.
   Wall FPS, p50/p95 latency, RSS, detection P/R/F1, identity accuracy,
   event-level trigger recall and false-trigger rate.
4. CPU dispatch scaling: ``cv2.setUseOptimized`` off vs on (SIMD dispatch),
   and OpenCV thread scaling 1/2/4.

Results go to ``outputs/benchmarks.json`` and ``outputs/benchmarks.csv`` and a
few rows are appended to ``logs/experiments.tsv``.

The host is shared (load average > 30 during development), so the benchmark
pins itself to four performance cores with ``sched_setaffinity`` and reports
the observed spread instead of a single best-case number.  That limitation is
also stated in the submission report.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.agent import AutonomousVisionAgent  # noqa: E402
from src.cloud import OfflineCloud  # noqa: E402
from src.config import VisionConfig, load_config  # noqa: E402
from src.perception.face import FacePipeline  # noqa: E402
from src.perception.motion import MotionAnalyzer  # noqa: E402
from src.telemetry import summarise_telemetry  # noqa: E402

OUT_JSON = ROOT / "outputs" / "benchmarks.json"
OUT_CSV = ROOT / "outputs" / "benchmarks.csv"
LOG_TSV = ROOT / "logs" / "experiments.tsv"
DEMO = ROOT / "data" / "demo" / "demo.mp4"
DEMO_GT = ROOT / "data" / "demo" / "demo_gt.json"
EVAL = ROOT / "data" / "eval"

# Every timing block is measured more than once.  The shared host has a noisy
# load average, so the report keeps the best repetition (least contention) and
# the observed spread rather than pretending a single run is stable.
REPS = max(1, int(os.environ.get("VIGIL_BENCH_REPS", "3")))


# --------------------------------------------------------------------- utils
def pin_cores(cores: int = 4) -> list[int]:
    """Pin to the first N logical CPUs (P-cores on hybrid laptops)."""
    try:
        available = sorted(os.sched_getaffinity(0))
        chosen = available[:cores]
        os.sched_setaffinity(0, chosen)
        return chosen
    except Exception:
        return []


def rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 2)


def percentile(values: Iterable[float], p: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    idx = min(len(data) - 1, max(0, int(round(p / 100.0 * len(data) + 0.5)) - 1))
    return round(float(data[idx]), 3)


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based Mann-Whitney AUC with average ranks for ties."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks = np.zeros(len(pairs), dtype=np.float64)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    labels_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    n_pos = float(labels_arr.sum())
    n_neg = float(len(labels_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels_arr == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_faces(gt: list[dict], pred: list[dict], threshold: float = 0.35) -> dict:
    """Greedy IoU matching; returns TP/FP/FN and per-match identity stats."""
    used: set[int] = set()
    tp, id_correct, id_total = 0, 0, 0
    ious = []
    for target in gt:
        best_idx, best_iou = -1, threshold
        for idx, face in enumerate(pred):
            if idx in used:
                continue
            score = iou(target["box"], face["box"])
            if score >= best_iou:
                best_idx, best_iou = idx, score
        if best_idx >= 0:
            used.add(best_idx)
            tp += 1
            ious.append(best_iou)
            face = pred[best_idx]
            if target.get("known") and face.get("identity") is not None:
                id_total += 1
                if face.get("identity") == target.get("id"):
                    id_correct += 1
    return {
        "tp": tp,
        "fp": len(pred) - len(used),
        "fn": len(gt) - tp,
        "mean_iou": round(float(np.mean(ious)), 3) if ious else 0.0,
        "id_correct": id_correct,
        "id_total": id_total,
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# --------------------------------------------------------------- environment
def environment(cores: list[int]) -> dict:
    build = cv2.getBuildInformation()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "opencv_module": cv2.__file__,
        "numpy": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "pinned_cores": cores,
        "load_average_1m": round(os.getloadavg()[0], 2),
        "opencv_threads": cv2.getNumThreads(),
        "opencl": bool(cv2.ocl.haveOpenCL()),
        "simd": [line.strip() for line in build.splitlines() if "Dispatched code generation" in line],
        "ipp": [line.strip() for line in build.splitlines() if "Intel IPP:" in line],
        "build_info_sha256": __import__("hashlib").sha256(build.encode()).hexdigest()[:16],
    }


# ------------------------------------------------------------------ detection
def _detection_pass(cfg: VisionConfig, pipeline: FacePipeline) -> dict:
    """One full pass over the labelled corpus with wall and CPU timings."""
    meta = json.loads((EVAL / "detection" / "labels.json").read_text())
    items = meta["items"]
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    tp = fp = fn = 0
    size_bins = {"<48": [0, 0], "48-80": [0, 0], "80-128": [0, 0], ">=128": [0, 0]}
    frame_latencies = []
    cpu_latencies = []
    per_image = []
    for item in items:
        path = Path(meta["root"]) / item["file"]
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        c0 = time.process_time()
        t0 = time.perf_counter()
        hits = pipeline.detect(frame, target_side=640)
        frame_latencies.append((time.perf_counter() - t0) * 1000.0)
        cpu_latencies.append((time.process_time() - c0) * 1000.0)
        pred = [{"box": [int(v) for v in b], "score": s} for b, s, _ in hits]
        pred.sort(key=lambda d: -d["score"])
        if item["has_face"]:
            gt_box = item["box"]
            best, best_iou = None, 0.5
            for p in pred:
                score = iou(gt_box, p["box"])
                if score >= best_iou:
                    best, best_iou = p, score
            pos_scores.append(best["score"] if best else 0.0)
            bin_name = (
                "<48" if item["size"] < 48 else "48-80" if item["size"] < 80 else "80-128" if item["size"] < 128 else ">=128"
            )
            size_bins[bin_name][1] += 1
            if best:
                tp += 1
                size_bins[bin_name][0] += 1
            else:
                fn += 1
            neg_boxes = []
            rng = np.random.default_rng(hash(item["file"]) % 2**32)
            for _ in range(3):
                for _try in range(20):
                    w = int(rng.integers(32, 96))
                    h = int(rng.integers(32, 96))
                    x = int(rng.integers(0, max(1, frame.shape[1] - w)))
                    y = int(rng.integers(0, max(1, frame.shape[0] - h)))
                    candidate = [x, y, w, h]
                    if iou(candidate, gt_box) < 0.05:
                        neg_boxes.append(candidate)
                        break
            for box in neg_boxes:
                inside = [p["score"] for p in pred if iou(box, p["box"]) > 0.1]
                neg_scores.append(max(inside) if inside else 0.0)
        else:
            for p in pred:
                neg_scores.append(p["score"])
                fp += 1
            if not pred:
                neg_scores.append(0.0)
        per_image.append({"file": item["file"], "detections": len(pred), "has_face": item["has_face"]})
    auc = roc_auc(pos_scores + neg_scores, [1] * len(pos_scores) + [0] * len(neg_scores))
    metrics = prf(tp, fp, fn)
    bins = {k: {"recall": round(v[0] / v[1], 3) if v[1] else None, "n": v[1]} for k, v in size_bins.items()}
    return {
        "n_images": len(items),
        "n_positive": sum(1 for i in items if i["has_face"]),
        "n_negative": sum(1 for i in items if not i["has_face"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **metrics,
        "roc_auc": round(auc, 4),
        "recall_by_size": bins,
        "latency_ms_p50": percentile(frame_latencies, 50),
        "latency_ms_p95": percentile(frame_latencies, 95),
        "latency_ms_mean": round(float(np.mean(frame_latencies)), 3),
        "cpu_latency_ms_p50": percentile(cpu_latencies, 50),
        "cpu_latency_ms_p95": percentile(cpu_latencies, 95),
        "cpu_fps": round(1000.0 / max(1e-6, float(np.mean(cpu_latencies))), 2),
        "throughput_fps": round(1000.0 / max(1e-6, statistics.mean(frame_latencies)), 2),
        "threshold": cfg.det_threshold,
        "per_image": per_image,
    }


def detection_benchmark(cfg: VisionConfig, pipeline: FacePipeline, reps: int = REPS) -> dict:
    """Repeat the corpus pass and keep the pass with the lowest CPU p50.

    CPU time is far less sensitive to competing processes than wall time, so it
    selects the representative pass; both timing domains are still reported.
    """
    runs = [_detection_pass(cfg, pipeline) for _ in range(max(1, reps))]
    best = min(runs, key=lambda r: r["cpu_latency_ms_p50"])
    best["repetitions"] = [
        {
            "latency_ms_p50": run["latency_ms_p50"],
            "latency_ms_p95": run["latency_ms_p95"],
            "cpu_latency_ms_p50": run["cpu_latency_ms_p50"],
            "cpu_latency_ms_p95": run["cpu_latency_ms_p95"],
            "throughput_fps": run["throughput_fps"],
        }
        for run in runs
    ]
    best["wall_p50_spread_ms"] = [
        min(r["latency_ms_p50"] for r in runs),
        max(r["latency_ms_p50"] for r in runs),
    ]
    best["cpu_p50_spread_ms"] = [
        min(r["cpu_latency_ms_p50"] for r in runs),
        max(r["cpu_latency_ms_p50"] for r in runs),
    ]
    return best


# ---------------------------------------------------------------- recognition
def _recognition_pass(cfg: VisionConfig, pipeline: FacePipeline) -> dict:
    """Embed every evaluation pair once, timing both images of each pair."""
    meta = json.loads((EVAL / "recognition_pairs.json").read_text())
    scores, labels = [], []
    latencies, cpu_latencies = [], []
    for pair in meta["pairs"]:
        a = cv2.imread(str(EVAL / "recognition" / f"a_{pair['index']:03d}.jpg"))
        b = cv2.imread(str(EVAL / "recognition" / f"b_{pair['index']:03d}.jpg"))
        if a is None or b is None:
            continue
        pipeline.reset_tracking()
        c0 = time.process_time()
        t0 = time.perf_counter()
        ha = pipeline.analyze(a, embed=True, target_side=640)
        pipeline.reset_tracking()
        hb = pipeline.analyze(b, embed=True, target_side=640)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        cpu_latencies.append((time.process_time() - c0) * 1000.0)
        if not ha or not hb:
            continue
        va = max(ha, key=lambda h: h.score).embedding
        vb = max(hb, key=lambda h: h.score).embedding
        if va is None or vb is None:
            continue
        scores.append(float(np.dot(va, vb)))
        labels.append(1 if pair["genuine"] else 0)
    return {"scores": scores, "labels": labels, "latencies": latencies, "cpu_latencies": cpu_latencies}


def recognition_benchmark(cfg: VisionConfig, pipeline: FacePipeline, reps: int = REPS) -> dict:
    runs = [_recognition_pass(cfg, pipeline) for _ in range(max(1, reps))]
    best_run = min(runs, key=lambda r: percentile(r["cpu_latencies"], 50))
    scores, labels = best_run["scores"], best_run["labels"]
    latencies, cpu_latencies = best_run["latencies"], best_run["cpu_latencies"]
    auc = roc_auc(scores, labels)
    pos = np.asarray([s for s, l in zip(scores, labels) if l == 1])
    neg = np.asarray([s for s, l in zip(scores, labels) if l == 0])
    far_target = 0.01
    thresholds = sorted(set(scores), reverse=True)
    tar_at_far = 0.0
    best_thr = cfg.recognition_threshold
    for thr in thresholds:
        far = float(np.mean(neg >= thr)) if len(neg) else 0.0
        if far <= far_target:
            tar = float(np.mean(pos >= thr)) if len(pos) else 0.0
            if tar >= tar_at_far:
                tar_at_far, best_thr = tar, float(thr)
    accuracy = float(np.mean([(s >= cfg.recognition_threshold) == bool(l) for s, l in zip(scores, labels)]))
    return {
        "n_pairs": len(labels),
        "n_genuine": int(sum(labels)),
        "n_impostor": int(len(labels) - sum(labels)),
        "roc_auc": round(auc, 4),
        "tar_at_far_1e-2": round(tar_at_far, 4),
        "threshold_at_far_1e-2": round(best_thr, 4),
        "accuracy_at_config_threshold": round(accuracy, 4),
        "config_threshold": cfg.recognition_threshold,
        "genuine_mean": round(float(pos.mean()), 4) if len(pos) else None,
        "impostor_mean": round(float(neg.mean()), 4) if len(neg) else None,
        "embedding_latency_ms_p50": percentile(latencies, 50),
        "embedding_latency_ms_p95": percentile(latencies, 95),
        "embedding_cpu_ms_p50": percentile(cpu_latencies, 50),
        "embedding_cpu_ms_p95": percentile(cpu_latencies, 95),
        "embedding_dim": pipeline.descriptor_dim,
        "repetitions": [
            {
                "latency_ms_p50": percentile(run["latencies"], 50),
                "cpu_ms_p50": percentile(run["cpu_latencies"], 50),
                "cpu_ms_p95": percentile(run["cpu_latencies"], 95),
            }
            for run in runs
        ],
        "cpu_p50_spread_ms": [
            min(percentile(r["cpu_latencies"], 50) for r in runs),
            max(percentile(r["cpu_latencies"], 50) for r in runs),
        ],
    }


# ----------------------------------------------------------------------- agent
def _frames() -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(DEMO))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _run_config(name: str, frames: list[np.ndarray], gt: dict, cfg: VisionConfig, mode: str) -> dict:
    rss_before = rss_mb()
    if mode == "static":
        pipeline = FacePipeline(cfg)
        # A baseline detector at a fixed low resolution, no motion, no identity.
        records, latencies, cpu_latencies, preds = [], [], [], []
        for idx, frame in enumerate(frames):
            t0 = time.perf_counter()
            c0 = time.process_time()
            hits = pipeline.analyze(frame, embed=False, target_side=320)
            ms = (time.perf_counter() - t0) * 1000.0
            cpu_ms = (time.process_time() - c0) * 1000.0
            cpu_latencies.append(cpu_ms)
            latencies.append(ms)
            preds.append([{"box": list(h.box), "identity": None, "score": h.score} for h in hits])
            records.append({"processing_ms": ms, "agent": {"forensic": False, "escalated": False}, "faces": [{"identity": None}] * len(hits)})
        metrics = _task_metrics(preds, gt, records)
        return {
            "config": name,
            "mode": mode,
            "frames": len(frames),
            "wall_fps": round(len(frames) / max(1e-6, sum(latencies) / 1000.0), 2),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "latency_mean_ms": round(float(np.mean(latencies)), 3),
            "cpu_latency_p50_ms": percentile(cpu_latencies, 50),
            "cpu_latency_p95_ms": percentile(cpu_latencies, 95),
            "cpu_fps": round(1000.0 / max(1e-6, float(np.mean(cpu_latencies))), 2),
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_mb(),
            "rss_delta_mb": round(rss_mb() - rss_before, 1),
            **metrics,
        }

    pipeline = FacePipeline(cfg)
    gallery_dir = ROOT / "data" / "demo" / "gallery"
    consent = json.loads((gallery_dir / "consent.json").read_text()).get("labels", [])
    for ref_path in sorted(gallery_dir.glob("*.jpg")):
        if ref_path.stem not in consent:
            continue
        reference = cv2.imread(str(ref_path))
        pipeline.reset_tracking()
        hits = pipeline.analyze(reference, embed=True, target_side=640)
        if hits:
            pipeline.enroll(ref_path.stem, max(hits, key=lambda h: h.score).embedding, consent=True)
    pipeline.reset_tracking()
    cloud = OfflineCloud(cfg)
    agent = AutonomousVisionAgent(
        cfg,
        pipeline,
        motion=MotionAnalyzer(scale=cfg.flow_scale, anomaly_z=cfg.motion_anomaly_z),
        cloud=cloud.__call__,
    )
    telemetry, preds = [], []
    for idx, frame in enumerate(frames):
        outcome = agent.process(frame, frame_index=idx)
        telemetry.append(json.loads(outcome.telemetry.model_dump_json()))
        preds.append([{"box": list(f.box), "identity": f.identity, "score": f.score} for f in outcome.faces])
    latencies = [r["processing_ms"] for r in telemetry]
    cpu_latencies = [r.get("cpu_ms", r["processing_ms"]) for r in telemetry]
    metrics = _task_metrics(preds, gt, telemetry)
    return {
        "config": name,
        "mode": mode,
        "frames": len(frames),
        "wall_fps": round(len(frames) / max(1e-6, sum(latencies) / 1000.0), 2),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
        "cpu_latency_p50_ms": percentile(cpu_latencies, 50),
        "cpu_latency_p95_ms": percentile(cpu_latencies, 95),
        "cpu_fps": round(1000.0 / max(1e-6, float(np.mean(cpu_latencies))), 2),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_mb(),
        "rss_delta_mb": round(rss_mb() - rss_before, 1),
        "forensic_frames": sum(1 for r in telemetry if r["agent"]["forensic"]),
        "escalations": sum(1 for r in telemetry if r["agent"]["escalated"]),
        "adaptive_steps_total": sum(r["agent"]["adaptive_steps"] for r in telemetry),
        "verification_pass_rate": round(
            float(np.mean([r["verification"]["passed"] for r in telemetry])), 4
        ),
        **metrics,
    }


def _task_metrics(preds: list[list[dict]], gt: dict, telemetry: list[dict]) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0, "id_correct": 0, "id_total": 0}
    per_frame_iou = []
    for idx, (pred, meta) in enumerate(zip(preds, gt["frames_meta"])):
        m = match_faces(meta["faces"], pred, threshold=0.35)
        for key in totals:
            totals[key] += m[key]
        if m["tp"]:
            per_frame_iou.append(m["mean_iou"])
    metrics = prf(totals["tp"], totals["fp"], totals["fn"])
    metrics["tp"] = totals["tp"]
    metrics["fp"] = totals["fp"]
    metrics["fn"] = totals["fn"]
    metrics["identity_accuracy"] = (
        round(totals["id_correct"] / totals["id_total"], 4) if totals["id_total"] else None
    )
    metrics["mean_iou"] = round(float(np.mean(per_frame_iou)), 3) if per_frame_iou else 0.0

    windows = []
    for key, (start, end) in gt["events"].items():
        if key == "incident_expected":
            continue
        windows.append((key, start, end))
    trigger_frames = {
        idx
        for idx, record in enumerate(telemetry)
        if record.get("agent", {}).get("forensic") or record.get("agent", {}).get("escalated")
    }
    event_hits = {}
    for key, start, end in windows:
        event_hits[key] = any(start - 2 <= f <= end + 6 for f in trigger_frames)
    union = set()
    for _, start, end in windows:
        union.update(range(start, end + 1))
    if "incident_expected" in gt["events"]:
        start, end = gt["events"]["incident_expected"]
        union.update(range(start, end + 1))
    tp_events = sum(1 for f in trigger_frames if f in union)
    precision = tp_events / len(trigger_frames) if trigger_frames else 0.0
    outside = [f for f in trigger_frames if f not in union]
    denom = max(1, len(telemetry) - len(union))
    metrics["event_recall"] = round(sum(event_hits.values()) / max(1, len(event_hits)), 4)
    metrics["event_hits"] = event_hits
    metrics["trigger_precision"] = round(precision, 4)
    metrics["false_trigger_rate"] = round(len(outside) / denom, 4)
    metrics["trigger_frames"] = len(trigger_frames)
    return metrics


# -------------------------------------------------------------- dispatch/COOL
def dispatch_benchmark(cfg: VisionConfig) -> dict:
    frame = cv2.imread(str(next((EVAL / "detection" / "positive").glob("*.jpg"))))
    frame = cv2.resize(frame, (640, 360))
    ops = {
        "resize_2x": lambda: cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR),
        "gaussian_blur_9": lambda: cv2.GaussianBlur(frame, (9, 9), 2.0),
        "canny": lambda: cv2.Canny(frame, 80, 160),
        "farneback": lambda: cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 17, 3, 5, 1.2, 0
        ),
        "laplacian": lambda: cv2.Laplacian(frame, cv2.CV_64F),
    }
    results = {}
    cv2.setUseOptimized(True)
    for name, fn in ops.items():
        fn()
        timings = []
        for _ in range(8):
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            timings.append((time.perf_counter() - t0) / 5 * 1000.0)
        results[name] = {"optimized_ms": percentile(timings, 50)}
    cv2.setUseOptimized(False)
    for name, fn in ops.items():
        fn()
        timings = []
        for _ in range(8):
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            timings.append((time.perf_counter() - t0) / 5 * 1000.0)
        base = percentile(timings, 50)
        opt = results[name]["optimized_ms"]
        results[name].update(
            {
                "baseline_ms": base,
                "speedup": round(base / opt, 3) if opt else None,
            }
        )
    cv2.setUseOptimized(True)
    return {
        "note": "x86 SIMD dispatch measurement; COOL/Graviton evidence is produced separately on Arm.",
        "ops": results,
        "mean_speedup": round(float(np.mean([v["speedup"] for v in results.values() if v["speedup"]])), 3),
    }


def thread_benchmark(frames: list[np.ndarray], cfg: VisionConfig, reps: int = REPS) -> dict:
    subset = frames[10:70]
    out = {}
    for threads in (1, 2, 4):
        runs = []
        for _ in range(max(1, reps)):
            cv2.setNumThreads(threads)
            pipeline = FacePipeline(cfg)
            motion = MotionAnalyzer(scale=cfg.flow_scale)
            latencies, cpu_latencies = [], []
            for idx, frame in enumerate(subset):
                c0 = time.process_time()
                t0 = time.perf_counter()
                motion.update(frame, idx)
                pipeline.analyze(frame, motion=motion, target_side=512)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                cpu_latencies.append((time.process_time() - c0) * 1000.0)
            runs.append({"latencies": latencies, "cpu_latencies": cpu_latencies})
        best = min(runs, key=lambda r: percentile(r["cpu_latencies"], 50))
        out[str(threads)] = {
            "p50_ms": percentile(best["latencies"], 50),
            "p95_ms": percentile(best["latencies"], 95),
            "fps": round(1000.0 / max(1e-6, float(np.mean(best["latencies"]))), 2),
            "cpu_p50_ms": percentile(best["cpu_latencies"], 50),
            "cpu_fps": round(1000.0 / max(1e-6, float(np.mean(best["cpu_latencies"]))), 2),
            "cpu_p50_spread_ms": [
                min(percentile(r["cpu_latencies"], 50) for r in runs),
                max(percentile(r["cpu_latencies"], 50) for r in runs),
            ],
            "wall_fps_repetitions": [
                round(1000.0 / max(1e-6, float(np.mean(r["latencies"]))), 2) for r in runs
            ],
        }
    cv2.setNumThreads(cfg.threads)
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    cores = pin_cores(4)
    cfg = load_config()
    cv2.setNumThreads(cfg.threads)
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(cfg.seed)

    started = time.perf_counter()
    rss_start = rss_mb()
    pipeline = FacePipeline(cfg)
    model_load_s = time.perf_counter() - started
    rss_after_models = rss_mb()

    detection = detection_benchmark(cfg, pipeline)
    recognition = recognition_benchmark(cfg, pipeline)
    gt = json.loads(DEMO_GT.read_text())
    frames = _frames()
    # Warm the runtime; timings below exclude these frames.
    warm_pipeline = FacePipeline(cfg)
    warm_motion = MotionAnalyzer(scale=cfg.flow_scale)
    for idx, frame in enumerate(frames[:8]):
        warm_motion.update(frame, idx)
        warm_pipeline.analyze(frame, motion=warm_motion, target_side=cfg.base_resolution)

    agent_runs = {}
    for name, mode, run_cfg in (
        ("static_320_detector", "static", cfg),
        ("adaptive_no_forensics", "agent", cfg.model_copy(update={"enable_forensics": False})),
        ("full_agent", "agent", cfg),
    ):
        runs = [_run_config(name, frames, gt, run_cfg, mode) for _ in range(REPS)]
        best = min(runs, key=lambda r: r.get("cpu_latency_p50_ms", r["latency_p50_ms"]))
        best["repetitions"] = [
            {
                "wall_fps": r["wall_fps"],
                "latency_p50_ms": r["latency_p50_ms"],
                "latency_p95_ms": r["latency_p95_ms"],
                "cpu_fps": r["cpu_fps"],
                "cpu_latency_p50_ms": r["cpu_latency_p50_ms"],
                "cpu_latency_p95_ms": r["cpu_latency_p95_ms"],
                "f1": r["f1"],
                "event_recall": r["event_recall"],
            }
            for r in runs
        ]
        best["wall_p50_spread_ms"] = [
            min(r["latency_p50_ms"] for r in runs),
            max(r["latency_p50_ms"] for r in runs),
        ]
        best["cpu_p50_spread_ms"] = [
            min(r["cpu_latency_p50_ms"] for r in runs),
            max(r["cpu_latency_p50_ms"] for r in runs),
        ]
        best["wall_fps_spread"] = [
            min(r["wall_fps"] for r in runs),
            max(r["wall_fps"] for r in runs),
        ]
        agent_runs[name] = best
    dispatch = dispatch_benchmark(cfg)
    threads = thread_benchmark(frames, cfg)

    payload = {
        "schema": "vigil.benchmarks/1.0",
        "environment": environment(cores),
        "runtime": {
            "model_load_s": round(model_load_s, 3),
            "rss_start_mb": rss_start,
            "rss_after_models_mb": rss_after_models,
            "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "backend": f"{pipeline.detector_backend}+{pipeline.recognizer_backend}",
            "degraded": pipeline.degraded,
            "notes": pipeline.notes,
            "total_benchmark_s": round(time.perf_counter() - started, 2),
        },
        "benchmarks": {
            "detection": detection,
            "recognition": recognition,
            "agent": agent_runs,
            "dispatch": dispatch,
            "thread_scaling": threads,
        },
        "methodology": {
            "repetitions": REPS,
            "repetition_policy": "Each timing block runs VIGIL_BENCH_REPS times; the pass with the lowest CPU-time p50 is reported, together with the observed wall/CPU spread.",
            "detection": "100 labelled synthetic images (60 positives / 40 negatives), exact boxes, IoU>=0.5; ROC-AUC over GT-face scores vs background/unmatched scores.",
            "recognition": "30 genuine / 30 impostor pairs built from 13 clustered identities with rotation, scale, exposure and JPEG augmentation.",
            "agent": "288-frame demo clip (640x360, 24 fps) with scripted occlusion, fast entry, low light and blur; IoU>=0.35 matching, event windows +/- 2/6 frames.",
            "threads": "60 frames, OpenCV threads 1/2/4, pinned to 4 logical CPUs.",
            "dispatch": "cv2.setUseOptimized(False) vs True on 5 representative operators.",
        },
        "limitations": [
            "Synthetic eval corpus; not directly comparable to WIDER FACE / LFW leaderboards.",
            "Single shared laptop host (load average reported); absolute latencies vary with background load.",
            "CPU-only on x86; Graviton/COOL numbers must be produced on Arm with aws/run_cool_bench.sh.",
            "No certified presentation-attack detection; liveness is an explicitly labelled proxy.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    rows = []
    for name, result in agent_runs.items():
        for metric in ("cpu_fps", "wall_fps", "latency_p50_ms", "latency_p95_ms", "cpu_latency_p50_ms", "cpu_latency_p95_ms", "precision", "recall", "f1", "identity_accuracy", "event_recall", "trigger_precision", "false_trigger_rate", "rss_delta_mb"):
            rows.append([name, metric, result.get(metric), ""])
    for metric in ("roc_auc", "precision", "recall", "f1", "latency_ms_p50", "latency_ms_p95", "cpu_latency_ms_p50", "cpu_fps", "throughput_fps"):
        rows.append(["detection_yunet_640", metric, detection.get(metric), ""])
    for metric in ("roc_auc", "tar_at_far_1e-2", "accuracy_at_config_threshold", "embedding_latency_ms_p50", "embedding_cpu_ms_p50"):
        rows.append(["recognition_sface", metric, recognition.get(metric), ""])
    for op, values in dispatch["ops"].items():
        rows.append([f"dispatch_{op}", "speedup", values.get("speedup"), "optimized_vs_baseline"])
    for threads_n, values in threads.items():
        rows.append([f"threads_{threads_n}", "fps", values.get("fps"), ""])
        rows.append([f"threads_{threads_n}", "cpu_fps", values.get("cpu_fps"), ""])
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "metric", "value", "notes"])
        writer.writerows(rows)

    LOG_TSV.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not LOG_TSV.exists():
        LOG_TSV.write_text("track\tmetric\tvalue\tnotes\n")
    with LOG_TSV.open("a") as handle:
        handle.write(f"07-opencv-ai-competition-2026\tdetection_roc_auc\t{detection['roc_auc']}\tstamp={stamp} synthetic 100-image set\n")
        handle.write(f"07-opencv-ai-competition-2026\trecognition_roc_auc\t{recognition['roc_auc']}\tstamp={stamp} 60 pairs\n")
        handle.write(f"07-opencv-ai-competition-2026\tfull_agent_p50_ms\t{agent_runs['full_agent']['latency_p50_ms']}\tstamp={stamp} 640x360 demo, 4 pinned cores\n")
        handle.write(f"07-opencv-ai-competition-2026\tfull_agent_fps\t{agent_runs['full_agent']['wall_fps']}\tstamp={stamp}\n")

    print(json.dumps({"detection": {k: detection[k] for k in ("roc_auc", "precision", "recall", "f1")},
                      "recognition": {k: recognition[k] for k in ("roc_auc", "tar_at_far_1e-2", "accuracy_at_config_threshold")},
                      "agent": {name: {k: res.get(k) for k in ("wall_fps", "latency_p50_ms", "latency_p95_ms", "f1", "identity_accuracy", "event_recall")} for name, res in agent_runs.items()},
                      "dispatch_mean_speedup": dispatch["mean_speedup"]}, indent=2))


if __name__ == "__main__":
    main()
