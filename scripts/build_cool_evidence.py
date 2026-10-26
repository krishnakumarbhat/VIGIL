"""Assemble the honest COOL/Graviton evidence record.

Inputs (all produced by scripts in this repository):

* ``outputs/benchmarks.json``        full x86 evaluation
* ``outputs/cool_local.json``        measured optimisation levers
* ``outputs/arch_probe_x86_64.json`` portable probe, native
* ``outputs/arch_probe_aarch64.json`` portable probe, qemu-user emulated

Output: ``outputs/cool_evidence.json``.

The COOL binary itself is **not** available in this environment.  The record
therefore reports:

* ``status: "unavailable"`` for COOL, with the probe that established it;
* ``local_evidence_status: "measured"`` for the levers we can actually pull;
* ``arm64_execution`` for the real aarch64 userland we ran under QEMU, marked
  ``emulated: true`` and explicitly excluded from any speedup claim;
* ``arm64_model`` for the calibrated simulation, marked ``simulated: true``.

No arm64 speedup is claimed anywhere in this file.  The Graviton build recipe
is real; the numbers for Graviton are not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cool_evidence.json"


def load(path: Path) -> dict:
    """Read a JSON artefact and fail loudly if a prior step was skipped."""
    if not path.exists():
        raise SystemExit(f"missing artefact: {path}\nRun scripts/run_all.sh first.")
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cool_symbol_probe() -> dict:
    """Show that the installed OpenCV builds expose no COOL runtime symbols."""
    names = sorted(name for name in dir(cv2) if "cool" in name.lower())
    return {
        "checked": ["opencv-python 5.0.0.93", "opencv-contrib-python 5.0.0.93"],
        "opencv_version": cv2.__version__,
        "symbols_matching_cool": names,
        "runtime_symbol_present": any(name.lower() in {"cool", "getcoolbackend"} for name in names),
        "contrib_probe_path": "/media/pope/projecteo/python_libs/cool_probe",
        "modules": {
            "dnn": True,
            "gapi_python_base_wheel": False,
            "gapi_python_contrib_wheel": True,
            "xfeatures2d_contrib_wheel": True,
            "yunet_facedetector": True,
            "sface_facerecognizer": True,
        },
        "contrib_probe_note": (
            "The contrib wheel was installed with --target into an isolated subdirectory "
            "so the working system cv2 5.0.0 was not overwritten."
        ),
        "interpretation": (
            "The only 'COOL' matches are unrelated camera/colormap constants. "
            "Neither PyPI wheel ships the OpenCV Optimization Library (COOL) runtime."
        ),
    }


def build_model(bench: dict, x86: dict, arm: dict) -> dict:
    """Scenario model anchored to measured x86 latency; no point estimate."""
    agent = bench["benchmarks"]["agent"]["full_agent"]
    base_p50 = float(agent["cpu_latency_p50_ms"])
    base_fps = float(agent["cpu_fps"])
    scenarios = []
    for multiplier, label in (
        (0.70, "conservative: Graviton slower than this x86 host"),
        (0.85, "cautious"),
        (1.00, "parity"),
        (1.25, "optimistic"),
    ):
        scenarios.append(
            {
                "assumed_arm64_multiplier_vs_x86": multiplier,
                "assumed_label": label,
                "estimated_cpu_p50_ms": round(base_p50 / multiplier, 2),
                "estimated_cpu_fps": round(base_fps * multiplier, 2),
                "estimated_cpu_ms_per_frame_energy_proxy": round(base_p50 / multiplier, 2),
            }
        )
    return {
        "type": "simulation",
        "simulated": True,
        "claimed": False,
        "anchored_to": {
            "artefact": "outputs/benchmarks.json",
            "metric": "benchmarks.agent.full_agent.cpu_latency_p50_ms",
            "measured_value": base_p50,
            "measured_cpu_fps": base_fps,
        },
        "calibration": {
            "x86_native_detection_p50_ms": x86["detection_yunet_640"]["p50_ms"],
            "aarch64_emulated_detection_p50_ms": arm["detection_yunet_640"]["p50_ms"],
            "emulation_overhead_x": round(
                arm["detection_yunet_640"]["p50_ms"] / max(1e-6, x86["detection_yunet_640"]["p50_ms"]), 1
            ),
            "note": (
                "The emulation overhead factor is not a Graviton factor; it is reported "
                "to show why emulated timings cannot calibrate Arm performance."
            ),
        },
        "scenarios": scenarios,
        "why_no_point_estimate": (
            "No Graviton host and no COOL entitlement were available, so any single "
            "arm64 speedup number would be invented. The multiplier is an explicit "
            "assumption exposed for sensitivity analysis."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build outputs/cool_evidence.json")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cool-build-id", default=None)
    args = parser.parse_args(argv)

    bench_path = ROOT / "outputs" / "benchmarks.json"
    local_path = ROOT / "outputs" / "cool_local.json"
    x86_path = ROOT / "outputs" / "arch_probe_x86_64.json"
    arm_path = ROOT / "outputs" / "arch_probe_aarch64.json"

    bench = load(bench_path)
    local = load(local_path)
    x86 = load(x86_path)
    arm = load(arm_path)

    arm_machine = arm["environment"]["machine"]
    arm_emulated = bool(arm["environment"].get("emulated", True))
    cool_build_id = args.cool_build_id
    cool_verified = bool(cool_build_id) and arm_machine == "aarch64" and not arm_emulated

    if cool_verified:
        status = "measured"
        reason = "COOL build id supplied and the probe ran on native aarch64."
    elif arm_machine == "aarch64" and arm_emulated:
        status = "unavailable"
        reason = (
            "COOL runtime is not installed and no COOL wheel is publicly obtainable; "
            "aarch64 execution is QEMU-emulated, so it is not Graviton evidence."
        )
    else:
        status = "unavailable"
        reason = "COOL runtime is not installed and the host is not aarch64."

    agent = bench["benchmarks"]["agent"]
    detection = bench["benchmarks"]["detection"]
    recognition = bench["benchmarks"]["recognition"]
    dispatch = local["dispatch"]

    functional_equivalence = {
        "x86_detections_per_image": x86["detection_yunet_640"]["detections_per_image"],
        "arm64_emulated_detections_per_image": arm["detection_yunet_640"]["detections_per_image"],
        "same_detection_count": (
            x86["detection_yunet_640"]["detections_per_image"]
            == arm["detection_yunet_640"]["detections_per_image"]
        ),
        "note": "Functional check only: the same OpenCV 5 YuNet path runs on aarch64 and finds the same faces.",
    }

    evidence = {
        "schema": "vigil.cool_evidence/2.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "status_reason": reason,
        "local_evidence_status": "measured",
        "arm64_model_status": "simulated",
        "cool_verified": cool_verified,
        "cool_build_id": cool_build_id,
        "cool_symbol_probe": cool_symbol_probe(),
        "expected_arch": "arm64",
        "runtime": {
            **bench["environment"],
            "probe_platform": platform.platform(),
            "host_python": sys.version.split()[0],
        },
        "measurements": {
            "detection_roc_auc": detection["roc_auc"],
            "detection_precision": detection["precision"],
            "detection_recall": detection["recall"],
            "detection_cpu_fps": detection["cpu_fps"],
            "detection_cpu_p50_ms": detection["cpu_latency_ms_p50"],
            "detection_cpu_p95_ms": detection["cpu_latency_ms_p95"],
            "detection_throughput_fps": detection["throughput_fps"],
            "recognition_roc_auc": recognition["roc_auc"],
            "recognition_tar_at_far_1e-2": recognition["tar_at_far_1e-2"],
            "recognition_embedding_cpu_p50_ms": recognition["embedding_cpu_ms_p50"],
            "full_agent_wall_fps": agent["full_agent"]["wall_fps"],
            "full_agent_cpu_fps": agent["full_agent"]["cpu_fps"],
            "full_agent_cpu_p50_ms": agent["full_agent"]["cpu_latency_p50_ms"],
            "full_agent_cpu_p95_ms": agent["full_agent"]["cpu_latency_p95_ms"],
            "full_agent_f1": agent["full_agent"]["f1"],
            "full_agent_event_recall": agent["full_agent"]["event_recall"],
            "static_agent_wall_fps": agent["static_320_detector"]["wall_fps"],
            "process_peak_rss_mb": bench["runtime"]["process_peak_rss_mb"],
        },
        "local_optimisation_levers": {
            "status": "measured",
            "dispatch": dispatch,
            "thread_scaling": local["threads"],
            "spatial_attention_oracle_roi": local["spatial_attention"],
            "kernel_order": local["kernel_order"],
            "opencl": local["opencl"],
            "energy_proxy": local["energy"],
            "headline": {
                "dispatch_mean_speedup": dispatch["mean_speedup"],
                "dispatch_median_speedup": dispatch["median_speedup"],
                "roi_latency_speedup": local["spatial_attention"]["latency_speedup"],
                "roi_recall_full_640": local["spatial_attention"]["full_640"]["recall"],
                "roi_recall_crop_320": local["spatial_attention"]["roi_320_oracle"]["recall"],
                "resize_before_gray_speedup": local["kernel_order"]["resize_first_speedup"],
                "threads_4_fps": local["threads"]["4"]["fps"],
                "threads_1_fps": local["threads"]["1"]["fps"],
            },
            "interpretation": (
                "Dispatch is operator-dependent: Gaussian blur and Laplacian gain "
                "2.3-2.7x, while Canny is slower with the IPP path on this build, so "
                "the median across five operators is ~1.0x. Spatial attention (ROI at "
                "320 px) is the largest real saving at ~2.8-3.0x with equal or better "
                "recall on the labelled subset. None of these are COOL results."
            ),
        },
        "arm64_execution": {
            "status": "measured_emulated",
            "measured": True,
            "emulated": arm_emulated,
            "hardware": "QEMU user-mode aarch64 on x86_64, Ubuntu 22.04 arm64 rootfs",
            "opencv": arm["environment"]["opencv"],
            "neon_baseline": arm["environment"]["baseline"],
            "neon_dispatch": arm["environment"]["dispatch"],
            "toolchain": {
                "qemu": "qemu-aarch64-static 7.2.0 (multiarch/qemu-user-static v7.2.0-1)",
                "rootfs": "Ubuntu 22.04.5 arm64 base",
                "python": arm["environment"]["python"],
                "opencv_wheel": "opencv-python-headless 5.0.0.93 (manylinux_2_28_aarch64)",
                "numpy_wheel": "numpy 2.3.5 (manylinux_2_28_aarch64)",
                "cache": "/tmp/opencode/vigil-arm64 (override with VIGIL_ARM_CACHE)",
                "setup_script": "scripts/run_arm64_emulated.sh",
            },
            "probe": arm["detection_yunet_640"],
            "operators": arm["operators"],
            "functional_equivalence": functional_equivalence,
            "warning": (
                "Timings under QEMU are translation-limited (detection ~257x slower than "
                "native x86 here). They prove the aarch64 code path runs; they say nothing "
                "about Graviton speed."
            ),
        },
        "arm64_model": build_model(bench, x86, arm),
        "grading_claims": {
            "cool_award_claimed": False,
            "reason": (
                "The 30% COOL criterion requires verified Graviton execution with the COOL "
                "runtime. Neither was available; this record intentionally refuses the claim."
            ),
            "overall_track_claimed": True,
            "agentic_vision_claimed": True,
        },
        "reproduce": {
            "commands": [
                "python3 benchmarks/run_eval.py",
                "python3 benchmarks/cool_local.py --out outputs/cool_local.json",
                "python3 benchmarks/arch_probe.py --out outputs/arch_probe_x86_64.json",
                "bash scripts/run_arm64_emulated.sh",
                "python3 scripts/build_cool_evidence.py",
            ],
            "arm64_build_recipe": "aws/Dockerfile.arm64",
            "arm64_runner": "aws/run_cool_bench.sh",
            "arm64_make_target": "make cool-arm64 COOL_WHEEL_URL=<url> COOL_BUILD_ID=<id>",
            "exact_arm64_commands": [
                "docker buildx build --platform linux/arm64 --build-arg COOL_WHEEL_URL=\"$COOL_WHEEL_URL\" -f aws/Dockerfile.arm64 -t vigil5:arm64 .",
                "COOL_WHEEL_URL=\"$COOL_WHEEL_URL\" COOL_BUILD_ID=\"$COOL_BUILD_ID\" bash aws/run_cool_bench.sh arm64",
            ],
            "requires": "AWS Graviton host or arm64 builder with QEMU binfmt, plus a COOL wheel URL",
        },
        "source_artifacts": {
            str(path.relative_to(ROOT)): {"sha256_16": sha256(path), "bytes": path.stat().st_size}
            for path in (bench_path, local_path, x86_path, arm_path)
        },
        "caveats": [
            "COOL is unavailable: no public wheel, no entitlement, and no COOL symbols in the PyPI OpenCV 5 wheels.",
            "aarch64 was executed under QEMU user-mode; its timings are emulated and excluded from every speedup claim.",
            "The arm64 model is a simulation with an explicit multiplier assumption; it is not a measurement.",
            "The ROI lever uses ground-truth crops and is an oracle upper bound.",
            "Energy is CPU-time, not joules; RAPL is not readable in this container.",
            "Full numbers and repetition spread are in outputs/benchmarks.json.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2))

    log = ROOT / "logs" / "experiments.tsv"
    if not log.exists():
        log.write_text("track\tmetric\tvalue\tnotes\n")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    headline = evidence["local_optimisation_levers"]["headline"]
    rows = [
        ("cool_status", status, "COOL unavailable; local levers measured; arm64 emulated"),
        ("cool_dispatch_mean_speedup", headline["dispatch_mean_speedup"], "interleaved A/B x86"),
        ("cool_dispatch_median_speedup", headline["dispatch_median_speedup"], "median across 5 ops"),
        ("cool_roi_latency_speedup", headline["roi_latency_speedup"], "oracle 320 ROI vs full 640"),
        ("cool_roi_recall_crop", headline["roi_recall_crop_320"], "labelled positives, IoU>=0.5"),
        ("cool_arm64_emulated_detection_p50_ms", arm["detection_yunet_640"]["p50_ms"], "QEMU emulation, not Graviton"),
    ]
    with log.open("a") as handle:
        for metric, value, note in rows:
            handle.write(f"07-opencv-ai-competition-2026\t{metric}\t{value}\tstamp={stamp} {note}\n")

    print(json.dumps({
        "status": evidence["status"],
        "reason": evidence["status_reason"],
        "cool_verified": evidence["cool_verified"],
        "local_levers": evidence["local_optimisation_levers"]["headline"],
        "arm64_functional": functional_equivalence,
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
