"""Submission-artifact checks: the claims in the report must match files on disk."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def test_cool_evidence_is_honest() -> None:
    evidence = json.loads((OUT / "cool_evidence.json").read_text())
    assert evidence["status"] in {"measured", "simulated", "unavailable"}
    assert evidence["status"] == "unavailable"  # no COOL runtime on this host
    assert evidence["cool_verified"] is False
    assert evidence["local_evidence_status"] == "measured"
    assert evidence["arm64_execution"]["emulated"] is True
    assert evidence["arm64_model"]["simulated"] is True
    assert evidence["arm64_model"]["claimed"] is False
    assert evidence["grading_claims"]["cool_award_claimed"] is False
    assert evidence["reproduce"]["arm64_build_recipe"] == "aws/Dockerfile.arm64"
    assert "COOL" in evidence["status_reason"]


def test_demo_video_is_real_and_annotated() -> None:
    path = OUT / "demo_annotated.mp4"
    assert path.exists() and path.stat().st_size > 500_000
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 288
        assert int(cap.get(cv2.CAP_PROP_FPS)) == 24
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 360
    finally:
        cap.release()


def test_benchmarks_have_measured_metrics() -> None:
    bench = json.loads((OUT / "benchmarks.json").read_text())
    agent = bench["benchmarks"]["agent"]["full_agent"]
    assert agent["wall_fps"] > 0
    assert agent["cpu_latency_p50_ms"] > 0
    assert agent["cpu_latency_p95_ms"] >= agent["cpu_latency_p50_ms"]
    assert bench["benchmarks"]["detection"]["roc_auc"] > 0.9
    assert bench["methodology"]["repetitions"] >= 1
    assert bench["environment"]["opencv"].startswith("5.")


def test_report_maps_every_rubric_item_to_evidence() -> None:
    report = (OUT / "DEVPOST_SUBMISSION.md").read_text()
    assert report.count("**EVIDENCE") >= 8
    for section in ("Overall (100 points)", "Agentic Vision", "Best Use of COOL"):
        assert section in report
    for column in ("Criterion", "Max", "Claimed", "Evidence file", "Evidence metric"):
        assert column in report
    assert "not demonstrated" in report.lower()
    assert "COOL award is *not* claimed" in report


def test_experiment_log_contains_cool_rows() -> None:
    rows = (ROOT / "logs" / "experiments.tsv").read_text().splitlines()
    header = rows[0].split("\t")
    assert header[:4] == ["track", "metric", "value", "notes"]
    metrics = {line.split("\t")[1] for line in rows[1:] if len(line.split("\t")) >= 4}
    assert {"cool_status", "cool_dispatch_mean_speedup", "cool_roi_latency_speedup"} <= metrics


def test_make_and_run_all_entry_points_exist() -> None:
    makefile = (ROOT / "Makefile").read_text()
    runner = (ROOT / "scripts" / "run_all.sh").read_text()
    assert re.search(r"^all:", makefile, re.MULTILINE)
    assert "pytest" in runner and "run_eval.py" in runner and "generate_demo_video.py" in runner
