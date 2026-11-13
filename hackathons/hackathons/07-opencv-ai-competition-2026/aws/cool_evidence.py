"""Turn a benchmark run into a provenance record for COOL / Graviton claims.

The competition's COOL category asks for evidence that the Arm-optimised
OpenCV build was actually used.  This tool refuses to infer that: a run is only
marked ``cool_verified`` when it executed on aarch64 *and* the caller supplies
the build identifier printed by the COOL container.  Everything else is
reported as ``unavailable`` with the observed numbers and an explicit reason.

Usage
-----
    python3 aws/cool_evidence.py outputs/benchmarks.json \
        --output outputs/cool_evidence.json [--cool-build-id ID] [--require-arm]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARM_MACHINES = {"aarch64", "arm64"}


def _metric(payload: dict[str, Any], section: str, name: str) -> Any:
    return payload.get("benchmarks", {}).get(section, {}).get(name)


def build_evidence(
    payload: dict[str, Any],
    cool_build_id: str | None = None,
    expected_arch: str = "arm64",
) -> dict[str, Any]:
    """Build a conservative, machine-readable provenance record."""
    environment = payload.get("environment", {})
    machine = str(environment.get("machine", "unknown")).lower()
    verified_on_arm = machine in ARM_MACHINES
    cool_verified = bool(verified_on_arm and cool_build_id)
    agent = payload.get("benchmarks", {}).get("agent", {})
    full = agent.get("full_agent", {})
    static = agent.get("static_320_detector", {})
    caveats = []
    if not verified_on_arm:
        caveats.append(
            f"Executed on {machine or 'unknown'}, not {expected_arch}; Arm dispatch and Graviton "
            "latency remain unverified."
        )
    if not cool_build_id:
        caveats.append("No COOL build identifier supplied; COOL optimisation is not claimed.")
    if environment.get("load_average_1m", 0) and environment["load_average_1m"] > 8:
        caveats.append(
            f"Shared host load average {environment['load_average_1m']}; wall latencies are contended, "
            "CPU-time metrics are the stable comparison."
        )
    return {
        "schema": "vigil.cool_evidence/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_on_arm": verified_on_arm,
        "cool_verified": cool_verified,
        "cool_build_id": cool_build_id,
        "expected_arch": expected_arch,
        "runtime": {
            "machine": machine,
            "platform": environment.get("platform"),
            "python": environment.get("python"),
            "opencv": environment.get("opencv"),
            "opencv_module": environment.get("opencv_module"),
            "build_info_sha256": environment.get("build_info_sha256"),
            "simd": environment.get("simd", []),
            "ipp": environment.get("ipp", []),
            "pinned_cores": environment.get("pinned_cores", []),
            "load_average_1m": environment.get("load_average_1m"),
            "benchmark_timestamp_utc": environment.get("timestamp_utc"),
        },
        "measurements": {
            "detection_roc_auc": _metric(payload, "detection", "roc_auc"),
            "detection_cpu_fps": _metric(payload, "detection", "cpu_fps"),
            "detection_cpu_p50_ms": _metric(payload, "detection", "cpu_latency_ms_p50"),
            "recognition_roc_auc": _metric(payload, "recognition", "roc_auc"),
            "full_agent_wall_fps": full.get("wall_fps"),
            "full_agent_cpu_fps": full.get("cpu_fps"),
            "full_agent_cpu_p50_ms": full.get("cpu_latency_p50_ms"),
            "full_agent_cpu_p95_ms": full.get("cpu_latency_p95_ms"),
            "full_agent_f1": full.get("f1"),
            "full_agent_event_recall": full.get("event_recall"),
            "static_agent_wall_fps": static.get("wall_fps"),
            "dispatch_mean_speedup": _metric(payload, "dispatch", "mean_speedup"),
            "thread_scaling": payload.get("benchmarks", {}).get("thread_scaling", {}),
        },
        "caveats": caveats,
        "status": "measured" if cool_verified else "unavailable",
        "status_reason": (
            "COOL build id supplied on native aarch64."
            if cool_verified
            else (
                "No COOL build id supplied."
                if verified_on_arm
                else f"Host is {machine or 'unknown'}, not {expected_arch}; no COOL runtime."
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build COOL/Graviton evidence from benchmarks.json")
    parser.add_argument("benchmarks", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/cool_evidence.json"))
    parser.add_argument("--cool-build-id", default=None, help="Build id printed by the COOL container")
    parser.add_argument("--expected-arch", default="arm64")
    parser.add_argument("--require-arm", action="store_true", help="Exit 3 unless the run is on Arm")
    args = parser.parse_args(argv)

    payload = json.loads(args.benchmarks.read_text())
    evidence = build_evidence(payload, args.cool_build_id, args.expected_arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k: evidence[k] for k in ("status", "verified_on_arm", "cool_verified", "measurements")}, indent=2))
    if args.require_arm and not evidence["verified_on_arm"]:
        print("FAIL: benchmark did not run on Arm", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
