"""Reproduce metric, canonical-key, and scheduling audits without a GPU.

Only competition artifacts are read. Resampling is a conditional sensitivity
experiment, not a hidden-set forecast or an accuracy experiment.
"""
from __future__ import annotations

import hashlib
import heapq
import importlib.util
import itertools
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SIBLING = ROOT.parent / "04-arc-prize-2026-arc-agi-2"
OUT = ROOT / "outputs"
SEED = 20261106


def corrected_key(grid: np.ndarray) -> tuple[int, int, bytes]:
    """Canonicalise colours after each D4 action and preserve dimensions."""
    variants = []
    for turns in range(4):
        rotated = np.rot90(grid, turns)
        variants.extend((rotated, np.fliplr(rotated)))
    keys = []
    for variant in variants:
        palette: dict[int, int] = {}
        flat = [palette.setdefault(int(v), len(palette)) for v in variant.ravel()]
        keys.append((variant.shape[0], variant.shape[1], bytes(flat)))
    return min(keys)


def completed_tasks(latencies: np.ndarray, budget_s: float, workers: int = 4) -> int:
    """Simulate non-preemptive FIFO tasks; count completions before deadline."""
    available = [0.0] * workers
    completed = 0
    for latency in latencies:
        finish = heapq.heappop(available) + float(latency)
        heapq.heappush(available, finish)
        completed += int(finish <= budget_s)
    return completed


def run_audit() -> dict[str, Any]:
    """Verify invariance exhaustively and run seeded latency resampling."""
    spec = importlib.util.spec_from_file_location("audited_canonical", SIBLING / "src/canonical.py")
    assert spec is not None and spec.loader is not None
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    counterexample = None
    checks = 0
    for values in itertools.product(range(3), repeat=6):
        grid = np.array(values, dtype=np.int8).reshape(2, 3)
        expected = corrected_key(grid)
        for _, action in canonical.D4:
            transformed = action(grid)
            if counterexample is None and canonical.canonical_key(grid) != canonical.canonical_key(transformed):
                counterexample = {"input": grid.tolist(), "transform": transformed.tolist()}
            for permutation in itertools.permutations(range(3)):
                relabelled = np.array(permutation, dtype=np.int8)[transformed]
                assert corrected_key(relabelled) == expected
                checks += 1
    assert counterexample is not None
    a = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int8)
    b = a.reshape(2, 3)
    assert canonical.canonical_key(a) == canonical.canonical_key(b)
    assert corrected_key(a) != corrected_key(b)
    assert completed_tasks(np.full(240, 600), 42600) == 240
    assert completed_tasks(np.full(240, 1200), 42600) == 140
    log_path = SIBLING / "outputs/arc_nvarc_validation_v3/out/arc-nvarc-validated.log"
    timings: dict[str, float] = {}
    for task, seconds in re.findall(r"finished (\w+) in ([\d.]+)s", log_path.read_text()):
        value = float(seconds)
        if task in timings:
            assert timings[task] == value
        timings[task] = value
    assert len(timings) == 12
    durations = np.array(list(timings.values()))
    rng = np.random.default_rng(SEED)
    samples = rng.choice(durations, size=(5000, 240), replace=True)
    scenarios = []
    for scale in (1.0, 1.25, 1.5):
        counts = np.array([completed_tasks(sample * scale, 42600) for sample in samples])
        scenarios.append({"latency_multiplier": scale, "median_completed": float(np.median(counts)),
                          "p05_completed": float(np.quantile(counts, .05)),
                          "p95_completed": float(np.quantile(counts, .95))})
    metrics = {}
    for name in ("eval_metrics_training_v2.json", "eval_metrics_evaluation.json"):
        path = SIBLING / "outputs" / name
        data = json.loads(path.read_text())
        assert data["solved"] == sum(int(row["solved"]) for row in data["per_task"].values())
        metrics[name] = {k: data[k] for k in ("tasks", "solved", "graded_pass1", "graded_pass2")}
    validation_path = SIBLING / "outputs/arc_nvarc_validation_v3/out/validation_result.json"
    validation = json.loads(validation_path.read_text())
    assert validation["n_test_rows_selected"] == len(validation["per_test_rows"]) == 18
    assert sum(row["pass1"] for row in validation["per_test_rows"]) == 1
    assert sum(row["pass2"] for row in validation["per_test_rows"]) == 3
    task_successes = {}
    for row in validation["per_test_rows"]:
        task_successes.setdefault(row["task_id"], []).append(row["pass2"])
    assert len(task_successes) == 12
    all_rows_solved = sum(all(rows) for rows in task_successes.values())
    assert all_rows_solved == 2
    evidence = {"seed": SEED, "replicates": 5000, "invariance_checks": checks,
                "original_rotation_counterexample": counterexample,
                "shape_collision": {"input_a": a.tolist(), "input_b": b.tolist()},
                "latencies_s": dict(sorted(timings.items())), "mean_s": float(durations.mean()),
                "median_s": float(np.median(durations)), "nearest_rank_p90_s": float(np.sort(durations)[10]),
                "scenario_results": scenarios, "metrics": metrics,
                "validation": {"rows": 18, "pass1": 1, "pass2": 3,
                               "task_normalized_pass2": validation["task_normalized_pass2"],
                               "fully_solved_tasks_selected": all_rows_solved, "tasks_selected": 12},
                "checks": "34992 invariance assertions; shape and scheduling boundary assertions; metric recounts",
                "source_sha256": {str(p.relative_to(SIBLING)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (log_path, validation_path, SIBLING / "src/canonical.py",
                                            SIBLING / "src/search.py", SIBLING / "src/solver_v2.py")}}
    OUT.mkdir(exist_ok=True)
    (OUT / "AUDIT_FINAL.json").write_text(json.dumps(evidence, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    x = np.arange(12)
    axes[0].bar(x, np.sort(durations), color="#286c8e")
    axes[0].axhline(1200, color="#bd3d35", linestyle="--", label="soft cap: 1200 s")
    axes[0].set(xlabel="Tasks sorted by measured latency", ylabel="Seconds", title="Measured: twelve task latencies")
    axes[0].legend(fontsize=8)
    med = np.array([s["median_completed"] for s in scenarios]) / 240
    low = np.array([s["p05_completed"] for s in scenarios]) / 240
    high = np.array([s["p95_completed"] for s in scenarios]) / 240
    axes[1].errorbar([1, 1.25, 1.5], med, yerr=[med-low, high-med], fmt="o-", capsize=5, color="#286c8e")
    axes[1].set(xlabel="Assumed latency multiplier", ylabel="Completed fraction (out of 240)",
                title="Simulated: empirical resampling only", ylim=(.5, 1))
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.set_axisbelow(True)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"figures/fig09_audit_sensitivity.{extension}", dpi=200)
    plt.close(fig)
    return evidence


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({k: result[k] for k in ("invariance_checks", "original_rotation_counterexample", "mean_s", "scenario_results", "validation")}, indent=2))
