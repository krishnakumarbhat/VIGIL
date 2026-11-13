"""Reproduce the two paper figures from measured ARC-AGI-2 run artifacts.

Inputs are the measured per-task latencies from the validation kernel log and the
public leaderboard snapshot; nothing is synthetic.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/pope/Desktop/gpu /hackathons/03-arc-prize-2026-paper-track/outputs/figures")
LATENCIES = [219.4, 283.7, 500.8, 587.7, 600.7, 602.9, 678.6, 886.6, 924.1, 1144.0, 1262.6, 1302.0]
CAP = 1200.0
OBSERVED_SCORE = 30.56
TOP200 = 33.06


def timing_figure() -> None:
    """Plot the twelve measured per-task latencies against the internal cap."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=200)
    y = np.arange(len(LATENCIES))
    ax.barh(y, LATENCIES, color="#3b6ea5", height=0.68, edgecolor="white", linewidth=0.6)
    ax.axvline(CAP, color="#c0392b", linestyle="--", linewidth=1.2, label=f"decode cap {CAP:.0f}s")
    ax.set_yticks(y)
    ax.set_yticklabels([f"t{i+1}" for i in y], fontsize=7)
    ax.set_xlabel("wall-clock seconds per task (4x L4, one worker per task)")
    ax.set_xlim(0, 1450)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Measured task latency, 12-task validation run", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_task_latency.pdf")
    fig.savefig(OUT / "fig07_task_latency.png", dpi=200)
    plt.close(fig)


def coverage_figure() -> None:
    """Plot score against coverage for the measured per-attempt solve range."""
    coverage = np.linspace(0.40, 1.0, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
    for p, style in ((0.32, ":"), (0.40, "-"), (0.49, "--")):
        ax.plot(coverage * 100, coverage * p * 100, style, color="#3b6ea5", linewidth=1.3,
                label=f"p = {p:.2f}")
    ax.axhline(TOP200, color="#c0392b", linewidth=1.1, label="top-200 threshold 33.06")
    ax.plot([94.7], [OBSERVED_SCORE], marker="o", markersize=6, color="#c0392b",
            label="observed 30.56 at projected coverage")
    ax.fill_betweenx([0, 60], 62, 95, color="#3b6ea5", alpha=0.06, linewidth=0)
    ax.set_xlabel("coverage C: fraction of the 240 hidden tasks reached (%)")
    ax.set_ylabel("leaderboard score S = C x p (%)")
    ax.set_xlim(40, 100)
    ax.set_ylim(10, 52)
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Coverage tax: score factorises into coverage and per-attempt quality", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig08_coverage_model.pdf")
    fig.savefig(OUT / "fig08_coverage_model.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    timing_figure()
    coverage_figure()
    print("wrote", sorted(p.name for p in OUT.glob("fig0[78]*")))
    print("mean latency", round(sum(LATENCIES) / len(LATENCIES), 1))
    print("coverage at mean", round(4 * 710 * 60 / (240 * (sum(LATENCIES) / len(LATENCIES))), 3))
