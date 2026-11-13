"""Render the Agentic-Vision workflow diagram (perception -> decision -> action).

Produces ``outputs/agent_workflow.png`` and ``.svg``.  Pure matplotlib, no
graphviz dependency, so it runs anywhere the project already runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
STAGES = [
    (
        "SEE",
        "OpenCV 5 · YuNet @ 640",
        "motion ROI (Farneback)\nface quality + liveness\nSFace embedding cache",
    ),
    (
        "REASON",
        "reason codes, not scores",
        "low_resolution · low_quality\nlow_confidence · spoof_suspect\nmotion_anomaly · unknown_id\nCPU budget check",
    ),
    (
        "ACT",
        "one reversible action",
        "increase_resolution 320→768\nzoom_roi · forensic_inspection\n(upscale + flip-TTA)\nescalate_human",
    ),
    (
        "VERIFY",
        "did it actually improve?",
        "Δ confidence · detection count\nIoU · forensic consistency\naccept  |  rollback",
    ),
    (
        "EMIT",
        "one validated record",
        "Pydantic telemetry + overlay\nidentity or unknown\nincident digest, no raw frame",
    ),
]
EDGE = ["#1f4e79", "#2e75b6", "#c55a11", "#548235", "#7030a0"]
FILL = ["#dbe9f6", "#e3eef9", "#fbe5d6", "#e2efda", "#e9def2"]


def box(ax, x, y, w, h, title, subtitle, body, edge, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=2.0, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.050, title, ha="center", va="top",
            fontsize=16, fontweight="bold", color=edge, zorder=3)
    ax.text(x + w / 2, y + h - 0.115, subtitle, ha="center", va="top",
            fontsize=9.0, style="italic", color="#333333", zorder=3)
    ax.text(x + w / 2, y + h - 0.190, body, ha="center", va="top",
            fontsize=8.2, color="#222222", linespacing=1.40, zorder=3)


def head(ax, p0, p1, color="#404040", lw=2.0):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17,
                        linewidth=lw, color=color, zorder=1)
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 10.2), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.975, "VIGIL-5 — agent workflow", ha="center", va="top",
            fontsize=25, fontweight="bold", color="#111111")
    ax.text(0.5, 0.938,
            "OpenCV 5 perception output changes the next decision; every action is verified and reversible",
            ha="center", va="top", fontsize=11.2, color="#444444")

    w, h, y = 0.170, 0.375, 0.505
    xs = [0.020, 0.216, 0.412, 0.608, 0.804]
    for (title, sub, body), x, edge, fill in zip(STAGES, xs, EDGE, FILL):
        box(ax, x, y, w, h, title, sub, body, edge, fill)
    for i in range(4):
        head(ax, (xs[i] + w + 0.003, y + h / 2), (xs[i + 1] - 0.003, y + h / 2))

    # retry loop: VERIFY bottom -> down -> left -> up into REASON
    ymid = 0.235
    x_reason = xs[1] + w / 2
    x_verify = xs[3] + w / 2
    ax.plot([x_verify, x_verify], [y - 0.002, ymid], color="#c00000", lw=1.9, zorder=1)
    ax.plot([x_verify, x_reason], [ymid, ymid], color="#c00000", lw=1.9, zorder=1)
    head(ax, (x_reason, ymid), (x_reason, y - 0.002), color="#c00000", lw=1.9)
    ax.text((x_reason + x_verify) / 2, ymid + 0.012,
            "retry loop — max 2 adaptive steps, only while a reason code is still active",
            ha="center", va="bottom", fontsize=9.2, color="#c00000", style="italic")

    # escalation lane under ACT
    ex, ew = xs[2] - 0.030, w + 0.060
    ey, eh = 0.330, 0.105
    ax.add_patch(
        FancyBboxPatch((ex, ey), ew, eh, boxstyle="round,pad=0.010,rounding_size=0.018",
                       linewidth=1.9, edgecolor="#c00000", facecolor="#fdeaea", zorder=2)
    )
    ax.text(ex + ew / 2, ey + eh * 0.66, "ESCALATE → human review",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#c00000", zorder=3)
    ax.text(ex + ew / 2, ey + eh * 0.28, "incident digest · no biometric payload · cooldown 120 frames",
            ha="center", va="center", fontsize=7.8, color="#7f1d1d", zorder=3)
    head(ax, (xs[2] + w / 2, y - 0.002), (xs[2] + w / 2, ey + eh + 0.002), color="#c00000", lw=1.8)
    ax.text(xs[2] + w / 2 + 0.008, (ey + eh + y) / 2,
            "policy fires only when both hold:\n≥10 unknown frames AND a critical cue",
            ha="left", va="center", fontsize=8.0, color="#7f1d1d")

    # policy strip
    ax.add_patch(
        FancyBboxPatch((0.020, 0.030), 0.960, 0.120, boxstyle="round,pad=0.008,rounding_size=0.012",
                       linewidth=1.1, edgecolor="#7f7f7f", facecolor="#f5f5f5", zorder=2)
    )
    ax.text(
        0.5, 0.115,
        "Thresholds:  quality_floor 0.42   ·   det_confidence_floor 0.72   ·   min_face_px 56   ·   "
        "anomaly_z 2.5   ·   latency budget 100 ms (72 % soft stop)",
        ha="center", va="center", fontsize=8.8, color="#222222", zorder=3,
    )
    ax.text(
        0.5, 0.068,
        "Human control: escalation after ≥10 persistent unknown frames + critical cue (spoof, anomaly, low light), "
        "120-frame cooldown  ·  consent required for enrolment  ·  no embeddings in telemetry by default",
        ha="center", va="center", fontsize=8.8, color="#222222", zorder=3,
    )

    fig.savefig(OUT / "agent_workflow.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "agent_workflow.svg", bbox_inches="tight", facecolor="white")
    print("wrote outputs/agent_workflow.png and .svg")


if __name__ == "__main__":
    main()
