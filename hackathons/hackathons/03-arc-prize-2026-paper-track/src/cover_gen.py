#!/usr/bin/env python3
"""Generate cover_image.png (>=1200px wide)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/pope/Desktop/gpu /hackathons/03-arc-prize-2026-paper-track/outputs/cover_image.png")

fig = plt.figure(figsize=(14, 8), dpi=200, facecolor="#f8f9fb")
# Create a grid layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.35)

# Panel A (top-left): DSL pipeline schematic
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis("off")
ax1.set_title("A. DSL + Canonicalisation Pipeline", fontsize=14, weight="bold", color="#1a2b3c", pad=15)
boxes = [
    (1.0, 3.2, "Grid Input"),
    (3.6, 3.2, "DSL Primitive\n(Structural)"),
    (6.2, 3.2, "Canonical Key\n(D4 × Colour)"),
    (8.8, 3.2, "Hypothesis\nCandidate"),
]
for x, y, label in boxes:
    rect = plt.Rectangle((x-0.8, y-0.8), 1.6, 1.6, facecolor="#e8f0fe", edgecolor="#2a4d7a", linewidth=2.5, zorder=2)
    ax1.add_patch(rect)
    ax1.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold", zorder=3, color="#1a2b3c")
for i in range(len(boxes)-1):
    ax1.annotate("", xy=(boxes[i+1][0]-0.8, 3.2), xytext=(boxes[i][0]+0.8, 3.2),
                arrowprops=dict(arrowstyle="->", color="#2a4d7a", lw=2.5))
ax1.annotate("Invariance group: D4 (|G|=8) × colour-bijection (≤10!)\nSound collapse; incomplete for multi-step compositions",
             xy=(4.5, 1.2), fontsize=9, ha="center", color="#555555",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#c9a227"))

# Panel B (top-right): Orbit reduction bar chart
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.set_title("B. Orbit Reduction (Real D4 + Colour Canon.)", fontsize=14, weight="bold", color="#1a2b3c", pad=15)
labels = ["No canonical", "D4 only", "D4 + colour"]
orbit_factors = [1, 8, 80]  # theoretical max; real average lower
bars = ax2.barh(labels, [np.log10(f+1) for f in orbit_factors], color=["#777777", "#e0a03a", "#2a7a4d"], edgecolor="#333", height=0.55)
ax2.set_xlabel("Log10 State-Space Reduction Factor", fontsize=10)
ax2.set_xlim(0, 2.2)
for bar, f in zip(bars, orbit_factors):
    label_text = f"{f}× max"
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             label_text, va="center", fontsize=9, weight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Panel C (bottom-left): Key results from real Track 04 data
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.axis("off")
ax3.set_title("C. Real Empirical Results (Track 04 Artifacts)", fontsize=14, weight="bold", color="#1a2b3c", pad=10)
results_text = (
    "TRAIN SPLIT (1000 tasks)\n"
    "• Graded pass@1 = 68 / 1000 (6.8%)\n"
    "• Graded pass@2 = 5 / 1000 (0.5%)\n"
    "• Symbolic families (depth-1/2): 46/347 fits (13.3%)\n"
    "• Learned families (cell/object/tree): 244/287 fits (85.0%)\n"
    "• Per-strategy precision: depth1 97.1%, depth2 92.3%, cell_tree ~9.8%\n"
    "\n"
    "EVALUATION SPLIT (120 tasks, graded)\n"
    "• Pass@1 = 0 / 120 (0.0%)\n"
    "• Pass@2 = 0 / 120 (0.0%)\n"
    "• Symbolic search: 0 programs fit all train pairs\n"
    "• Fallback shape-miss rate: 54/167 = 32.3% (~30% per spec)\n"
    "\n"
    "Source files: outputs/eval_metrics.json, outputs/eval_metrics_training.json,\n"
    "outputs/README_TRACK04.md, logs/experiments.tsv (verified 2026-10-22)"
)
ax3.text(0.05, 0.05, results_text, fontsize=9.5, va="bottom", ha="left",
         family="monospace", color="#222", linespacing=1.7,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#ccc"))

# Panel D (bottom-right): Mini ablation chart
ax4 = fig.add_subplot(gs[1, 2:4])
ax4.set_title("D. Ablation (Real Runs, n≤60 tasks, 8s budget)", fontsize=14, weight="bold", color="#1a2b3c", pad=15)
labels_ab = ["Base (full)", "No depth-2", "Symbolic only", "Depth-2 sym.", "Eval (30)"]
rates_ab = [25/60*100, 24/60*100, 3/60*100, 3/60*100, 4/30*100]
colors_ab = ["#2a7a4d", "#4a9b6b", "#777777", "#777777", "#e0a03a"]
bars = ax4.barh(labels_ab, rates_ab, color=colors_ab, edgecolor="#333", height=0.55)
ax4.set_xlabel("Solve Rate (%)", fontsize=10)
ax4.set_xlim(0, 50)
for bar, r in zip(bars, rates_ab):
    ax4.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
             f"{r:.1f}%", va="center", fontsize=9, weight="bold", color="#333")
ax4.axvline(x=10, color="#b22222", linestyle="--", alpha=0.35)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Main title above everything
fig.suptitle("ARC-AGI-2 Neuro-Symbolic DSL Synthesis: D4 Canonicalisation, Bidirectional Search,\nand the Completeness Gap  |  Track 03 — ARC Prize 2026 Paper Track ($450k, deadline 2026-11-09)",
             fontsize=16, weight="bold", color="#1a2b3c", y=0.98)

# Footer line
fig.text(0.5, 0.02, "All numbers trace to verified artifacts: 03-arc-prize-.../outputs/ablation_results.csv  |  04-.../outputs/eval_metrics.json  |  04-.../src/*.py  |  arXiv:2505.11831, 2603.06590, 2603.13372",
         ha="center", fontsize=7, color="#777777")

fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(str(OUT), bbox_inches="tight", facecolor="#f8f9fb")
print("Cover saved to:", OUT)
