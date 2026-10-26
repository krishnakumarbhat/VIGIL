"""Build the <=5 minute submission video for Devpost.

Structure (about 100 s):
    title -> problem -> the loop -> architecture -> live demo -> agent trace
    -> measured results -> limitations -> close

Slides are rendered with matplotlib at 1920x1080, then concatenated with ffmpeg
together with the annotated 288-frame demo. No audio is added, so the result
plays anywhere. Output: ``outputs/demo_submission.mp4``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BG = "#0b1f33"
FG = "#f2f6fa"
ACCENT = "#4fc3f7"
WARM = "#ffb74d"
RED = "#ef5350"
GREEN = "#81c784"
W, H = 19.2, 10.8


def slide(path: Path, draw) -> None:
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw(ax, fig)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def title(ax, text, sub=None):
    ax.text(0.06, 0.86, text, ha="left", va="top", fontsize=46, fontweight="bold", color=FG)
    if sub:
        ax.text(0.06, 0.74, sub, ha="left", va="top", fontsize=24, color=ACCENT)


def bullets(ax, items, y=0.60, size=27, dy=0.105, color=FG, x=0.07):
    for i, item in enumerate(items):
        ax.text(x, y - i * dy, item, ha="left", va="top", fontsize=size, color=color)


def loop_slide(ax, fig):
    title(ax, "The loop", "compute is spent only when the current view is not good enough")
    stages = [
        ("SEE", "YuNet @ 640 + motion ROI", "#1f4e79"),
        ("REASON", "reason codes + CPU budget", "#2e75b6"),
        ("ACT", "ladder · zoom · forensic · escalate", "#c55a11"),
        ("VERIFY", "accept or roll back", "#548235"),
        ("EMIT", "one validated JSON record", "#7030a0"),
    ]
    x = 0.05
    for name, desc, color in stages:
        ax.add_patch(FancyBboxPatch((x, 0.46), 0.165, 0.15, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    linewidth=2, edgecolor=color, facecolor="#12293f"))
        ax.text(x + 0.0825, 0.555, name, ha="center", va="center", fontsize=24,
                fontweight="bold", color=FG)
        ax.text(x + 0.0825, 0.485, desc, ha="center", va="center", fontsize=11, color="#cfd8dc")
        x += 0.19
    ax.text(0.5, 0.36, "every action is reversible; verification can undo it",
            ha="center", va="center", fontsize=22, color=WARM, style="italic")
    ax.text(0.5, 0.22, "20.2 FPS full agent  ·  36 ms CPU p50 fused  ·  1.00 event recall",
            ha="center", va="center", fontsize=20, color=GREEN)


def architecture(ax, fig):
    title(ax, "Architecture", "OpenCV 5 perception loop, typed telemetry, AWS delivery (offline-validated)")
    img = plt.imread(OUT / "architecture_diagram.png")
    h, w = img.shape[:2]
    box_w, box_h = 0.72, 0.54
    ax_img = fig.add_axes([0.14, 0.14, box_w, box_h])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_facecolor(BG)
    ax.text(0.5, 0.075, "Lambda handler · arm64/x86 containers · ECS task definitions · Terraform/CloudFormation",
            ha="center", va="center", fontsize=15, color="#b0bec5")


def trace_slide(ax, fig):
    title(ax, "One frame, one decision", "frame 166 — the perception result changes the next action")
    img = plt.imread("/tmp/opencode/t07frames/f166.png")
    ax_img = fig.add_axes([0.05, 0.16, 0.52, 0.52])
    ax_img.imshow(img)
    ax_img.axis("off")
    items = [
        "motion z = 40.27",
        "YuNet: 0 faces at 640 px",
        "→ forensic recovery scan",
        "→ ROI zoom at 640 px",
        "verification: 0 new faces",
        "→ rollback, no hallucination",
    ]
    bullets(ax, items, y=0.62, size=18, dy=0.075, color=FG, x=0.62)


def results(ax, fig):
    title(ax, "Measured results", "same 288-frame clip, four pinned x86 cores, OpenCV threads = 1")
    axes = [fig.add_axes([0.06, 0.20, 0.24, 0.40]),
            fig.add_axes([0.38, 0.20, 0.24, 0.40]),
            fig.add_axes([0.70, 0.20, 0.24, 0.40])]
    for a in axes:
        a.set_facecolor("#12293f")
        for s in a.spines.values():
            s.set_color("#455a64")
        a.tick_params(colors=FG, labelsize=12)

    a = axes[0]
    a.bar(["agent", "static"], [1.0, 0.0], color=[GREEN, "#78909c"])
    a.set_ylim(0, 1.1)
    a.set_title("event recall", color=FG, fontsize=16)
    a.text(0, 1.02, "1.00", ha="center", color=FG, fontsize=14)
    a.text(1, 0.02, "0.00", ha="center", color=FG, fontsize=14)

    a = axes[1]
    a.bar(["agent", "static"], [24.81, 216.33], color=[ACCENT, "#78909c"])
    a.set_title("wall FPS", color=FG, fontsize=16)
    a.text(0, 30, "24.8", ha="center", color=FG, fontsize=14)
    a.text(1, 222, "216.3", ha="center", color=FG, fontsize=14)

    a = axes[2]
    a.bar(["det AUC", "TAR@FAR", "identity"], [0.9917, 1.000, 1.000],
          color=[ACCENT, GREEN, GREEN])
    a.set_ylim(0, 1.1)
    a.set_title("quality", color=FG, fontsize=16)
    for i, v in enumerate([0.9917, 1.000, 1.000]):
        a.text(i, v + 0.02, f"{v:.4f}" if i == 0 else f"{v:.3f}", ha="center", color=FG, fontsize=11)

    ax.text(0.5, 0.09, "The static baseline is 8.7x faster and sees nothing. Throughput is not the product metric.",
            ha="center", va="center", fontsize=16, color=WARM, style="italic")


def limitations(ax, fig):
    title(ax, "What is not demonstrated", "stated explicitly, because the report and the repository agree")
    bullets(ax, [
        "COOL / Graviton runtime: unavailable — 0 of 30 points claimed",
        "Live AWS deployment: templates validate, no account access",
        "Synthetic corpus: 100 detection images, 60 pairs, 288 frames",
        "Escalation did not fire in the demo; it is unit-traced",
        "Liveness is a texture/motion proxy, not certified PAD",
    ], y=0.66, size=21, dy=0.095, color="#e0e0e0")


def close(ax, fig):
    ax.text(0.5, 0.58, "VIGIL-5", ha="center", va="center", fontsize=68, fontweight="bold", color=FG)
    ax.text(0.5, 0.45, "OpenCV AI Competition 2026 · Track 07", ha="center", va="center",
            fontsize=24, color=ACCENT)
    ax.text(0.5, 0.32, "report · code · evidence · 61 passing tests", ha="center", va="center",
            fontsize=20, color="#b0bec5")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vigil5_video_", dir="/tmp/opencode"))
    slides = []

    def add(name, draw, duration):
        p = tmp / f"{name}.png"
        slide(p, draw)
        slides.append((p, duration))

    add("00_title", lambda ax, fig: (
        ax.text(0.5, 0.62, "VIGIL-5", ha="center", va="center", fontsize=78, fontweight="bold", color=FG),
        ax.text(0.5, 0.47, "an autonomous, budget-aware vision agent on OpenCV 5",
                ha="center", va="center", fontsize=26, color=ACCENT),
        ax.text(0.5, 0.36, "it spends compute only when the current view is not good enough",
                ha="center", va="center", fontsize=18, color="#b0bec5"),
        ax.text(0.5, 0.14, "OpenCV AI Competition 2026 · Track 07 · solo build", ha="center",
                va="center", fontsize=14, color="#78909c"),
    ), 5)
    add("01_problem", lambda ax, fig: (
        title(ax, "The failure is the confident wrong answer",
              "a fixed detector cannot say: the view is not good enough"),
        bullets(ax, [
            "216 FPS static 320 px detector — event recall 0.00",
            "It never asks for a better view; it never rolls back",
            "VIGIL-5 makes ambiguity a first-class state",
        ], y=0.62, size=25, dy=0.11),
    ), 9)
    add("02_loop", loop_slide, 10)
    add("03_arch", architecture, 10)
    add("05_trace", trace_slide, 11)
    add("06_results", results, 12)
    add("07_limits", limitations, 10)
    add("08_close", close, 6)

    demo = OUT / "demo_annotated.mp4"
    seg_dir = tmp / "segments"
    seg_dir.mkdir()
    segs = []
    for i, (png, dur) in enumerate(slides):
        seg = seg_dir / f"seg_{i:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", str(png),
            "-t", str(dur), "-r", "24", "-vf", "scale=1920:1080,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(seg),
        ], check=True)
        segs.append(seg)

    # demo segment, letterboxed to 1080p
    demo_seg = seg_dir / "demo.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(demo),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
               "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0b1f33,fps=24,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(demo_seg),
    ], check=True)

    # insert the demo after the architecture slide (index 3)
    order = segs[:4] + [demo_seg] + segs[4:]
    listing = tmp / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in order))
    out = OUT / "demo_submission.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ], check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
                            "-of", "default=nw=1", str(out)], capture_output=True, text=True)
    print(probe.stdout)
    print("wrote", out)


if __name__ == "__main__":
    main()
