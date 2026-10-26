"""Render the VIGIL-5 architecture diagram (matplotlib, no external assets)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "architecture_diagram.png"

INK = "#17202b"
MUTED = "#5b6b7c"
BG = "#f6f8fb"
INPUT = "#dbeafe"
PERCEPTION = "#dcfce7"
AGENT = "#fef3c7"
VERIFY = "#ede9fe"
CLOUD = "#ffe4e6"
TELEMETRY = "#e0f2fe"


def box(ax, x, y, w, h, title, body="", color="#ffffff", edge=INK, title_size=9.2, body_size=7.4):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.9",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 1.05, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK, zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 2.55, body, ha="center", va="top", fontsize=body_size,
                color=MUTED, linespacing=1.35, zorder=3)


def arrow(ax, start, end, color=INK, style="-|>", dashed=False, width=1.2, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=width,
            linestyle="--" if dashed else "-",
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=1,
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 9.4), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 152)
    ax.set_ylim(0, 94)
    ax.axis("off")

    ax.text(3, 90.5, "VIGIL-5", fontsize=26, fontweight="bold", color=INK, va="center")
    ax.text(22.5, 90.5, "autonomous vision agent", fontsize=14, color=MUTED, va="center")
    ax.text(3, 86.6, "See → Reason → Act → Verify, with a hard CPU budget and privacy-preserving telemetry",
            fontsize=10.5, color=MUTED, va="center")

    # ---------------------------------------------------------------- inputs
    ax.text(3, 80.5, "INGEST", fontsize=9, fontweight="bold", color=MUTED)
    box(ax, 3, 66.5, 19, 11.5, "Camera / RTSP", "live stream, backpressure\nand dropped-frame counter", INPUT)
    box(ax, 3, 53.5, 19, 11.5, "Video / image corpus", "MP4, JPEG, PNG\nlocal or mounted volume", INPUT)
    box(ax, 3, 40.5, 19, 11.5, "S3 / Lambda event", "bucket + key or base64 frame\noptional cloud path", INPUT)
    for y in (72.25, 59.25, 46.25):
        arrow(ax, (22.3, y), (26.2, y))

    # ------------------------------------------------------------ perception
    ax.text(27, 80.5, "PERCEPTION  ·  OpenCV 5.0.0", fontsize=9, fontweight="bold", color=MUTED)
    box(ax, 27, 66.5, 31, 11.5, "Face detection — YuNet",
        "cv2.FaceDetectorYN  ·  score + 5 landmarks\nfallback: cv2.dnn decode → skin contour", PERCEPTION)
    box(ax, 27, 53.5, 31, 11.5, "Face recognition — SFace",
        "128-D L2 embedding  ·  cosine gallery + consent\nfallback: cv2.dnn + alignment → LBP", PERCEPTION)
    box(ax, 27, 40.5, 31, 11.5, "Motion — Farneback + phase corr.",
        "dense flow, global drift, MOG2 ROI\nrobust median/MAD anomaly z-score", PERCEPTION)
    box(ax, 27, 28.5, 31, 10.8, "Quality & liveness proxies",
        "blur, exposure, pose, occlusion\nmicro-motion + moiré screen heuristic", PERCEPTION)
    for y in (72.25, 59.25):
        arrow(ax, (58.3, y), (62.2, y))
    arrow(ax, (58.3, 46.25), (62.2, 52.0), rad=-0.08)
    arrow(ax, (58.3, 33.9), (62.2, 48.0), rad=-0.06)

    # ----------------------------------------------------------- agent loop
    ax.text(63, 80.5, "AGENT DECISION LOOP  ·  budget = 100 ms CPU / frame", fontsize=9, fontweight="bold", color=MUTED)
    loop_y = 62.0
    stages = [
        (63.5, "1  SEE", "perceive at base\nresolution 640 px", AGENT),
        (82.0, "2  REASON", "quality, motion z,\nconfidence, identity", AGENT),
        (100.5, "3  ACT", "ladder / ROI / forensic\nescalation, inside budget", AGENT),
        (119.0, "4  VERIFY", "IoU, score delta,\nembedding consistency", VERIFY),
    ]
    for x, title, body, color in stages:
        box(ax, x, loop_y, 16.5, 12.0, title, body, color)
    for i in range(3):
        arrow(ax, (80.3 + i * 18.5, loop_y + 6), (81.8 + i * 18.5, loop_y + 6))
    arrow(ax, (135.8, loop_y + 6), (137.6, loop_y + 6), style="-|>")
    ax.text(138.3, loop_y + 6, "commit", fontsize=8, color=MUTED, va="center")
    # feedback loop
    arrow(ax, (127.2, loop_y - 0.3), (71.7, loop_y - 0.3), color="#7c3aed", dashed=True, rad=-0.16)
    ax.text(99.4, loop_y - 7.3, "next frame  ·  rollback if verification rejects an action",
            ha="center", fontsize=8, color="#6d28d9", style="italic")

    # action detail row
    ax.text(63, 51.5, "ACTIONS (bounded, reversible)", fontsize=8.5, fontweight="bold", color=MUTED)
    details = [
        (63.5, "resolution ladder", "320 → 480 → 640 → 768\nnever loses detections"),
        (84.0, "ROI zoom", "motion proposal or\nlargest face + margin"),
        (104.5, "forensic inspection", "multi-scale + flip-TTA\nrecovery at 0.25 threshold"),
        (125.0, "escalate / rollback", "human review + cloud hook\nor restore previous pass"),
    ]
    for x, title, body in details:
        box(ax, x, 39.0, 19.3, 11.0, title, body, "#fff7ed", title_size=8.6, body_size=7.0)
        arrow(ax, (x + 9.65, 50.3), (x + 9.65, 50.0), width=0.9)

    # ------------------------------------------------------------- outputs
    ax.text(3, 26.6, "OUTPUTS", fontsize=9, fontweight="bold", color=MUTED)
    box(ax, 3, 14.2, 33, 11.5, "Structured telemetry",
        "one validated JSON object per frame\nschema vigil.telemetry/1.0  ·  JSONL", TELEMETRY)
    box(ax, 38.5, 14.2, 33, 11.5, "Annotated evidence",
        "boxes, identities, motion ROI,\nforensic flag, state banner", TELEMETRY)
    box(ax, 74, 14.2, 33, 11.5, "Incident records",
        "consent-gated identity match,\nreason codes, incident id, summary", TELEMETRY)
    box(ax, 109.5, 14.2, 39.5, 11.5, "Deployment targets",
        "local engine  ·  arm64 Lambda  ·  ECS Fargate\nDocker + Terraform + CloudFormation", CLOUD)

    # ---------------------------------------------------- cloud / deployment
    ax.text(3, 11.8, "AWS PATH (optional, offline-first)", fontsize=9, fontweight="bold", color=MUTED)
    aws_boxes = [
        (3, "S3", "frames in, JSON out\nSSE-AES256"),
        (28, "Lambda / ECS", "arm64 Graviton\ntarget, COOL wheel"),
        (53, "Rekognition", "DetectFaces on\nin-memory JPEG"),
        (78, "CloudWatch", "latency, faces,\nforensics, escalations"),
        (103, "Bedrock", "Nova Lite operator\nsummary, no PII"),
        (128, "Offline mock", "same contract,\nzero credentials"),
    ]
    for x, title, body in aws_boxes:
        box(ax, x, 2.0, 21.0, 8.6, title, body, CLOUD, title_size=8.2, body_size=6.8)
    for x in (24.3, 49.3, 74.3, 99.3, 124.3):
        arrow(ax, (x, 6.3), (x + 1.0, 6.3), color="#9f1239", width=0.9)

    ax.text(3, 0.4, "Models: YuNet 2023mar (232 KB) + SFace 2021dec (38 MB) · 13 synthetic consenting identities · no raw embeddings persisted",
            fontsize=7.2, color=MUTED)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
