"""Render the cover image for the Kaggle dataset "Name That LLM: 18-Model Prompt Blocks".

Clean typographic design: title, one subtitle line, and the 18 competition model names as a grid of coloured tiles
(one hue per model family, no logos). Models absent from this version are drawn as dashed outline tiles marked "pending".
Writes cover_1200x675.png and dataset-cover-image.png (560 x 280, the crop Kaggle's metadata upload uses for the header).

usage: python make_cover.py [--out-dir DIR] [--labels a,b,...] [--subtitle TEXT]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

LABELS = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "glm-5.3", "glm-5.3-flash", "kimi-k3", "qwen3.8-max"]
FAMILY = {"claude": ["#B85A36", "#C96A42", "#A44E2E", "#D27A52", "#9A4529"],
          "gpt": ["#1D7266", "#258576", "#176055", "#2E9483", "#12524A"],
          "gemini": ["#2D63C4", "#3B74D8", "#2456AD", "#4C82E0"],
          "glm": ["#6447AE", "#7958C2"], "kimi": ["#3B3F48"], "qwen": ["#B9832A"]}
BG, INK, MUTED = "#F4F2ED", "#1C1D21", "#5E6168"


def family(label):
    for k in FAMILY:
        if label.startswith(k):
            return k
    return "kimi"


def colour(label):
    fam = family(label)
    members = [l for l in LABELS if family(l) == fam]
    return FAMILY[fam][members.index(label) % len(FAMILY[fam])]


def render(width, height, path, present, subtitle, scale=2):
    W, H = width * scale, height * scale
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    pad = 0.055 * W
    small = width < 800
    title_fs = (19 if small else 40) * scale * 0.72
    ax.text(pad, 0.11 * H, "Name That LLM", fontsize=title_fs, fontweight="bold", color=INK, va="top", family="DejaVu Sans")
    ax.text(pad, 0.11 * H + title_fs * 1.45, "18-Model Prompt Blocks", fontsize=title_fs * 0.78, color=INK, va="top", family="DejaVu Sans")
    if subtitle:
        ax.text(pad, 0.11 * H + title_fs * 2.75, subtitle, fontsize=title_fs * 0.36, color=MUTED, va="top", family="DejaVu Sans")
    cols, rows = 6, 3
    top = (0.44 if small else 0.36) * H
    bottom = H - (0.10 if small else 0.12) * H
    gap = 0.012 * W
    tw = (W - 2 * pad - (cols - 1) * gap) / cols
    th = (bottom - top - (rows - 1) * gap) / rows
    fs = (5.0 if small else 11.5) * scale * 0.78
    for i, lab in enumerate(LABELS):
        r, c = divmod(i, cols)
        x, y = pad + c * (tw + gap), top + r * (th + gap)
        have = lab in present
        box = FancyBboxPatch((x, y), tw, th, boxstyle=f"round,pad=0,rounding_size={0.12 * th}", linewidth=(0 if have else 1.6 * scale),
                             edgecolor=(None if have else "#A7AAB0"), facecolor=(colour(lab) if have else BG), linestyle=("-" if have else (0, (4, 3))))
        ax.add_patch(box)
        name = lab.replace("-preview", "-prev.") if len(lab) > 18 else lab
        ax.text(x + tw / 2, y + th * (0.5 if have else 0.42), name, ha="center", va="center", fontsize=fs, color=("white" if have else "#8C8F96"),
                family="DejaVu Sans Mono", fontweight="bold")
        if not have:
            ax.text(x + tw / 2, y + th * 0.72, "pending", ha="center", va="center", fontsize=fs * 0.75, color="#9A9DA3", family="DejaVu Sans")
    ax.text(W - pad, H - 0.045 * H, "AI-generated text  ·  one prompt, every model  ·  prvsiyan, 2026", ha="right", va="center",
            fontsize=title_fs * 0.27, color=MUTED, family="DejaVu Sans")
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=100, facecolor=BG)
    plt.close(fig)
    im = Image.open(tmp).convert("RGB").resize((width, height), Image.LANCZOS)
    im.save(path, optimize=True)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--labels", default=",".join(LABELS), help="models present in this version (others drawn as pending)")
    ap.add_argument("--subtitle", default="Same prompts answered by every model, with API-logged provenance")
    a = ap.parse_args()
    present = set(x for x in a.labels.split(",") if x)
    os.makedirs(a.out_dir, exist_ok=True)
    render(1200, 675, os.path.join(a.out_dir, "cover_1200x675.png"), present, a.subtitle)
    render(560, 280, os.path.join(a.out_dir, "dataset-cover-image.png"), present, a.subtitle, scale=3)
    print("wrote", os.path.join(a.out_dir, "cover_1200x675.png"), "and dataset-cover-image.png")


if __name__ == "__main__":
    main()
