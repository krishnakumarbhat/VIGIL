#!/usr/bin/env python3
"""Regenerate MASTER_STATUS.md for the multi-track hackathon marathon.

Reads each track's logs/experiments.tsv + outputs/ manifests + live GPU state.
Safe to run every 20 minutes from cron or the autoresearch driver.
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone

ROOT = "/home/pope/Desktop/gpu /hackathons"
TRACKS = [
    ("01-nfl-big-data-bowl-2027", "NFL Big Data Bowl 2027", "$100k", "2027-01-06"),
    ("02-gemma-4-developer-agent-paper", "Gemma 4 Developer Agent Paper", "$35k", "2026-11-12"),
    ("03-arc-prize-2026-paper-track", "ARC Prize 2026 Paper Track", "$450k", "2026-11-09"),
    ("04-arc-prize-2026-arc-agi-2", "ARC Prize 2026 ARC-AGI-2", "$700k", "2026-11-02"),
    ("05-name-that-llm", "Name That LLM", "$50k", "2027-01-14"),
    ("06-playground-s6e10-airline-satisfaction", "Playground S6E10 Airline", "swag", "2026-10-31"),
    ("07-opencv-ai-competition-2026", "OpenCV AI Competition 2026", "$20,250", "2026-10-26"),
    ("08-openzee-openair-2026", "OpenZee OpenAir 2026", "?", "?"),
]


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def count_files(path):
    n = 0
    size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            n += 1
            try:
                size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n, size


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# MASTER STATUS - Multi-Track Kaggle Hackathon Marathon", "", f"_Last update: {now}_", "",
             "## GPU Lanes", "",
             "| Lane | State | Detail |", "|---|---|---|"]

    gpu = sh("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
    lines.append(f"| Local RTX 3050 | {'ACTIVE' if gpu else 'offline'} | `{gpu}` |")

    for slug in ("ntl-stage2-deberta", "ps6e10-stack"):
        st = sh(f"kaggle kernels status krishnakumarbhat/{slug} 2>&1 | tail -1", 40)
        state = "ERROR" if "ERROR" in st else ("RUNNING" if "RUNNING" in st else
                                               ("COMPLETE" if "COMPLETE" in st else st[-40:]))
        lines.append(f"| Kaggle GPU [{slug}] | {state} | `{st[-70:]}` |")

    coll = sh("pgrep -af 'run_track05_colab|colab_cli' | head -2", 20)
    lines.append(f"| Colab Pro T4 | {'RUNNING' if coll else 'idle'} | `{coll[:120]}` |")

    tok = ""
    try:
        tok = json.load(open("/home/pope/.kaggle/kaggle.json"))["key"]
        q = sh(f"curl -s -H 'Authorization: Basic {__import__('base64').b64encode(('krishnakumarbhat:'+tok).encode()).decode()}' "
               f"'https://www.kaggle.com/api/v1/kernels/quota'", 40)
        g = json.loads(q)["gpuQuota"]
        lines.append(f"| Kaggle quota | {g['timeUsed']['seconds']/3600:.2f}h used / "
                     f"{g['totalTimeAllowed']['seconds']/3600:.0f}h total | resets {json.loads(q)['quotaRefreshTime']} |")
    except Exception as e:
        lines.append(f"| Kaggle quota | n/a | {e!r} |")

    lines += ["", "## Tracks", "",
              "| # | Track | Prize | Deadline | Best validated score | Outputs | Size |", "|---|---|---|---|---|---|---|"]
    for i, (d, name, prize, dl) in enumerate(TRACKS, 1):
        tdir = os.path.join(ROOT, d)
        best = "-"
        exp = os.path.join(tdir, "logs", "experiments.tsv")
        if os.path.exists(exp):
            rows = [l.split("\t") for l in open(exp).read().splitlines() if l.strip()]
            rows = [r for r in rows if len(r) >= 3 and r[2] not in ("metric", "value", "")]
            if rows:
                best = f"{rows[-1][1]}={rows[-1][2][:40]}"
        n, size = count_files(os.path.join(tdir, "outputs")) if os.path.isdir(tdir) else (0, 0)
        lines.append(f"| {i:02d} | {name} | {prize} | {dl} | {best} | {n} files | {size/1024:.0f} KB |")

    lines += ["", "## Algorithm Update (2026-10-23)", "",
              "Track 05 rebuilt its stack while burning the remaining Colab Pro units. Every number is "
              "5-fold `StratifiedGroupKFold(shuffle, seed 42, groups=prompt_id)` over the 2,699 real rows, "
              "scored only on held-out rows.", "",
              "| Step | Cross-fitted balanced accuracy | Change |", "|---|---|---|",
              "| DeBERTa-v3-base alone (session start) | 0.4016 | — |",
              "| + Qwen2.5-0.5B LoRA, temperature + weight search + **per-class bias tuning** | 0.5557 | +15.4 pts, kills 4 collapsed classes (min recall 0.047 -> 0.35) |",
              "| + decorrelated TF-IDF/SGD member | 0.5957 | +4.0 pts from a member that scores only 0.36 alone |",
              "| + stylometry, char-only, word-only, cosine-kNN members | 0.5976 | +0.2 pts (diminishing) |",
              "| + `svmext` (TF-IDF trained with 40k external rows) | 0.6139 | +1.6 pts |",
              "| + Qwen2.5-0.5B LoRA **seed 1337** (mirror 6k, len 256) | **0.6817** | +6.8 pts; that member alone scores 0.6483 |",
              "", "Per-class recall is now 0.48-0.92 (was 0.047-0.88). Fold-level blend scores "
              "0.4148 / 0.4963 / 0.6815 / 0.8685 / 0.9481 — differences below ~0.02 are inside noise.",
              "", "| Approach | Verdict |", "|---|---|",
              "| Linear meta-learner over concatenated member log-probs (630 parameters) | 0.5709 — **worse** than the constrained weight+bias model; 2,159 rows cannot support it |",
              "| Qwen2.5-7B LoRA, A100 5-fold curriculum | Fold 0 = 0.4759, but the CLI session registry was pruned mid-run and orphaned the VM: artifacts unreachable. Stopped rather than burn units. |",
              "| DeBERTa-v3-large trained in pure bf16 | Did not converge (loss pinned at ln 18 for 2,400 steps). Needs fp32 master weights + autocast. |",
              "| `colab download` as the artifact channel | Replaced by in-band base64 over the exec stream + `src/ingest_b64.py`; the file API dies with the session registry. |",
              "", "Deployment: `outputs/ensemble_recipe.json` (temps/weights/biases) + `src/make_submission.py` "
              "reproduces the blend on any test frame from artifacts in `outputs/deploy/` and "
              "`outputs/kaggle_deberta/`. Full record: `05-name-that-llm/outputs/SESSION_NOTES.md`.",
              "", "## Recovery Events (2026-10-22 12:00-14:20 UTC)", "",
              "| Event | Outcome |", "|---|---|",
              "| Colab Pro A100 `pro7b-1792644801` | **LOST** - Google terminated it (404/401) after the exec; the runtime has no `zip` binary so `out.zip` never existed and the OOF matrices died with the VM. Fold accuracies survived in the log: 0.3608 / 0.4800 / - / **0.5063** / 0.5007 grouped balanced accuracy. |",
              "| Colab Pro A100 entitlement | 'Backend rejected accelerator A100 - no entitlement' -> new A100 sessions are impossible; Pro **T4 works** and is now the Pro lane. |",
              "| Kaggle `ntl-stage2-deberta` | **CANCELLED** by Kaggle (12 h limit) -> replaced by `ntl-real-deberta` (COMPLETE, real-row OOF balanced accuracy **0.4016**). |",
              "| host `triton` | broken install (no `triton.runtime`) silently broke every `generate()` -> 0 ARC candidates. Fixed with triton 3.2.0 in the first sys.path dir + eager/SDPA attention. |",
              "| Track 05 real-row blend | cross-fitted (weights on 4 folds, applied to the 5th) balanced accuracy **0.4016** with DeBERTa alone (stage-1 0.3667). |",
              "| Live lanes now | Colab Pro T4 (Qwen2.5-3B LoRA) + Kaggle T4 x2 (ntl-qwen-lora 7B 4-bit, arc-agi2-llm-ttc v5) + local RTX 3050 (Qwen2.5-0.5B LoRA). |",
              "", "## Deliverable Checklist", ""]
    checks = [
        ("01-nfl-big-data-bowl-2027", "notebooks/nfl_combine_to_game_kce.ipynb"), ("01-nfl-big-data-bowl-2027", "outputs/WRITEUP.md"),
        ("01-nfl-big-data-bowl-2027", "outputs/figures/fig1_cross_product_inside_outside.png"),
        ("01-nfl-big-data-bowl-2027", "outputs/figures/fig8_per_position_breakdown.png"),
        ("01-nfl-big-data-bowl-2027", "outputs/model_metrics.json"),
        ("01-nfl-big-data-bowl-2027", "outputs/DATA_SCHEMA.md"),
        ("03-arc-prize-2026-paper-track", "outputs/paper.tex"),
        ("03-arc-prize-2026-paper-track", "outputs/CITATIONS.md"),
        ("04-arc-prize-2026-arc-agi-2", "outputs/submission_competition.json"),
        ("04-arc-prize-2026-arc-agi-2", "build_competition_submission.py"),
        ("05-name-that-llm", "outputs/SUBMISSION_NOTES.md"),
        ("05-name-that-llm", "src/real_row_blend.py"),
        ("05-name-that-llm", "outputs/SESSION_NOTES.md"),
        ("05-name-that-llm", "outputs/ensemble_recipe.json"),
        ("05-name-that-llm", "src/make_submission.py"),
        ("05-name-that-llm", "outputs/deploy/manifest.json"),
        ("05-name-that-llm", "outputs/VERIFICATION.md"),
        ("05-name-that-llm", "outputs/deploy/lora_qwen05b/deploy_meta.json"),
        ("06-playground-s6e10-airline-satisfaction", "outputs/model_report.md"),
        ("02-gemma-4-developer-agent-paper", "src/gg_spv_agent.py"), ("02-gemma-4-developer-agent-paper", "src/agent.yaml"), ("02-gemma-4-developer-agent-paper", "outputs/PAPER_WRITEUP.md"),
        ("02-gemma-4-developer-agent-paper", "outputs/paper.tex"),
        ("03-arc-prize-2026-paper-track", "outputs/WRITEUP.md"), ("03-arc-prize-2026-paper-track", "outputs/cover_image.png"),
        ("04-arc-prize-2026-arc-agi-2", "outputs/submission.json"), ("04-arc-prize-2026-arc-agi-2", "outputs/eval_metrics.json"),
        ("04-arc-prize-2026-arc-agi-2", "notebooks/arc_agi_2_solver.ipynb"),
        ("05-name-that-llm", "outputs/oof_predictions.parquet"), ("05-name-that-llm", "outputs/cv_report.json"),
        ("05-name-that-llm", "outputs/submission.parquet"), ("05-name-that-llm", "notebooks/inference_submission.ipynb"),
        ("06-playground-s6e10-airline-satisfaction", "outputs/oof_blend.npy"), ("06-playground-s6e10-airline-satisfaction", "outputs/cv_scores.json"), ("06-playground-s6e10-airline-satisfaction", "outputs/submission.csv"),
        ("07-opencv-ai-competition-2026", "outputs/DEVPOST_SUBMISSION.md"), ("07-opencv-ai-competition-2026", "outputs/architecture_diagram.png"),
        ("07-opencv-ai-competition-2026", "benchmarks/run_eval.py"), ("07-opencv-ai-competition-2026", "src/generate_demo_video.py"),
        ("08-openzee-openair-2026", "outputs/RECON.md"),
    ]
    for tid, rel in checks:
        p = os.path.join(ROOT, tid, rel)
        ok = os.path.exists(p)
        sz = os.path.getsize(p) if ok else 0
        lines.append(f"- [{'x' if ok else ' '}] T{tid} `{rel}` {'(' + str(sz) + ' B)' if ok else '- MISSING'}")

    lines += ["", "## Hard Rules", "",
              "- NO Kaggle/Devpost submissions: all predictions live in `outputs/`.",
              "- Kaggle GPU <= 44h/week, keep 1h emergency reserve.",
              "- Colab Pro: billable from second 1; always stop the session when done.",
              "- Local RTX 3050 first for anything under 3.2 GB VRAM.",
              "- Never modify `/media/pope/virtbox/a_resume/`.", "",
              f"_generated by gen_master_status.py at {now}_"]
    open(os.path.join(ROOT, "MASTER_STATUS.md"), "w").write("\n".join(lines) + "\n")
    print(f"MASTER_STATUS.md updated ({len(lines)} lines)")


if __name__ == "__main__":
    main()
