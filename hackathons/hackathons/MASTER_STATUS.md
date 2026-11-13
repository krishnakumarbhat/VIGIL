# MASTER STATUS - Multi-Track Kaggle Hackathon Marathon

_Last update: 2026-10-23 10:31 UTC_

## GPU Lanes

| Lane | State | Detail |
|---|---|---|
| Local RTX 3050 | ACTIVE | `NVIDIA GeForce RTX 3050 Laptop GPU, 14 MiB, 4096 MiB, 0 %` |
| Kaggle GPU [ntl-stage2-deberta] | "KernelWorkerStatus.CANCEL_ACKNOWLEDGED" | `ntl-stage2-deberta has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"` |
| Kaggle GPU [ps6e10-stack] | COMPLETE | `krishnakumarbhat/ps6e10-stack has status "KernelWorkerStatus.COMPLETE"` |
| Colab Pro T4 | RUNNING | `433184 /bin/sh -c pgrep -af 'run_track05_colab|colab_cli' | head -2` |
| Kaggle quota | 41.91h used / 45h total | resets 2026-10-24T00:00:00Z |

## Tracks

| # | Track | Prize | Deadline | Best validated score | Outputs | Size |
|---|---|---|---|---|---|---|
| 01 | NFL Big Data Bowl 2027 | $100k | 2027-01-06 | translation_attrib::both::lgbm=0.2825 | 25 files | 30294 KB |
| 02 | Gemma 4 Developer Agent Paper | $35k | 2026-11-12 | fidelity_sweep_pass|localized=0.571->0.714 | 26 files | 1114 KB |
| 03 | ARC Prize 2026 Paper Track | $450k | 2026-11-09 | canonical_stats=d4=8;orbit_distinct=8 | 16 files | 1537 KB |
| 04 | ARC Prize 2026 ARC-AGI-2 | $700k | 2026-11-02 | pass2_evaluation=0.0 | 10 files | 3834 KB |
| 05 | Name That LLM | $50k | 2027-01-14 | stack_v2_crossfit_bal_acc=0.68909 | 212 files | 5014593 KB |
| 06 | Playground S6E10 Airline | swag | 2026-10-31 | cat_sweep_cat_d8_i3k_lr03_s42_full=0.958858 | 47 files | 118768 KB |
| 07 | OpenCV AI Competition 2026 | $20,250 | 2026-10-26 | cool_arm64_emulated_detection_p50_ms=2886.217 | 13 files | 2814 KB |
| 08 | OpenZee OpenAir 2026 | ? | ? | - | 2 files | 5 KB |

## Algorithm Update (2026-10-23)

Track 05 rebuilt its stack while burning the remaining Colab Pro units. Every number is 5-fold `StratifiedGroupKFold(shuffle, seed 42, groups=prompt_id)` over the 2,699 real rows, scored only on held-out rows.

| Step | Cross-fitted balanced accuracy | Change |
|---|---|---|
| DeBERTa-v3-base alone (session start) | 0.4016 | — |
| + Qwen2.5-0.5B LoRA, temperature + weight search + **per-class bias tuning** | 0.5557 | +15.4 pts, kills 4 collapsed classes (min recall 0.047 -> 0.35) |
| + decorrelated TF-IDF/SGD member | 0.5957 | +4.0 pts from a member that scores only 0.36 alone |
| + stylometry, char-only, word-only, cosine-kNN members | 0.5976 | +0.2 pts (diminishing) |
| + `svmext` (TF-IDF trained with 40k external rows) | 0.6139 | +1.6 pts |
| + Qwen2.5-0.5B LoRA **seed 1337** (mirror 6k, len 256) | **0.6817** | +6.8 pts; that member alone scores 0.6483 |

Per-class recall is now 0.48-0.92 (was 0.047-0.88). Fold-level blend scores 0.4148 / 0.4963 / 0.6815 / 0.8685 / 0.9481 — differences below ~0.02 are inside noise.

| Approach | Verdict |
|---|---|
| Linear meta-learner over concatenated member log-probs (630 parameters) | 0.5709 — **worse** than the constrained weight+bias model; 2,159 rows cannot support it |
| Qwen2.5-7B LoRA, A100 5-fold curriculum | Fold 0 = 0.4759, but the CLI session registry was pruned mid-run and orphaned the VM: artifacts unreachable. Stopped rather than burn units. |
| DeBERTa-v3-large trained in pure bf16 | Did not converge (loss pinned at ln 18 for 2,400 steps). Needs fp32 master weights + autocast. |
| `colab download` as the artifact channel | Replaced by in-band base64 over the exec stream + `src/ingest_b64.py`; the file API dies with the session registry. |

Deployment: `outputs/ensemble_recipe.json` (temps/weights/biases) + `src/make_submission.py` reproduces the blend on any test frame from artifacts in `outputs/deploy/` and `outputs/kaggle_deberta/`. Full record: `05-name-that-llm/outputs/SESSION_NOTES.md`.

## Recovery Events (2026-10-22 12:00-14:20 UTC)

| Event | Outcome |
|---|---|
| Colab Pro A100 `pro7b-1792644801` | **LOST** - Google terminated it (404/401) after the exec; the runtime has no `zip` binary so `out.zip` never existed and the OOF matrices died with the VM. Fold accuracies survived in the log: 0.3608 / 0.4800 / - / **0.5063** / 0.5007 grouped balanced accuracy. |
| Colab Pro A100 entitlement | 'Backend rejected accelerator A100 - no entitlement' -> new A100 sessions are impossible; Pro **T4 works** and is now the Pro lane. |
| Kaggle `ntl-stage2-deberta` | **CANCELLED** by Kaggle (12 h limit) -> replaced by `ntl-real-deberta` (COMPLETE, real-row OOF balanced accuracy **0.4016**). |
| host `triton` | broken install (no `triton.runtime`) silently broke every `generate()` -> 0 ARC candidates. Fixed with triton 3.2.0 in the first sys.path dir + eager/SDPA attention. |
| Track 05 real-row blend | cross-fitted (weights on 4 folds, applied to the 5th) balanced accuracy **0.4016** with DeBERTa alone (stage-1 0.3667). |
| Live lanes now | Colab Pro T4 (Qwen2.5-3B LoRA) + Kaggle T4 x2 (ntl-qwen-lora 7B 4-bit, arc-agi2-llm-ttc v5) + local RTX 3050 (Qwen2.5-0.5B LoRA). |

## Deliverable Checklist

- [x] T01-nfl-big-data-bowl-2027 `notebooks/nfl_combine_to_game_kce.ipynb` (123503 B)
- [x] T01-nfl-big-data-bowl-2027 `outputs/WRITEUP.md` (12945 B)
- [x] T01-nfl-big-data-bowl-2027 `outputs/figures/fig1_cross_product_inside_outside.png` (219059 B)
- [x] T01-nfl-big-data-bowl-2027 `outputs/figures/fig8_per_position_breakdown.png` (129923 B)
- [x] T01-nfl-big-data-bowl-2027 `outputs/model_metrics.json` (128461 B)
- [x] T01-nfl-big-data-bowl-2027 `outputs/DATA_SCHEMA.md` (11995 B)
- [x] T03-arc-prize-2026-paper-track `outputs/paper.tex` (7115 B)
- [x] T03-arc-prize-2026-paper-track `outputs/CITATIONS.md` (3307 B)
- [x] T04-arc-prize-2026-arc-agi-2 `outputs/submission_competition.json` (300032 B)
- [x] T04-arc-prize-2026-arc-agi-2 `build_competition_submission.py` (6058 B)
- [x] T05-name-that-llm `outputs/SUBMISSION_NOTES.md` (3757 B)
- [x] T05-name-that-llm `src/real_row_blend.py` (9943 B)
- [x] T05-name-that-llm `outputs/SESSION_NOTES.md` (6228 B)
- [x] T05-name-that-llm `outputs/ensemble_recipe.json` (1579 B)
- [x] T05-name-that-llm `src/make_submission.py` (9332 B)
- [x] T05-name-that-llm `outputs/deploy/manifest.json` (972 B)
- [x] T05-name-that-llm `outputs/VERIFICATION.md` (2914 B)
- [x] T05-name-that-llm `outputs/deploy/lora_qwen05b/deploy_meta.json` (556 B)
- [x] T06-playground-s6e10-airline-satisfaction `outputs/model_report.md` (3699 B)
- [x] T02-gemma-4-developer-agent-paper `src/gg_spv_agent.py` (56792 B)
- [x] T02-gemma-4-developer-agent-paper `src/agent.yaml` (724 B)
- [x] T02-gemma-4-developer-agent-paper `outputs/PAPER_WRITEUP.md` (12381 B)
- [x] T02-gemma-4-developer-agent-paper `outputs/paper.tex` (7245 B)
- [x] T03-arc-prize-2026-paper-track `outputs/WRITEUP.md` (10569 B)
- [x] T03-arc-prize-2026-paper-track `outputs/cover_image.png` (452426 B)
- [x] T04-arc-prize-2026-arc-agi-2 `outputs/submission.json` (1916570 B)
- [x] T04-arc-prize-2026-arc-agi-2 `outputs/eval_metrics.json` (31218 B)
- [x] T04-arc-prize-2026-arc-agi-2 `notebooks/arc_agi_2_solver.ipynb` (7591 B)
- [x] T05-name-that-llm `outputs/oof_predictions.parquet` (51427 B)
- [x] T05-name-that-llm `outputs/cv_report.json` (2659 B)
- [x] T05-name-that-llm `outputs/submission.parquet` (1755 B)
- [x] T05-name-that-llm `notebooks/inference_submission.ipynb` (6960 B)
- [x] T06-playground-s6e10-airline-satisfaction `outputs/oof_blend.npy` (5597208 B)
- [x] T06-playground-s6e10-airline-satisfaction `outputs/cv_scores.json` (1096 B)
- [x] T06-playground-s6e10-airline-satisfaction `outputs/submission.csv` (7951070 B)
- [x] T07-opencv-ai-competition-2026 `outputs/DEVPOST_SUBMISSION.md` (29107 B)
- [x] T07-opencv-ai-competition-2026 `outputs/architecture_diagram.png` (440302 B)
- [x] T07-opencv-ai-competition-2026 `benchmarks/run_eval.py` (33332 B)
- [x] T07-opencv-ai-competition-2026 `src/generate_demo_video.py` (15439 B)
- [x] T08-openzee-openair-2026 `outputs/RECON.md` (4178 B)

## Hard Rules

- NO Kaggle/Devpost submissions: all predictions live in `outputs/`.
- Kaggle GPU <= 44h/week, keep 1h emergency reserve.
- Colab Pro: billable from second 1; always stop the session when done.
- Local RTX 3050 first for anything under 3.2 GB VRAM.
- Never modify `/media/pope/virtbox/a_resume/`.

_generated by gen_master_status.py at 2026-10-23 10:31 UTC_
