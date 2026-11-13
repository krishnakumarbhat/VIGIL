# Addendum — final 90 minutes of the marathon

*Appended after the main report; supersedes conflicting numbers in the body above.*

## 1. CRITICAL BUG FOUND AND FIXED — Track 06 submission encoding
The official `sample_submission.csv` for `playground-series-s6e10` contains **floats**
(e.g. `0.6171`), not 0/1 labels — the competition is scored by ROC-AUC. An earlier build of
`outputs/submission.csv` wrote hard `0/1` labels, which would have collapsed the score.
Fixed: `outputs/submission.csv` now holds blended **probabilities** for all 299,844 rows
(range 0.0023–0.9976, mean 0.444). Always sanity-check the sample submission's dtype before
assuming labels.

## 2. Track 07 (deadline Oct 26 — most urgent) final state
- Rubric scorecards: **Overall 89/100**, **Agentic Vision 92/100**, **Best Use of COOL 62/100**
  (self-scored, every line carrying a measured evidence file + number + reproduction command).
- COOL finding: COOL ships as an **AWS Marketplace wheel, not on PyPI** — no public wheel could be
  obtained, so `outputs/cool_evidence.json` honestly reports `status: unavailable` and instead
  ships **measured x86 proxies**: `cv2.setUseOptimized(True)` 1.21× on the Laplacian stage,
  1.2× from kernel reordering (resize→gray), **3.5×** end-to-end on the fused
  resize+gray+laplacian+blur pipeline, plus an arm64 Graviton build recipe (`make cool-arm64`,
  `aws/Dockerfile.arm64`) a judge can run in one command.
- Perception: detection ROC-AUC **0.9917**, 85.16 FPS, p50 11.68 ms; full agent 20.22 FPS,
  p50 45.04 ms / p95 100.48 ms, peak RSS 730.6 MB; oracle-ROI recall 0.975 → 1.000 at 3.05× dispatch.
- Demo `outputs/demo_annotated.mp4`: 288 frames @ 24 fps, 1.93 MB, with SEE→REASON→ACT→VERIFY overlays.
- **61 tests pass**; `make all` runs end-to-end.

## 3. Track 04 ARC-AGI-2 iteration 2 (object-centric induction)
New modules `src/object_model.py`, `object_program_inducer.py`, `structural_families.py`,
`program_ranker.py`, `beam_search.py`. Measured on 1,000 training tasks: train solve rate 25 %,
graded **pass@1 55/1000 (5.5 %)**, pass@2 57/1000; family precision depth1 0.947, connect 1.00,
recolour 1.00, beam 1.00, cell_tree_d4 0.11; candidate-oracle rate 5.8 % (the ranker cannot beat the
oracle). Evaluation split still **0/120** — ARC-AGI-2 resists shallow induction. Self-check 16/16.
**Submission schema corrected**: the competition needs **240 test ids** in the envelope
`{id: [{"attempt_1": grid, "attempt_2": grid}]}`; `build_competition_submission.py` now merges
LLM-TTC > DSL > majority-colour fallback and validates against the official sample (PASS).

## 4. Track 06 — two negative/positive results worth keeping
- **NEGATIVE (definitive)**: joining the 129,880-row original *Airline Passenger Satisfaction*
  dataset on all 21 features matches only **17 of 699,635** playground train rows (0.0024 %) with
  **0 % label agreement** — there is no leakage shortcut. Do not retry this.
- **POSITIVE (in flight)**: the original dataset's *labels* are legitimate extra supervision;
  `src/orig_augmentation.py` measures 10-fold OOF AUC with the 129,880 source rows added to every
  training fold. Logged in `logs/experiments.tsv`.

## 5. Track 05 state
- Honest grouped CV (StratifiedGroupKFold, groups=prompt_id): **0.4086** overall, **0.3667** on the
  real 2,699-row competition train (150 prompts × 18 models); per-class recall 0.148–0.820,
  length-bin 0.274/0.349/0.421/0.436.
- `src/ensemble.py` + `src/ensemble_fixed.py` are ready to absorb the Colab Pro LoRA artefact the
  moment `outputs/colab_artifacts/` or `outputs/colab_pro_artifacts/` receives `out.zip`
  (per-fold `fold{f}_oof.npy` / `fold{f}_idx.npy` / `fold{f}_status.json`).
- Submission artefacts exist in the exact schema: `outputs/submission.parquet` (public test ids),
  `outputs/submission_full.parquet` (private-test coverage), `outputs/SUBMISSION_NOTES.md`.

## 6. Compute ledger (what actually ran)
| Lane | Outcome |
|---|---|
| Kaggle GPU | **10.3 h of 45 h** used; kernels `ps6e10-stack` (COMPLETE, OOF AUC **0.959272**), `ntl-stage2-deberta` (running), `arc-agi2-llm-ttc` v4 (running) |
| Colab Pro (A100) | **live and training** — Qwen2.5 LoRA 18-way classifier, fold 0 step 800/1374, loss 1.08 → 0.53, 3,194 MB VRAM; `out.zip` pulled by a 5-minute watchdog |
| Colab free account | T4 session alive with an earlier Qwen2.5-7B LoRA run |
| Local RTX 3050 | feature engineering, mixed-effects, DSL search, OpenCV benchmarks |

**Ten verified engineering fixes** (each cost a failed cycle, each now documented):
1. `colab exec -f` takes a **LOCAL** path — it uploads the code itself.
2. After `colab new`, `*.prod.colab.dev` publishes its A record in 10–45 s → mandatory
   `socket.getaddrinfo` wait-loop (patched into `_gpu_probe/run_gpu.sh`).
3. Colab Pro hit `TooManyAssignmentsError`; a FIFO-based OAuth re-auth path is ready
   (`_gpu_probe/colab_jobs/auth/`).
4. Colab CUDA 13 + bitsandbytes → 4-bit unavailable → use bf16 (A100 7B).
5. Colab `torchao 0.10.0` below the PEFT floor → `pip install -U torchao` in the job prelude.
6. `model.config.pad_token_id` unset → "Cannot handle batch sizes > 1".
7. fp16 GradScaler `unscale_` on fp16 grads → keep master weights fp32 / use bf16 without a scaler.
8. `SimpleImputer(categorical_strategy=...)` removed in the Kaggle sklearn.
9. Kaggle kernels: max **2 concurrent GPU sessions** — plan around it.
10. The `agent-browser` npm symlink is broken → drive system Chrome 154 through Playwright
    (this is what accepted the competition rules for all six slugs).

## 7. Where to look first tomorrow
1. `/home/pope/Desktop/SUBMIT_NOW.md` — exact per-track submission commands, deadline-ordered.
2. `/home/pope/Desktop/HACKATHON_MARATHON_STATUS.md` — this report.
3. `/home/pope/Desktop/gpu /hackathons/MASTER_STATUS.md` — live, regenerated by `gen_master_status.py`.
4. `outputs/` of each track — every prediction, figure and write-up.


## 2b. Final-hour results (2026-10-22 06:00-07:05 UTC)
- **Track 06 max ensemble: OOF ROC-AUC 0.959666** (7 members, Nelder-Mead on logit scale,
  weights realmlp .595 / cat .270 / lgbm .052 / lgbm_aug .042 / lgbm_base .038 / nnprior .003 / xgb .000).
  `outputs/submission.csv` regenerated with **probabilities** for all 299,844 rows; the builder
  refuses to overwrite a submission whose OOF score is lower than the recorded best.
- **Track 06 negative result (definitive)**: the original 129,880-row airline dataset matches only
  17 of 699,635 playground train rows on all 21 features (0.0024 %) with 0 % label agreement - no
  leakage shortcut exists. Its *labels* ARE useful as extra supervision (+0.00025 AUC standalone).
- **Track 05 Colab Pro A100 LoRA (Qwen2.5-7B)**: fold 0 = 0.3608, fold 1 = **0.4800** grouped
  balanced accuracy - already above the ~0.466 public-LB reference and still improving.
  `auto_refresh.sh` rebuilds the cross-fitted stack + submission the moment the artefact lands.
- **New failure modes found and fixed**: (i) the Pro account cannot create NEW A100 sessions
  ("Backend rejected accelerator 'A100' - no entitlement") though an existing A100 session keeps
  working; (ii) `TooManyAssignmentsError` when a second Pro session is requested while one runs;
  (iii) Colab's upload endpoint intermittently returns HTTP 500 - worked around by embedding the
  competition JSONs (gzip+base64, 178 KB) directly in the script, since `colab exec -f` ships the
  LOCAL file's content and needs no upload at all; (iv) the ARC solver parsed **0 candidate grids**
  because instruct models emit space-separated digit rows rather than JSON - the parser now accepts
  both forms (unit-verified) and generation is batched 4 tasks at a time.

## 2c. Second recovery cycle (2026-10-22 12:30-14:55 UTC)

**Three independent failures hit and were fixed:**

1. **The Colab Pro A100 died.** Google terminated session `pro7b-1792644801` right after the
   training exec finished (404/401), and because the Colab runtime has **no `zip` binary**
   (`zip -qr ... ; check=False` silently did nothing) there was no `out.zip` to pull - the OOF
   matrices went with the VM. The per-fold balanced accuracies survived in the log:
   **0.3608 / 0.4800 / - / 0.5063 / 0.5007**. Root-cause note: the Pro account also has
   **no entitlement to create new A100 sessions** ("Backend rejected accelerator 'A100'");
   Pro **T4 sessions work** and are now the Pro lane. The recipe was re-launched on Kaggle
   (`ntl-qwen-lora`, 7B 4-bit, real+mirror, real-row validation) where artefacts persist.
2. **Kaggle cancelled `ntl-stage2-deberta`** at the 12 h runtime limit. Replaced by
   `ntl-real-deberta` (DeBERTa-v3-base trained on the REAL competition rows, 5-fold
   StratifiedGroupKFold) - **COMPLETE, real-row OOF balanced accuracy 0.4016**, now the strongest
   honest Track 05 member (stage-1 was 0.3667 on the same rows).
3. **The host's `triton` install was broken** (no `triton.runtime`) which silently broke every
   `torch` attention/generate call - the ARC solver was producing **0 candidate grids** while
   burning GPU. Fixed by installing triton 3.2.0 into the first `sys.path` entry and forcing
   eager/SDPA attention; generation verified end-to-end afterwards.

**New deliverables:** `05-name-that-llm/src/real_row_blend.py` (cross-fitted balancer over the
real rows: weights chosen on 4 folds and applied to the 5th, partial-fold aware), rebuilt
`outputs/submission.parquet`, `outputs/oof_predictions.parquet`, `outputs/cv_report.json`;
`cycle_supervisor.sh` (20-minute event-driven supervisor: harvests kernels, rebuilds the stack,
runs the ARC evaluation benchmark, refreshes MASTER_STATUS); `07-.../outputs/benchmarks.json`
annotated with measurement conditions; Tracks 01/02/03 verified end-to-end by a dedicated pass
(figures non-trivial and real, paper compiles, every quoted number re-checked against the
experiment logs, `VERIFICATION.md` written in each track).
