# HACKATHON MARATHON — MASTER STATUS REPORT

**Generated:** 2026-10-22 03:57 UTC (09:27 IST, machine local +0530)
**Workspace:** `/home/pope/Desktop/gpu /hackathons/` (note the literal trailing space in `gpu ` — always quote)
**Kaggle account:** `krishnakumarbhat`
**Session window covered:** 2026-10-21 ~23:40 UTC → 2026-10-22 03:57 UTC (~4 h 20 min; the "~5-hour marathon")
**Sub-agent report file:** `/home/pope/Desktop/HACKATHON_MARATHON_STATUS.md` (copy: `hackathons/MASTER_REPORT.md`)

---

## 1. Executive summary

In this marathon session eight Kaggle/Devpost tracks were bootstrapped from zero to
reviewable artefact sets on three compute tiers (local RTX 3050, two Kaggle T4 GPU
kernels, and a Colab Pro T4/A100 lane) with a hard "never submit" policy — every
prediction ships in its track's `outputs/` for the user to submit manually.
Five of the eight tracks now have a complete deliverable package with measured
numbers and provenance: **T07 OpenCV** (61 tests pass, agent 31–44 FPS, detection
ROC-AUC 0.9917, Devpost writeup self-scored 82/100), **T02 Gemma-4 GG-SPV paper**
(recall@1 0.045 → 0.182 with graph guidance, −43 % tokens, verifier AUROC 0.875/0.905,
LaTeX + PDF), **T03 ARC paper track** (writeup + LaTeX + PDF + cover image + ablations),
**T01 NFL BDB 2027** (headline OOF R² 0.6561 on a self-declared *proxy* target,
8 figures, notebook 8/8 cells, 1876-word writeup), and **T08 OpenZee** (reconclusive
negative: the competition does not exist). **T05 Name-That-LLM** improved grouped-CV
balanced accuracy from 0.3906 → **0.4086** via a cross-fitted probability stacker,
then correctly *overturned* a later +0.10 length-prior "win" (0.5119) as an
estimator-coupling artefact — the honest headline stays 0.4086. **T06 S6E10 Airline**
moved OOF ROC-AUC 0.959265 → **0.959272** (4-model blend) with a RealMLP + kNN-prior
stack re-verification in flight that already reads 0.9596 strict cross-fitted.
**T04 ARC-AGI-2** moved train pass@1 6.7 % → 8.67 % on the v2 object-centric beam
solver. Substantial GPU waste occurred and is itemised below (two failed Colab Pro
A100 runs, a broken Kaggle TTC kernel, three Colab uploads lost to HTTP 500). One
live risk is open right now: **T06 `outputs/submission.csv` contains hard 0/1 labels
where the competition sample submission expects probabilities** — it is not
submission-ready in its current form.

---

## 2. Track table — ordered EARLIEST DEADLINE FIRST

| # | Competition (track) | Prize | Deadline (UTC) | Best measured score (metric) | Key artefacts | Readiness |
|---|---|---|---|---|---|---|
| 07 | OpenCV AI Competition 2026 | $20,250 | **2026-10-26** | `full_agent_fps` = **31.15** (last run 03:30 UTC; range 20.22–44.38 across runs), `detection_roc_auc` **0.9917**, `recognition_roc_auc` **1.0**, **61 tests pass** — *self-built synthetic benchmark, PROXY for the official rubric* | `outputs/DEVPOST_SUBMISSION.md`, `outputs/benchmarks.json`, `outputs/demo_annotated.mp4`, `outputs/cool_evidence.json`, `outputs/architecture_diagram.png`, `aws/`, `tests/` | **Submission-ready** (Devpost, manual) |
| 06 | Playground Series S6E10 — Predicting Airline Satisfaction | Swag | **2026-10-31** | `final_oof_auc` = **0.959272** (`outputs/cv_scores.json`, 03:38 UTC) — official metric **is** ROC-AUC; strict cross-fitted re-verification of the RealMLP stack in flight reads **0.959607** (unverified, `/tmp/opencode/t06_strict.log`) | `outputs/cv_scores.json`, `outputs/oof_*.npy`, `outputs/submission.csv`, `outputs/cv_stack_nn_lr.json` | **In progress** — submission.csv format is WRONG (see §7 risk R1) |
| 04 | ARC Prize 2026 — ARC-AGI-2 | $700,000 | **2026-11-02** | v1 train (1000 tasks) `graded_pass1_rate` **0.067** (68/1000), `graded_pass2_rate` **0.005** (5/1000); v2 train subset (150) `graded_pass1` **0.13 → 8.67 %**; evaluation split (120) **0.0 / 0.0** — official metric is exact-match | `outputs/eval_metrics_training.json`, `outputs/eval_metrics_training_v2.json`, `outputs/eval_metrics_evaluation.json`, `outputs/submission.json`, `notebooks/arc_agi_2_solver.ipynb`, `kaggle_ttc/` | **In progress** — v2 evaluation-split run NOT yet done |
| 03 | ARC Prize 2026 — Paper Track | $450,000 | **2026-11-09** | Ablation `base_full` **0.4167** (25/60 solved, 8 s budget) vs `symbolic_only` **0.05** (3/60); `canonical_stats` d4=8, orbit_distinct=8 — *paper rubric, no official scalar metric* | `outputs/WRITEUP.md`, `outputs/paper.tex`, `outputs/paper.pdf`, `outputs/cover_image.png`, `outputs/CITATIONS.md`, `outputs/ablation_results.csv` | **Submission-ready** (manual review + upload) |
| 02 | Google — Gemma 4 Developer Agent **Paper Track** | $35,000 | **2026-11-12** | `recall_at_5` **0.3182** vs flat 0.2727; `recall_at_1` **0.1818** vs 0.0455 (4.0×); `mean_tokens_read` 2101.6 vs 3691.2 (**−43 %**); `verifier_auroc_hard_neg` **0.875** / easy **0.905**; TPR 0.90 @ FPR 0.0385; simulated `pass_rate` 0.125 vs 0.025 flat — *localization measured on real artifacts, patch synthesis SIMULATED* | `outputs/PAPER_WRITEUP.md`, `outputs/paper.tex`, `outputs/paper.pdf`, `outputs/eval_metrics.json`, `outputs/ablation_results.csv`, `src/gg_spv_agent.py`, `src/agent.yaml` | **Submission-ready** (manual) |
| 01 | NFL Big Data Bowl 2027 | $100,000 | **2027-01-06** | Headline `cv_r2::sep_gain::by_game_play::lightgbm` = **0.6561** (MAE 1.3503, Spearman 0.8254); cushion-ablated `sep_gain_no_cushion::by_game_play` **0.1074**; `career_snaps_log::by_player` 0.5979 — **PROXY**: competition metric is not yet published | `outputs/WRITEUP.md` (1876 words), `outputs/model_metrics.json`, `outputs/INSIGHTS.md`, `outputs/DATA_SCHEMA.md`, `outputs/figures/fig1–fig8`, `notebooks/nfl_combine_to_game_kce.ipynb` (8/8 cells) | **In progress / review-ready** (analysis is complete; no submission object exists yet) |
| 05 | Name That LLM | $50,000 | **2027-01-14** | `bal_acc_stack_probs_c8` = **0.40863** (StratifiedGroupKFold 5, `groups=prompt_id`, clean n=9120) — official metric is balanced accuracy over 18 classes; baseline 0.39061 | `outputs/cv_stack_report.json`, `outputs/final_model_stacked.joblib`, `outputs/submission.parquet`, `outputs/cv_report.json`, `outputs/SUBMISSION_NOTES.md`, `notebooks/inference_submission.ipynb` | **In progress** (Kaggle DeBERTa stage-2 kernel still RUNNING; Colab LoRA lane blocked) |
| 08 | OpenZee OpenAir 2026 | unknown | unknown | **None — competition does not exist.** Absent from Kaggle API, Devpost API/HTML, web search, DNS probes and local Chrome history as of 2026-10-22 | `outputs/RECON.md`, `outputs/STARTER_TEMPLATE.md`, `src/{dataset,model,trainer}.py` | **Blocked** (phantom track; recommend dropping) |

Deadline data cross-checked against `COMPETITION_ACCESS.md` (Kaggle API `competitions list`)
and `gen_master_status.py`. Note two deadline corrections vs the working assumptions:
the **Gemma-4 main competition** (not the paper track) closes **2026-12-02** for $75k,
and **ARC-AGI-3** (also joined) closes **2026-11-02** for $850k — neither is tracked here.

---

## 3. GPU / compute utilisation

### 3.1 Local RTX 3050 (measured 09:19 IST / 03:49 UTC)

```
NVIDIA GeForce RTX 3050 Laptop GPU — Driver 580.173.02, CUDA 13.0
1573 MiB / 4096 MiB used, 5 % util, 50 °C, 9 W / 60 W
  PID 3286691  python3            402 MiB   (RealMLP / T06 strict-stack job)
  PID 2525     /usr/lib/xorg/Xorg   4 MiB
```
The 3.2 GB usable budget was respected: the RealMLP-TD 10-fold job
(`06-.../src/realmlp_stack.py`, launched ~07:18 IST) and the T04 v2 training rerun
(`src.run_v2 --split training … --workers 12`, PID 3311526) both ran under it.

### 3.2 Kaggle T4 kernels

Quota endpoint (`/api/v1/kernels/quota`) measured live: **31,872 s = 8.85 h used of
162,000 s = 45.00 h** (19.7 % consumed; 36.15 h remain). TPU quota untouched (0 / 20 h).
`gen_master_status.py` read 5.11 h at 01:08 UTC — the extra ~3.7 h is the ARC TTC and
PS6E10 stack kernels.

| Kernel | Type | Status (live) | Last run | Notes |
|---|---|---|---|---|
| `krishnakumarbhat/ntl-stage2-deberta` | GPU (DeBERTa-v3-base 5-fold) | **RUNNING** | 2026-10-21 22:29 UTC | in flight >5 h; OOF probs are the next T05 stack member |
| `krishnakumarbhat/ps6e10-stack` | GPU | **COMPLETE** | 2026-10-22 02:04 UTC | output download timed out locally (>120 s) — not yet pulled |
| `krishnakumarbhat/arc-agi2-llm-ttc` | GPU | **ERROR** | 2026-10-22 03:44 UTC | died at t=18.4 s: `NameError: name '_subprocess' is not defined` (see §3.4) |
| `nfl2027-data-extract` | CPU | complete | 2026-10-21 20:39 UTC | mounted competition data out via `/kaggle/working` |
| `gemma4-data-extract-cpu` | CPU | complete | 2026-10-21 20:31 UTC | same trick for T02 |
| `nfl2027-mount-probe`, `conn-probe-comp-data`, `conn-probe-gpu` | CPU | complete | 2026-10-21 | access probes |

The **2-simultaneous-GPU-slot limit** was hit: `ps6e10-stack v4` sat queued behind
`ntl-stage2-deberta` (`experiments/worklog.md`, GPU-lane note).

### 3.3 Colab Pro — accounts, sessions, compute units

Two accounts, two config files: `~/.config/colab-cli/sessions.pro.json` (Pro) and
`sessions.old.json` (free/old).

| Session | Account | Hardware | State (live) |
|---|---|---|---|
| `proarc-1792640682` (`gpu-t4-s-kkb-ass1c1-sbtj066ifdik`, asia-southeast1) | pro | T4 | **ALIVE** — keep-alive process PID 3302327 running; no artifacts downloaded; **billable** |
| `track05-1792636477` | old | T4 | exec still attached (PID 3220251, `track05_colab_job.py`, 18000 s timeout) |
| `pro7b-1792640639` | pro | **A100 requested** | **TERMINATED** 03:47:17 UTC after job crash |
| `proarc-1792640515` | pro | T4 | **TERMINATED** 03:44:30 UTC, no artifacts |

Compute-unit accounting is not exposed by the CLI; only wall-clock is logged
(`_gpu_probe/usage.log` via `run_gpu.sh::usage_log`). Both Pro runs today cost
billable units for **zero** returned compute.

**The OAuth re-auth issue.** `_gpu_probe/colab_jobs/auth/pro_auth.log` ends with
`AUTH_FLOW_EXIT`; `auth/pro_oauth_url.txt` is **0 bytes**, and the flow is parked on a
FIFO (`auth/code.fifo`, `fifo_path.txt`) waiting for a human to paste the authorization
code. `colab_pro_oauth_url.txt` at the probe root is likewise empty. Two bash wrappers
(PIDs 3291336/3291341) are blocked on that FIFO right now — the Pro lane cannot be
relaunched until the code is pasted.

**`TooManyAssignmentsError`** — confirmed verbatim in
`05-name-that-llm/logs/colab_track05.nohup9.log`:
```
TooManyAssignmentsError: Failed to issue request POST
https://colab.research.google.com/tun/m/assign?nbh=…&variant=GPU&accelerator=T4:
Precondition Failed
```
raised from `colab_cli/.../raise TooManyAssignmentsError(str(e))` on HTTP 412.
Mitigation shipped in the launcher scripts: up to 6–10 attempts with 90–120 s backoff
and a **unique session name per attempt** (`run_track05_colab.sh`, `run_pro_burner.sh`,
`run_pro_arc.sh`) so a retry never stops somebody else's session.

### 3.4 What was burned and what was wasted

**Productive:** local RTX 3050 (RealMLP 10-fold + kNN-prior + T04 v2 rerun), 2 Kaggle
GPU kernels, ~8.85 h of the 45 h weekly quota, and the T05 old-account T4 lane.

**Wasted (all Pro, all billable):**
1. **`pro7b-1792640639` (A100, 03:44:57 → 03:47:02 UTC, ~2 min exec + session spin-up).**
   `Qwen/Qwen2.5-7B` + `Qwen2ForSequenceClassification` head:
   `ValueError: Cannot handle batch sizes > 1 if no padding token is defined.`
   The job never reached the training loop; `/content/out.zip` never existed, so the
   download step failed and no artifact returned. A second identical failure is recorded
   at 03:47:17 in `colab_pro7b.log` / `colab_pro7b.nohup2.log`.
2. **`pro7b` earlier attempt** (`colab_pro7b.nohup.log`): `bitsandbytes 0.50.2` installed
   fine, but `MODEL_INIT_ERROR ImportError('Found an incompatible version of torchao.
   Found version 0.10.0, but only versions above 0.16.0 are supported')` from
   `peft/tuners/lora/torchao.py::dispatch_torchao` — then the same `FALLBACK_INIT_ERROR`.
3. **`proarc-1792640682`**: three uploads of the ARC JSONs each failed with
   `Upload failed: 500 Server Error … /api/contents/content/arc/{eval_ch,eval_sol,test_ch}.json`,
   then the exec ran anyway on empty inputs; `Download failed: File or directory not
   found: /content/out.zip`; `outputs/pro_artifacts/` is **empty**.
4. **`kaggle/arc-agi2-llm-ttc` ERROR** — a one-line bug:
   `import subprocess, sys as _sys` appears at line ~36 but
   `_subprocess.run(...)` is called at line 40, *after* the name `_subprocess` was never
   bound. The kernel mounted the competition data correctly (`files /kaggle/input/
   competitions/arc-prize-2026-arc-agi-2/...` was printed) and then crashed.

### 3.5 Exact fixes discovered (the ones that actually unblocked work)

| # | Problem | Fix | Where it lives |
|---|---|---|---|
| F1 | `colab exec -f <path>` failed | **`-f` takes a LOCAL path** (the file is uploaded itself); do not pass a remote path | `_gpu_probe/colab_jobs/run_track05_colab.sh` (`=== SANITY EXEC (local file path) ===`), `run_gpu.sh::colab_run` |
| F2 | `NameResolutionError` right after `colab new` | **DNS wait-loop**: `*.prod.colab.dev` publishes its A record **10–45 s after** `colab new`. Poll `socket.getaddrinfo(host,443)` every 3 s, up to 20–40 tries, before uploading | `_gpu_probe/run_gpu.sh::colab_wait_dns`, inlined in all three `run_pro_*.sh` / `run_track05_colab.sh` |
| F3 | `bitsandbytes` / `torchao` version conflicts on Colab | bnb 0.50.2 installs OK but **torchao 0.10.0 is below the >0.16.0 floor** `peft` requires → run the LoRA job in **bf16 without bitsandbytes** | `experiments/worklog.md` (GPU-lane note), `colab_pro7b.nohup.log:58,74` |
| F4 | `SimpleImputer(categorical_strategy=…)` | scikit-learn removed `categorical_strategy`; the shipped code uses `SimpleImputer(strategy="median")` inside a `make_pipeline` | `06-.../kaggle_track06_stack.py:123,131`, `06-.../kaggle_kernel_stack/kaggle_track06_stack.py` *(the original error text is not preserved in any file I read — the workaround is what shipped; **flagged as not independently verified**)* |
| F5 | fp16 `GradScaler.unscale_` error | guarded scaler: `scaler = torch.amp.GradScaler("cuda") if USE_SCALER else None`, with `scaler.unscale_(opt)` only under that guard | `_gpu_probe/colab_jobs/track05_colab_job.py:206,226`, `track05_lora_train.py:158,178`, `05-.../kaggle_track05_stage2.py:115,124` *(again the failure text itself is not in a log — **flagged as not independently verified**)* |
| F6 | Kaggle GPU slot contention | only 2 GPU kernels run at once; queue the third and poll `kaggle kernels status` to a terminal state | `experiments/worklog.md`; `run_gpu.sh::kaggle` deadline loop (`MAX_MIN*60 + 900` s slack) |
| F7 | Kaggle rules-acceptance API absent from CLI 1.7.4.5 | **Playwright + system Chrome 154** (the `agent-browser` npm **symlink is broken**) drove the rules page; verified afterwards by `curl …/data/download-all/<slug>` returning **HTTP 302** | `experiments/worklog.md:14`; `ACCESS_STATUS.md` |
| F8 | `colab exec` returns exit 0 even when the cell raises | grep the captured log for `Traceback \(most recent call last\)|SystemExit.*[1-9]` and force `rc=1` | `_gpu_probe/run_gpu.sh::colab_run` |
| F9 | `kaggle kernels list -m <user>` | wrong flag — the user goes in `--user <name>`; `-p` is the page number | live CLI error observed this session |
| F10 | Colab Pro auth expires mid-marathon | re-auth must be pasted by a human into a FIFO; the launcher is parked on it | `_gpu_probe/colab_jobs/auth/{pro_auth.log,code.fifo,fifo_path.txt}` |

---

## 4. Measured results per track (with provenance and PROXY flags)

### T01 — NFL Big Data Bowl 2027 — **PROXY METRIC**
Source: `01-.../outputs/model_metrics.json` (JSON), `01-.../logs/experiments.tsv`.

| Metric | Value | Type |
|---|---|---|
| `cv_r2::sep_gain::by_game_play::lightgbm` | **0.6561** (MAE 1.3503, ρ 0.8254) | **PROXY** — official metric unpublished |
| `cv_r2::sep_gain_no_cushion::by_game_play::lightgbm` | **0.1074** (MAE 2.2285) | honest ablation: `cushion` removed from features |
| `cv_r2::sep_at_pass_forward::by_game_play::lightgbm` | 0.282 (MAE 1.3864) | PROXY |
| `cv_r2::career_snaps_log::by_player::lightgbm` | 0.5979 | PROXY |
| `cv_r2::player_mean_sep_preregistered::by_player::lightgbm` | **−0.1263** | negative result, kept |
| `derived_proxy_validation_pearson` | 0.0334 (n=4078) | **REJECTED** — derived separation proxy fails validation |
| `mixedlm_population_oof_r2` | 0.1869 (5 folds converged) | mixed model, player random intercept |
| `mixedlm_kce_align_beta` | 0.00801 | alignment side is a *feature*, not a label |
| cuts detected | 5,618 events / 509 athletes, mean turn 87.3° | data fact |
| `writeup_final_words` / `figures_written_final` | 1876 / 8 | deliverable fact |
| `notebook_executed` | 8/8 code cells, 0 failed | deliverable fact |

**Caveat carried forward in the writeup itself:** `sep_gain = separation_at_pass_forward −
cushion`, and `cushion` is also supplied as a feature, so the 0.6561 headline contains a
deliberately measured algebraic component. The ablated 0.1074 is the number to quote for
"how much separation did the route *create*". Data: `combine_tracking.csv` 463,189 × 14,
`player_play.csv` 314,197 × 64, `game_tracking_2023.csv` 3,665,443 × 12 (only 156 athletes
shared with `players.csv` — the binding data limitation).

### T02 — Gemma 4 Developer Agent Paper — **patch synthesis is SIMULATED**
Source: `02-.../logs/experiments.tsv`, `outputs/ablation_results.csv`, `outputs/eval_metrics.json`.

| Configuration | R@1 | R@5 | mean tokens | tool calls | pass rate |
|---|---|---|---|---|---|
| `flat_baseline` (12b, f=0.8) | 0.0455 | 0.2727 | 3691.2 | 12.18 | 0.025 |
| `gg_hops0` | **0.1818** | 0.3182 | **2101.6** | 3.48 | 0.10 |
| `gg_hops1` | 0.1818 | 0.3182 | 3022.7 | 3.48 | 0.10 |
| `gg_spv_12b` (hops2 + verifier) | 0.1818 | 0.3182 | 3031.7 | 4.70 | **0.125** |
| `gg_spv_31b` (f=0.9) | 0.1818 | 0.3182 | 3031.7 | 4.70 | 0.125 |

Verifier (12b, f=0.8): `auroc_hard` **0.875**, `auroc_easy` **0.905**,
`tpr_at_threshold` **0.90**, `fpr_hard_at_threshold` **0.0385**, `fpr_easy` 0.0,
`gold_mean_conf` 0.7736 (n=10 localized gold / 26 hard / 30 easy).
Fidelity sweep `fidelity_sweep_pass|localized` = **0.571 → 0.714** (4/7 → 5/7, monotone).
**Honest negative result logged:** recall@5 is identical at hops = 0/1/2 (0.3182) while
1–2-hop expansion costs ~920 extra tokens/task. 40 tasks with local graph+embedding
artifacts, 22 with symbol ground truth; token counts are a **chars/4 proxy**.

### T03 — ARC Prize 2026 Paper Track — **rubric-scored, no official scalar**
Source: `03-.../outputs/ablation_results.csv`, `logs/experiments.tsv`.

| Ablation | split / limit | budget | solved | rate |
|---|---|---|---|---|
| `base_full` | training, 60 | 8.0 s | 25 | **0.4167** |
| `no_depth2` | training, 60 | 8.0 s | 24 | 0.40 |
| `symbolic_only` | training, 60 | 8.0 s | 3 | 0.05 |
| `symbolic_depth2_only` | training, 60 | 8.0 s | 3 | 0.05 |
| `eval_full_30` | evaluation, 30 | 15.0 s | 4 | 0.1333 |
| `canonical_stats` | 12×12 sample | — | — | d4_elements=8, orbit_distinct_keys=8, present_colors=10 |

`experiments/worklog.md` records the paper itself as **1330 words, 6 figures, 5 verified
citations** (writeup + LaTeX + PDF + cover image all on disk).

### T04 — ARC Prize 2026 ARC-AGI-2 — **official metric is exact match**
Source: `04-.../outputs/eval_metrics*.json`.

| Run | generated | split | tasks | solved | graded pass@1 | graded pass@2 |
|---|---|---|---|---|---|---|
| v1 | 01:24:16 UTC | evaluation | 120 | 20 (0.1667) | **0 (0.0)** | **0 (0.0)** |
| v1 | 01:28:33 UTC | training | 1000 | 353 (0.353) | **67 → 6.7 %** | **5 → 0.5 %** |
| v2 object-beam-ranked | **09:05:56 IST (03:35 UTC)** | training subset | 150 | 41 (0.2733) | **13 → 8.67 %** | 13 → 8.67 % |

v2 details: `candidate_oracle_pass1` 13/150 (0.0867), `ranker_top1_correct` 13
(`ranker_top1_rate` 0.3171 conditional on a candidate existing), `wall_seconds` 55.36,
mean 4.29 s/task, 12 workers, 0 unfinished. **The v2 evaluation-split run has not been
executed** — `eval_metrics_evaluation.json` is still the 01:24 UTC v1 artifact, and
`README_TRACK04.md` (last written 09:22 IST) still describes `eval_metrics.json` as the
evaluation split even though `eval_metrics.json` is now byte-identical to
`eval_metrics_training_v2.json`. A v2 training rerun with tighter budgets is in flight
right now (PID 3311526, `/tmp/opencode/t04/v2train3.log`).

### T05 — Name That LLM — **official metric is balanced accuracy (18 classes)**
Source: `05-.../logs/experiments.tsv`, `outputs/cv_stack_report.json`, `outputs/cv_report.json`.

| Iteration | Metric | Value | Verdict |
|---|---|---|---|
| stage-1 single best (`svc_extra_claude_w50`) | bal_acc | 0.38525 | keep (member) |
| hand blend `0.8·svc_extra_claude_w50 + 0.2·nbsvm_all` | bal_acc | 0.39061 | baseline |
| 3-member simplex `blend3_svc35` | bal_acc | 0.39062 | tie |
| **`stack_probs_c8`** — cross-fitted LR(C=8, balanced) over OOF probs | bal_acc | **0.40863** | **KEEP** (+0.0180) |
| LGBM stack | bal_acc | 0.383 | reject |
| LR + kNN-vote / margin / rank-mean variants | bal_acc | 0.345–0.379 | reject |
| logit-adjustment α>0 | — | hurts at every C | reject |
| `stack_lenprior` c32_m10 (length-bin priors) | bal_acc | 0.51191 | **OVERTURNED → DISCARD** |
| `stack_lenprior_decoupled` C=8 | bal_acc | 0.17321 (permuted bins 0.19333 vs anchor 0.40084) | DISCARD |

**The 0.51191 was retracted by the same session that produced it**: the bin-permutation
control scored 0.509 ≈ the "real" 0.512, and the decoupled-table test collapsed to
0.173/0.193 — the gain was an estimator-coupling artefact from same-fold leave-one-out
table overlap, not signal. **Headline stands at 0.40863.**
C-sweep behind it: C2 0.40344 / C3 0.40622 / C4 0.40626 / **C8 0.40863** / C16 0.40858 /
C32 0.40834 / C1000 0.40126.
Protocol: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`,
`groups=prompt_id`, clean subset n=9120. The separate `cv_report.json` (0.36672) belongs
to the earlier **real-competition-train-only** phase and is explicitly labelled stale by
`SUBMISSION_NOTES.md`.

### T06 — Playground S6E10 — **official metric is ROC-AUC**
Source: `06-.../outputs/cv_scores.json` (03:38 UTC), `cv_stack_nn_lr.json`, `logs/experiments.tsv`.

| Config | OOF AUC |
|---|---|
| `lgbm` / `cat` / `xgb` / `logreg` members | 0.957583 / 0.959052 / 0.956385 / 0.940585 |
| 3-member logit blend (baseline) | 0.959265 |
| 4-model weighted blend (`cv_scores.json`, weights 0.2035/0.7756/0.0209/1.6e-7) | **0.959272** |
| nnprior member alone (k=5 inv-dist) | 0.943050 |
| 18-config cross-fitted LR stack incl. ranks (n5a) | best 0.959242 → **DISCARD** |
| RealMLP+nnprior coordinate-ascent (in-sample, `realmlp_stack.py`) | 0.959628 |
| strict cross-fitted LR over 4 (lgbm/cat/xgb/realmlp) — **IN FLIGHT** | 0.959601–0.959607 so far |

Protocol: `StratifiedKFold(10, shuffle=True, random_state=42)`, n=699,636. The kNN-prior
injection (idea I6 leg a) was rejected because a 0.943 member adds nothing at this
operating point; the RealMLP leg is the live hypothesis and is currently the best number.

### T07 — OpenCV AI Competition 2026 — **PROXY for the official 100-pt rubric**
Source: `07-.../logs/experiments.tsv`, `outputs/benchmarks.json`, `outputs/cool_evidence.json`.

| Metric | Value (last run 03:37–03:38 UTC) |
|---|---|
| `detection_roc_auc` | **0.9917** (stable across 8 runs; first run was 0.5 on a broken synthetic set) |
| `recognition_roc_auc` | **1.0** |
| `full_agent_fps` | **20.22–31.15** (p50 45.0 ms); earlier runs 32.51–44.38 at lower load |
| `cool_status` | **unavailable** — "COOL runtime is not installed and no COOL wheel is publicly obtainable; aarch64 execution is QEMU-emulated, so it is not Graviton evidence" |
| `cool_dispatch_mean_speedup` / `median` | 1.52–1.58 / 1.182 |
| `cool_roi_latency_speedup` / `cool_roi_recall_crop` | 3.053–3.075 / 1.0 |
| `cool_arm64_emulated_detection_p50_ms` | 2886.217 (QEMU, explicitly **not** Graviton) |
| tests | **61 test functions across 9 files, all passing** (`tests/`, per `DEVPOST_SUBMISSION.md`) |
| demo | `demo_annotated.mp4`, 288 frames @ 24 fps, 1,929,361 B; event recall 1.0 vs 0.0 static |

Self-scored rubric in `DEVPOST_SUBMISSION.md`: 82/100 (Technical 26/30, Innovation 17/20,
Impact 15/20, UX 8/10, Docs 9/10, Cloud 7/10). Environment caveat recorded: **load
average 47.26** at benchmark time, so FPS is the least stable number in the report.

### T08 — OpenZee OpenAir 2026 — no measurements
`outputs/RECON.md` documents the negative finding with sources (Devpost API + HTML, Kaggle
API search, web search, DNS for `openzee.dev/.org/.com`, `openair.dev`, and a local Chrome
History SQLite query — all zero hits; `openzee.com` 302-redirects to a parked domain
listing). Top hypothesis: 65 % synthetic/hallucinated placeholder. Smoke test of the
starter template: 64-row synthetic run, loss 1.456 → 1.276, accuracy 0.500 (1 epoch, CPU).

---

## 5. Data acquisition story

**What was 403-blocked.** `kaggle competitions download` returned **HTTP 403 (rules not
accepted)** for 5 competitions, and `kaggle competitions data download-all/<slug>` returned
the same before acceptance. The Kaggle API endpoints for joining / accepting rules are
**not present in CLI 1.7.4.5** (`experiments/worklog.md:13`), and the API probe in
`_gpu_probe/kaggle_rules_result.json` literally records `[{"status":"login_failed"}]`.

**How it was unblocked.** Two independent routes:
1. **Playwright + system Chrome 154** (the `agent-browser` npm symlink is broken — F7)
   drove the rules-acceptance page for all 6 competitions. Verified afterwards by
   `curl -s -o /dev/null -w '%{http_code}' …/api/v1/competitions/data/download-all/<slug>`
   → **HTTP 302** (redirect to a signed GCS bundle). Full evidence in `ACCESS_STATUS.md`.
2. **Kaggle kernel competition-source mounts** bypass the download path entirely: a CPU
   kernel mounts `/kaggle/input/competitions/<slug>/` and copies what it needs into
   `/kaggle/working`, which is then pulled with `kaggle kernels output`. Used for:
   - **NFL 2027** via `nfl2027-data-extract` → `01-.../data/kaggle_nfl/nfl_data/`:
     `combine_tracking.csv` 83.8 MB, `game_tracking_2023.csv` 335 MB,
     `player_play.csv` 147 MB, `players.csv`, `games.csv`, `combine_results.csv`,
     `player_career_successes.csv` (pulled 02:10–02:12 IST).
   - **Gemma-4 paper** via `gemma4-data-extract-cpu` → `02-.../data/gemma4_data/`.
   - **ARC-AGI-2** via `arc-agi2-llm-ttc` (mount confirmed in its kernel log) *and* a
     direct download into `04-.../data/competition/` (09:10 IST):
     `arc-agi_training_challenges.json` 4.01 MB, `arc-agi_evaluation_challenges.json`
     984 KB, `arc-agi_evaluation_solutions.json` 223 KB, `arc-agi_test_challenges.json`
     1.02 MB, `arc-agi_training_solutions.json` 658 KB, `sample_submission.json` 19,936 B.
3. **Public mirrors** (all downloaded to `05-.../data/`, used as *training-only*
   augmentation with hash-level leakage control):
   `name-that-llm-response-corpus` (394 MB zip), `ntl-qwen3-8b-lora-claude-head`
   (295 MB), `prvsiyans-extended-name-that-llm-dataset` (17 MB),
   `ntl-extra-api-passages` (1.6 MB). Compacted into `data/compact/{external.parquet
   217 MB, train_all_v2.parquet 55 MB, passages.parquet, labels.json}`.
4. **Direct downloads that just worked:** T06 `playground-series-s6e10.zip` 15.8 MB
   (train.csv 67.2 MB, test.csv 27.2 MB, sample_submission.csv 8.1 MB);
   T05 `name-that-llm.zip` 874 KB (train.csv 2.1 MB, test.csv 2.2 KB — **only 3 public
   test rows**).

**What remains blocked / absent:**
- The **real Name-That-LLM private test set** — only the 3-row public `test.csv` exists
  locally; `submission.parquet` is therefore a *pipeline validation*, not a scored
  prediction (stated openly in `SUBMISSION_NOTES.md` §6).
- **ARC-AGI-2 2026 test-challenge solutions** are private; only the 120 evaluation tasks
  have public solutions for honest exact-match measurement.
- **T08** has no dataset at all (competition does not exist).
- `_gpu_probe/bulk_download.log` shows the multi-competition download loop stopped after
  the first competition (`arc-prize-2026-arc-agi-2`, 487 KB) — the remaining folders were
  populated by other means, so that script is not a reliable one-shot path.

---

## 6. Deliverables checklist (ready for the user to review / submit manually)

Nothing has been submitted. All paths are relative to `/home/pope/Desktop/gpu /hackathons/`.

**T07 — OpenCV (deadline 26 Oct)** — *most urgent complete package*
- [x] `07-.../outputs/DEVPOST_SUBMISSION.md` (15,879 B) — full rubric-mapped entry
- [x] `07-.../outputs/benchmarks.json` (25,007 B, sha256 `fb55acba09d2ca84`) + `benchmarks.csv`
- [x] `07-.../outputs/demo_annotated.mp4` (1,929,361 B, 288 frames) + `outputs/telemetry/demo.jsonl`
- [x] `07-.../outputs/cool_evidence.json` (schema `vigil.cool_evidence/2.0`) + `cool_local.json`, `arch_probe_{x86_64,aarch64}.json`
- [x] `07-.../outputs/architecture_diagram.png` (440,302 B), `outputs/demo_notes.md`, `outputs/demo_summary.json`
- [x] `07-.../src/generate_demo_video.py`, `07-.../benchmarks/run_eval.py` (33,332 B)
- [x] `07-.../aws/` (CloudFormation, ECS task defs x86+arm64, Dockerfiles, `lambda_handler.py`), `07-.../Makefile`, `07-.../scripts/run_all.sh`
- [x] `07-.../tests/` — 61 tests

**T06 — Airline S6E10 (deadline 31 Oct)**
- [x] `06-.../outputs/cv_scores.json` — best OOF AUC 0.959272
- [x] `06-.../outputs/oof_{lgbm,cat,xgb,nnprior,realmlp,blend,blend_v2}.npy`
- [x] `06-.../outputs/cv_stack_nn_lr.json` (rejection evidence)
- [ ] `06-.../outputs/submission.csv` — **exists but WRONG FORMAT** (see R1)

**T04 — ARC-AGI-2 (deadline 2 Nov)**
- [x] `04-.../outputs/submission.json` (981,178 B) and `submission_competition.json` (300,032 B)
- [x] `04-.../outputs/eval_metrics*.json` (4 files), `outputs/ranker_rows_training.jsonl`
- [x] `04-.../notebooks/arc_agi_2_solver.ipynb`, `04-.../outputs/README_TRACK04.md`
- [ ] v2 evaluation-split metrics (run pending)
- [ ] `04-.../outputs/pro_artifacts/` — empty; the Colab Pro TTC run returned nothing

**T03 — ARC Paper Track (deadline 9 Nov)**
- [x] `03-.../outputs/WRITEUP.md` (9,789 B), `outputs/paper.tex`, `outputs/paper.pdf`, `outputs/paper.log`
- [x] `03-.../outputs/cover_image.png` (452,426 B), `outputs/CITATIONS.md`, `outputs/ablation_results.csv`

**T02 — Gemma-4 Paper Track (deadline 12 Nov)**
- [x] `02-.../outputs/PAPER_WRITEUP.md` (12,381 B), `outputs/paper.tex` (7,245 B), `outputs/paper.pdf`
- [x] `02-.../outputs/eval_metrics.json`, `outputs/ablation_results.csv` (12 configurations)
- [x] `02-.../src/gg_spv_agent.py` (56,792 B), `src/agent.yaml` (724 B), `src/run_benchmark.py`, `src/eval_config.yaml`

**T01 — NFL BDB 2027 (deadline 6 Jan)**
- [x] `01-.../outputs/WRITEUP.md` (1,876 words), `outputs/INSIGHTS.md`, `outputs/DATA_SCHEMA.md`
- [x] `01-.../outputs/model_metrics.json`, `outputs/translation_metrics.json`, `outputs/translation_attribution.json`, `outputs/feature_target_correlations.csv`, `outputs/kce_agg*.csv`
- [x] `01-.../outputs/figures/fig1_velocity_cross_product.png` … `fig8_position` (8 figures)
- [x] `01-.../notebooks/nfl_combine_to_game_kce.ipynb` (4,823 B, 8/8 cells executed)

**T05 — Name That LLM (deadline 14 Jan)**
- [x] `05-.../outputs/submission.parquet` (1,758 B, 3 rows, schema-validated)
- [x] `05-.../outputs/cv_stack_report.json` (headline 0.40863), `outputs/final_model_stacked.joblib`, `outputs/final_model_config.json`
- [x] `05-.../outputs/cv_report.json`, `outputs/blend_search.json`, `outputs/cv_lenprior_report.json` (retraction evidence)
- [x] `05-.../outputs/SUBMISSION_NOTES.md` (5,100 B) — read this first; it flags 4 known risks
- [x] `05-.../notebooks/inference_submission.ipynb` (row-count agnostic, `NTL_TEST_CSV=` aware)

**T08 — OpenZee (no deadline)**
- [x] `08-.../outputs/RECON.md` (3,217 B), `outputs/STARTER_TEMPLATE.md`, `src/{dataset,model,trainer}.py`

**Top-level status artefacts**
- `MASTER_STATUS.md` (live generator output — **stale, 01:08 UTC**), `gen_master_status.py`
- `ACCESS_STATUS.md`, `COMPETITION_ACCESS.md`, `AUTORESEARCH_DIRECTIVE.md`
- `autoresearch_research.md`, `autoresearch_research.jsonl`, `autoresearch_research_strategy_graph.jsonl`, `autoresearch_research.ideas.md`, `strategies.md`, `equations.md`, `experiments/worklog.md`, `.autoresearch-loop.log` (149,909 B)

---

## 7. Known risks, open blockers, and next actions

### Risks (ordered by severity)

**R1 — T06 `submission.csv` will be scored badly if uploaded as-is.** The file contains
hard `0`/`1` integers (`{0: 169578, 1: 130266}`, 299,844 rows) but
`data/sample_submission.csv` ships a **probability** (`699635,0.44357272006117476`) and the
metric is ROC-AUC, which is defined on score ordering. Thresholded labels destroy the
ranking and can only ever reach the balanced-accuracy-equivalent ceiling.
**Do not submit it.** Regenerate from the blend probabilities first.

**R2 — T04 has no v2 evaluation-split number.** The 0/120 evaluation result is a **v1**
artifact from 01:24 UTC. The v2 solver's only evaluation-side evidence is absent, so the
headline "improved" claim rests on a 150-task training subset.

**R3 — Two billable Colab Pro sessions are alive with no artifacts**
(`proarc-1792640682` keep-alive PID 3302327; `track05-1792636477` exec PID 3220251).
The Pro re-auth is parked on a FIFO nobody has fed. Every hour they sit costs compute units.

**R4 — T05 headline is a grouped-CV estimate on 2,699 competition rows with a 3-row
public test set.** `SUBMISSION_NOTES.md` §6 already warns: all three public rows predict the
same label at max blended probability 0.071–0.102. Any private-test number will differ.

**R5 — MASTER_STATUS.md is stale (01:08 UTC).** It reports T06 as `-` for score although
`cv_scores.json` now exists, reports the T04 row from v1, and its "best score" is simply
the *last line* of `experiments.tsv` — which for T05 currently points at a **retracted**
0.51191. Do not quote `MASTER_STATUS.md` for T05 without reading the last three TSV rows.

**R6 — T01's headline R² 0.6561 is a proxy and is partly algebraic** (`cushion` is both
feature and target term). Use 0.1074 for the honest "route-created separation" claim.

**R7 — T02's patch pass rates are SIMULATED** (fidelity knob f ∈ {0.3…1.0}); Gemma-4
weights were unavailable offline. Only localization and verifier numbers are real.

**R8 — T07 FPS is load-caveated** (load average 47.26 at benchmark time). Quote the
range 20.22–44.38, not a single figure.

**R9 — Kaggle GPU slot exhaustion risk.** 8.85 h / 45 h used, one kernel still RUNNING;
the 2-slot limit means any new GPU kernel will queue.

### Open blockers
- B1: Colab Pro OAuth code must be pasted into `_gpu_probe/colab_jobs/auth/code.fifo` (Pro lane is otherwise unusable).
- B2: `kaggle/arc-agi2-llm-ttc` needs the `_subprocess` → `subprocess` name fix before re-push.
- B3: T05 Colab Qwen LoRA job needs `pad_token_id` set (or `batch_size=1`) and torchao ≥ 0.16.0 — currently failing on both.
- B4: Colab uploads of the ARC JSONs returned HTTP 500 three times; needs a retry-with-delay or a base64-into-script alternative.

### Next actions, ordered by deadline
1. **Now (T06, 31 Oct):** finish the strict cross-fitted LR sweep (`src/stack_strict_v2.py`, PID 3309399), pick the best *cross-fitted* config, and **rewrite `outputs/submission.csv` with probability values in the sample_submission order**. Also pull `ps6e10-stack` outputs (COMPLETE, download timed out locally).
2. **Today (T07, 26 Oct):** nothing blocking — the package is complete. Optionally re-run `scripts/run_all.sh` on an idle machine to get FPS without the load-47 caveat, then upload to Devpost manually.
3. **Today (T04, 2 Nov):** let the in-flight v2 training rerun finish, then run
   `python3 -m src.run_v2 --split evaluation --budget 16 --workers 12 --ranker outputs/ranker_model.joblib`
   to produce a real v2 evaluation-split number; fix `README_TRACK04.md`'s stale pointer to `eval_metrics.json`.
4. **Tomorrow (T04 Colab lane):** fix `_subprocess` in `kaggle_ttc/kaggle_arc_llm_ttc.py` and re-push `arc-agi2-llm-ttc`; alternatively re-run `run_pro_arc.sh` once B1 and B4 are cleared, and `mkdir` the remote `/content/arc/` before uploading.
5. **This week (T03, 9 Nov / T02, 12 Nov):** both are paper-complete. Remaining work is
   judge-facing polish: re-check the 5 citations in `03-.../outputs/CITATIONS.md`, and for
   T02 decide whether to attach the code+artifacts as a Kaggle dataset (idea I8, "Best New Resource").
6. **When the DeBERTa kernel completes:** fold its OOF probabilities into the T05 C=8 meta-stacker as a 4th member (idea I2 / node n9) — do **not** re-open the length-prior idea, which is closed by the decoupled test.
7. **Low priority:** T01 (6 Jan) and T05 (14 Jan) have runway; T08 should be formally dropped to free orchestration slots.

---

## 8. Appendix — exact commands

### (a) Regenerate the live status
```bash
cd "/home/pope/Desktop/gpu /hackathons"
python3 gen_master_status.py                      # rewrites MASTER_STATUS.md
nvidia-smi                                        # local lane
kaggle kernels status krishnakumarbhat/ntl-stage2-deberta
kaggle kernels status krishnakumarbhat/ps6e10-stack
kaggle kernels status krishnakumarbhat/arc-agi2-llm-ttc
kaggle kernels list --user krishnakumarbhat --page-size 25
colab --config ~/.config/colab-cli/sessions.pro.json sessions
colab --config ~/.config/colab-cli/sessions.old.json sessions
python3 - <<'PY'
import json,base64,urllib.request
tok=json.load(open("/home/pope/.kaggle/kaggle.json"))["key"]
b=base64.b64encode(("krishnakumarbhat:"+tok).encode()).decode()
r=urllib.request.Request("https://www.kaggle.com/api/v1/kernels/quota",
                         headers={"Authorization":"Basic "+b})
print(json.load(urllib.request.urlopen(r))["gpuQuota"])
PY
```

### (b) Re-run each track's pipeline
```bash
H="/home/pope/Desktop/gpu /hackathons"

# T01 NFL
cd "$H/01-nfl-big-data-bowl-2027" && python3 src/run_analysis.py && python3 src/render_figures.py

# T02 Gemma-4 GG-SPV
cd "$H/02-gemma-4-developer-agent-paper" && python3 src/run_benchmark.py

# T03 ARC paper (imports Track 04 code, read-only)
cd "$H/03-arc-prize-2026-paper-track" && python3 src/ablations.py && python3 src/figures_gen.py && python3 src/cover_gen.py

# T04 ARC-AGI-2
cd "$H/04-arc-prize-2026-arc-agi-2"
python3 -m src.selfcheck
python3 -m src.run_v2 --split training  --budget 16 --workers 12 --collect --output-dir outputs
python3 -m src.run_v2 --train-ranker --dataset outputs/ranker_rows_training.jsonl
python3 -m src.run_v2 --split evaluation --budget 16 --workers 12 --ranker outputs/ranker_model.joblib
python3 notebooks/build_notebook_v2.py

# T05 Name That LLM (CPU, ~20 min)
cd "$H/05-name-that-llm"
python3 src/run_real_experiments.py      # stage-1 grid  -> outputs/oof_real.npz
python3 src/run_real_stage2.py           # C grid        -> oof_real_stage2.npz
python3 src/blend_search.py              # blend grid + nested estimate
python3 src/finalize_real.py             # refit + artefacts + submission
python3 src/report_real.py               # rebuild cv_report.json
python3 src/stack_probe.py               # the KEEP: stack_probs_c8 = 0.40863

# T06 Airline S6E10 (RTX 3050 lane, <3.2 GB VRAM)
cd "$H/06-playground-s6e10-airline-satisfaction"
python3 src/realmlp_stack.py             # RealMLP + kNN-prior members
python3 src/stack_nn_lr.py               # strict LR rejection evidence
python3 src/test_predict.py              # writes outputs/submission.csv  <-- FIX PROBABILITY FORMAT

# T07 OpenCV
cd "$H/07-opencv-ai-competition-2026"
bash scripts/run_all.sh                  # or: make all
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider   # 61 tests

# T08 OpenZee (smoke test only)
cd "$H/08-openzee-openair-2026" && python3 -c "import src.dataset, src.model, src.trainer"

# Remote lanes via the dispatcher (KAGGLE_KERNEL_SLUG selects the kernel)
cd "/home/pope/Desktop/gpu /_gpu_probe"
GPU_BACKEND=local  ./run_gpu.sh <script.py>
GPU_BACKEND=kaggle KAGGLE_KERNEL_SLUG=krishnakumarbhat/<slug> ./run_gpu.sh <script.py>
GPU_BACKEND=colab  COLAB_ACCOUNT=pro COLAB_GPU_TYPE=T4 COLAB_UPLOADS="a:/content/a" \
                   COLAB_DOWNLOADS="/content/out.zip:outputs/" ./run_gpu.sh <script.py>
```

### (c) Submit manually — only when the user decides (nothing here has been run)
```bash
H="/home/pope/Desktop/gpu /hackathons"

# Kaggle — competition submissions
kaggle competitions submit -c name-that-llm \
    -f "$H/05-name-that-llm/outputs/submission.parquet" -m "stack_probs_c8 (bal_acc 0.40863 SGKF5)"
kaggle competitions submit -c playground-series-s6e10 \
    -f "$H/06-playground-s6e10-airline-satisfaction/outputs/submission.csv" -m "OOF AUC 0.959272"
kaggle competitions submit -c arc-prize-2026-arc-agi-2 \
    -f "$H/04-arc-prize-2026-arc-agi-2/outputs/submission_competition.json" -m "v2 object-beam solver"
kaggle competitions submissions -c name-that-llm          # verify it landed

# Paper tracks (ARC Paper 03, Gemma-4 Paper 02) — upload the PDF + writeup via the
# competition's paper-submission form on kaggle.com; the CLI has no paper endpoint.
#   $H/03-arc-prize-2026-paper-track/outputs/{paper.pdf,WRITEUP.md,CITATIONS.md,cover_image.png}
#   $H/02-gemma-4-developer-agent-paper/outputs/{paper.pdf,PAPER_WRITEUP.md,ablation_results.csv}

# Devpost (T07) — browser upload; agent-browser's symlink is broken, so use Playwright
# + system Chrome 154 (the exact route used for rules acceptance):
#   $H/07-opencv-ai-competition-2026/outputs/DEVPOST_SUBMISSION.md  (+ demo video, images, repo link)
```

---

### Provenance note
Every number above was read from a file under `/home/pope/Desktop/gpu /hackathons/`
(`outputs/*.json`, `outputs/*.csv`, `logs/experiments.tsv`, `logs/*.log`, the writeups, or
the scripts named in §3.5) or produced by a command run during this session (`nvidia-smi`,
`kaggle kernels status/list`, the Kaggle quota REST endpoint, `colab … sessions`).
Three items are explicitly flagged **unverified** because the originating error text is not
preserved in any file on disk: **F4** (SimpleImputer `categorical_strategy`) and **F5**
(fp16 `GradScaler.unscale_`) — the shipped workarounds are visible in source, the failures
are not; and the **0.959607** T06 strict-stack figure, which is a live in-flight log
(`/tmp/opencode/t06_strict.log`) rather than a committed artefact.
