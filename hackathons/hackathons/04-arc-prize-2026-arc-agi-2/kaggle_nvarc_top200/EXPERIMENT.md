# ARC top-200 attempt — 2026-11-02

## Target and evidence

- Latest downloaded public-leaderboard snapshot: rank 200 = **33.06**, rank 100 = **40.56**.
- Our existing entry: **0.00**, rank **3586** in that snapshot.
- Candidate: `koushikrudra/failed-in-aimo/1`, public score **33.89** confirmed on the version-one page (script version 312063593).
- Both the notebook and `sorokin/qwen3_4b_grids15_sft139` model variation are published under Apache-2.0.
- The notebook titled `ARC AGI2 LB33.89 Minimal Perfpatch` actually shows **32.22**; its title was not accepted as score evidence.

## Implementation provenance

The initial subagent draft accidentally used the materially changed latest
notebook, which includes an additional induction leg. The parent explicitly
downloaded `/1` using Kaggle CLI 2.2.5 and rebuilt this fork from the real V1.
The loader, decoder, solver/training code and launch command are byte-identical
to V1. Only the local validation queue, time budget, label-dependent debug
benchmark removal and posthoc reporting differ. Notebook outputs were cleared.

The fixed validation selection uses seed 20261102, sampling 12 task IDs without
looking at their solutions. During the hidden competition rerun, every supplied
test task is queued, with the original 11h50m solver budget. All evaluation
solutions are loaded only in the final, validation-only reporting branch.

This is a **smoke benchmark, not an unbiased heldout estimate**: public evaluation
tasks may be included in the public model's training. Neither that benchmark nor
the source notebook's score guarantees a reproduced leaderboard rank.

## Runtime and local tests

- Private kernel: `krishnakumarbhat/arc-nvarc-validated`, version **2**.
- Offline, competition-attached, `NvidiaL4` (4×L4).
- Original Docker image pinned by immutable digest in `kernel-metadata.json`.
- CLI 1.7.4.5 silently ignores the GPU-shape field. Runs were pushed using CLI
  2.2.5 installed in `/tmp/opencode/kaggle225`, with the existing Python 3.13.
- Local checks passed: Python AST, original core-source equality, rectangular
  grids, invalid colors/types/dimensions, multi-test row weighting, second-attempt
  exact matches, and zero-solve scoring.
- V1 was initially pushed with a short session timeout; V2 corrects the notebook
  session ceiling to 12h while retaining the bounded ~2h local-validation budget.

## Submission gate

Validation has been launched; **no leaderboard submission has been attempted**.
The watcher downloads terminal validation artifacts to
`/tmp/opencode/arc_nvarc_validation_v2` and never submits automatically.
Review `validation_result.json`, completed-task coverage and worker logs before
deciding whether the one daily submission slot is justified.

## V2 outcome and repair

V2 failed before any inference. The pinned environment imported Unsloth on
Python 3.11 successfully, but the original hard-coded model directory did not
exist under Kaggle's current mount namespace. The primary error was
`Unsloth: No config file found`; the subsequent missing `inference_outputs`
directory was a consequence, not the underlying cause. No validation accuracy
or leaderboard result exists for this run, and the daily slot remains unused.

The implementation worker is adding mounted-model configuration discovery and
checked subprocess execution. Model-loading path resolution and error handling
will differ from V1; the learned weights, training settings and inference/
candidate-selection algorithms remain the original baseline.

## V3 validation result (2026-11-02 08:25 UTC)

The run completed and produced a well-formed 12-task submission.

| Metric | Value |
|---|---|
| Selected benchmark tasks | 12 (fixed seed 20261102, 18 test rows) |
| Rows solved pass@1 | 1 / 18 |
| Rows solved pass@2 | 3 / 18 (16.7%) |
| Task-normalized pass@2 | 0.194 |
| Tasks fully solved | 2 / 11 completed (18.2%) |
| Tasks with complete prediction bundles | 11 / 12 |
| Invalid submission entries | 0 |
| Measured per-task wall time | 284s / 679s / 886s / 1144s / 1302s (one hard timeout) |

With n=11 the 95% interval around 18% still contains the source notebook's
33.89, so this is not evidence the reproduction is broken — but it is also not
evidence that it reaches 33.89. Note the in-solver per-task timeout of 1302s;
measured durations imply roughly 190–220 of 240 hidden tasks would be reached
inside the 11h50m budget.

## Submission

- Submitted kernel version: `krishnakumarbhat/arc-nvarc-validated` **v3**.
- Kaggle submission ref: **57901718**, submitted 2026-11-02 08:28:28 UTC.
- The competition allows **1 submission per day**; the account now shows
  `0 submissions remaining today`, so this is the final ARC entry.
- A background watcher (`/tmp/opencode/arc_watch_submission.py`) polls the score
  and writes `/tmp/opencode/arc_submission_57901718.json`.

### Why submit despite the ambiguous benchmark

The previous best was 0.00 (rank 3586/3600), the slot would otherwise be unused,
and the submission is the only path to a positive score. The rerun uses the same
verified code, writes `submission.json` for all 240 test tasks, and cannot lower
the recorded best score (Kaggle keeps the maximum).

