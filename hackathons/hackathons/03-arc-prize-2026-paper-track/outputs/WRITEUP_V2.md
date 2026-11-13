# Two Systems Under a 12-Hour Budget: Symbolic Search, Test-Time-Adapted LLMs, and the Coverage Ceiling

**Team `krishnak` — ARC Prize 2026 Paper Track**

## Summary

This writeup reports two end-to-end ARC-AGI-2 systems built for the competition's real
constraint: a notebook that reruns offline for at most 12 hours on four L4 GPUs and must
emit `submission.json` for all 240 hidden tasks.

| System | Approach | Public leaderboard |
|---|---|---|
| A | Neuro-symbolic DSL synthesis (D4 × colour canonicalisation, bidirectional depth-2 search, learned rankers) | **0.00** |
| B | Reproduction of a public test-time-adapted LLM pipeline (Qwen3-4B ARC-SFT, per-task TTT, D4/permuation TTA, beam/DFS decoding) | **30.56** |

System B is a reproduction, not our invention: it is a fork of the Apache-2.0 notebook
`koushikrudra/failed-in-aimo` (version 1), with the original authors fully credited. Our
contribution here is the measurement, the failure analysis, and the coverage model below.

## What actually happened

- System A's entry, submitted 2026-10-31, scored **0.00** and is the only entry counted for
  the team (rank 3599 of 3627 on the final board).
- System B's entry, submitted 2026-11-02 08:28 UTC, scored **30.56** — a 3× improvement over
  the previous entry — but completed its ~12-hour rerun after the 23:59 close and was
  therefore classified as a **late submission** and excluded from the standings. At 30.56 it
  would have ranked approximately **1,271st**; the top-200 threshold was **33.06**.

We report the exclusion because it dominated every modelling decision we made.

## System A — why sound symbolic search was not enough

The synthesiser canonicalises grids under $G = D_4 \times B$ (rotations/reflections and
colour relabelling), shrinking the search frontier by up to 48–960× for 3–5 colour grids. It
then induces programs bidirectionally: a meeting key between the forward image set and the
backward inverse image set yields a candidate $b \circ f$, verified on every training pair.
Depth-2 beam enumeration composes ordered pairs under a pair-0 exactness filter.

Measured on verified artifacts:

| Split | Tasks | Solved (grid level) | Graded pass@1 |
|---|---|---|---|
| Training (v2) | 150 | 41 (27.3%) | 13 (8.7%) |
| Public evaluation | 120 | 20 (16.7%) | **0 (0.0%)** |

The 16.7% → 0/120 collapse is the result that matters: the learned ranker (cell-tree,
object-tree, patch-lookup, MLP) fits training pairs but selects structurally plausible yet
wrong grids under exact match. Ablations confirm the system is a learned ranker over symbolic
search: removing depth-2 costs 1.7 pp, removing the learned families costs ~37 pp.

## System B — test-time adaptation works, coverage caps it

Per task, the pipeline resets LoRA adapters, trains on 16 augmented demonstrations for 128
steps, then decodes over D4 × colour-permutation augmentations with a 12-token ARC vocabulary,
teacher-forced NLL thresholding and DFS, aggregating candidates across augmentations. A
1,200 s cap bounds each task.

Validation was deliberately treated as a **smoke benchmark, not a held-out estimate**: the ARC
evaluation split is public and may be in the model's training data. Twelve tasks were chosen by
a fixed permutation seed (20261102) without inspecting solutions; the solutions file is read
only in a posthoc cell after `submission.json` is written. Result: 18 test rows, **3 pass@2**,
1 pass@1, **2 of 11 completed tasks** (18.2%, Wilson 95% interval ≈ [5%, 48%]), 0 malformed
grids, 11 of 12 prediction bundles written.

## The coverage ceiling

Leaderboard score factorises as **S = C × p**, where C is the fraction of the 240 hidden tasks
the worker pool reaches before the clock and p is the per-attempt solve probability. With
W = 4 workers, B = 710 minutes and measured per-task latencies of 219–1302 s (median 641 s,
mean 749 s, p90 1144 s over 12 tasks), the projection is **C ≈ 0.62–0.95**. Inverting the
observed S = 0.3056 gives **p ≈ 0.32–0.49**.

Consequences:

- ∂S/∂C = p, so one percentage point of coverage is worth 0.32–0.49 points.
- 30.56 corresponds to ~73 of 240 tasks solved; the top-200 threshold (33.06) is ~79 tasks.
  The gap was **about six solved tasks, or roughly 12–19 additional attempted tasks**.
- That is a scheduling problem, not a modelling problem: the same code, submitted earlier and
  with the queue delay absorbed, could have crossed the line.

## The protocol failure

Kaggle requires the notebook run to *complete* before the deadline. Our entry was submitted
15.5 hours early but sat in the GPU queue; the 12-hour rerun finished after the close and was
scored but not ranked. The operational rule: the last safe submission time is
`close − (runtime + queue delay)`, and on the final day the queue delay was hours, not minutes.
Two hours spent on final-day validation came directly out of that slack.

## Reproducibility

- System B fork: `krishnakumarbhat/arc-nvarc-validated` (Apache-2.0-derived, original authors
  credited; Docker image pinned by digest to the one recorded by the source notebook).
- Changes to the original: model-mount discovery (the hard-coded path no longer exists),
  checked subprocess launch, GPU preflight assertion, deterministic validation queue, posthoc
  scoring. Weights, training settings, augmentation schedule, decoding and aggregation are
  unchanged.
- Submission validated for all 240 tasks: rectangular integer grids in [0, 9], two attempts per
  test row, no malformed entries.
- System A artifacts: `eval_metrics*.json`, `ablation_results.csv`, `figures/`.

## Limitations

The coverage interval is extrapolated from twelve measured tasks. The per-attempt solve
probability is inferred, not observed, because hidden labels are unavailable. Public-evaluation
results may be contaminated for public models. System A is depth-2 bounded and not
competitive; it is included because its negative result motivates the pivot. System B claims no
modelling novelty.
