# Track 04 — ARC-AGI-2 v2: object-centric induction, beam search, learned ranking

Offline solver for ARC-AGI-2 (public training/evaluation splits shipped in
`data/arc-agi-2/`). No network calls at run time, no Kaggle submission.
Everything below is produced by executing the code in `src/`; every number is
measured on this machine and stored in `outputs/*.json` and
`logs/experiments.tsv`.

## What changed from v1

v1 returned the *first* hypothesis that reproduced every training pair. Its
per-method precision on the training split was the problem:

| v1 method | accepted | correct | precision |
| --- | ---: | ---: | ---: |
| depth1 | 34 | 33 | 0.97 |
| depth2 | 12 | 11 | 0.92 |
| cell_tree | 205 | 31 | 0.15 |
| patch_lookup | 46 | 6 | 0.13 |
| object_tree | 133 | 4 | 0.03 |
| mlp | 1 | 1 | 1.00 |

Because the low-precision learners were tried last but *accepted*, they produced
most of the 353 "solved" tasks and almost none of the 67 correct ones, and the
evaluation split scored 0/120. v2 changes the shape of the problem:

1. **Object-centric induction** (`src/object_model.py`,
   `src/object_program_inducer.py`, `src/structural_families.py`). A grid is
   segmented into connected components carrying size, canonical D4 shape,
   colour set, bbox, border contact, hole count, symmetry and centroid. Programs
   are then fitted over *objects*: selection + crop, attribute-consistent
   recolouring, shared translation, shared duplication, predicate deletion,
   hole filling, outlining, template stamping, line connection, panel logic
   (select the odd panel, or a learned element-wise table over panel values),
   ray casting, window cropping, nearest-neighbour rescaling and denoising.
2. **Heuristic beam search** (`src/beam_search.py`) replaces exhaustive depth-2
   enumeration. States are scored by shape penalty + colour-histogram L1 +
   object-count error + symmetry error; the beam keeps 8 states per depth and
   searches to depth 3 with a 144-operator vocabulary (global D4/tiling, object
   selection/deletion/cropping, gravity, rays, interpolation, fill-enclosed).
   Verified hits are re-checked on every pair before becoming candidates.
3. **Learned ranker** (`src/program_ranker.py`). Candidates are described by
   gold-free features (family, complexity, leave-one-out evidence, output shape
   statistics, agreement between families, task descriptors). A model is fitted
   on the training split with a task-level 70/30 split; `attempt_1` is the top
   ranked candidate, `attempt_2` the top ranked *distinct* one.
4. **Leave-one-out evidence** for every learned rule: fit on `n-1` pairs,
   predict the held-out pair. It is a feature for the ranker rather than a hard
   gate, because strict gating collapsed recall from 4.5% to 0.5% in a measured
   ablation.
5. **Correct pass@2 and v1 bug fixes.** v1 reported only *attempt_2* in the
   pass@2 slot, so `graded_pass2` could be smaller than `graded_pass1`. Both
   `cell_rule_learner` and `object_rule_learner` also referenced an undefined
   `_D4_INVERSE`, which silently disabled every D4-augmented variant
   (`object_tree_d4_error:NameError` in the v1 notes).

## Measured results

Filled in by the final runs:

* `outputs/eval_metrics.json` — evaluation split (120 tasks), v2.
* `outputs/eval_metrics_training_v2.json` — training split (1000 tasks), v2.
* `outputs/ranker_metrics.json` — ranker fit + task-level validation.
* `outputs/submission.json` — `attempt_1` / `attempt_2` for every task id.
* `logs/experiments.tsv` — one row per measured metric.

## Files

| Path | Role |
| --- | --- |
| `src/object_model.py` | segmentation, object attributes, selection predicates |
| `src/object_program_inducer.py` | object-level program families + LOO |
| `src/structural_families.py` | panels, rays, windows, resize, denoise |
| `src/beam_search.py` | heuristic beam search over 144 operators |
| `src/program_ranker.py` | candidate features + learned ranking model |
| `src/solver_v2.py` | cascade, budgets, candidate pool, attempts |
| `src/run_v2.py` | parallel runner, metrics, submission, dataset, ranker training |
| `src/program.py` | shared `Program` type and colour-map helpers |
| `src/search.py` | v1 cascade (retained: depth-1/2 primitives, fallback) |
| `src/dsl_primitives.py` | grid primitive DSL (unchanged) |
| `src/canonical.py` | D4 + colour canonicalisation (unchanged) |
| `src/cell_rule_learner.py` | per-cell decision tree (`_D4_INVERSE` fixed) |
| `src/object_rule_learner.py` | per-component tree (`_D4_INVERSE` fixed) |
| `src/patch_lookup_learner.py` | local patch lookup with abstention |
| `src/selfcheck.py` | 16 synthetic tasks across v1 + v2 families |
| `notebooks/arc_agi_2_solver.ipynb` | executed reproduction notebook |
| `notebooks/build_notebook_v2.py` | executes the notebook (no Jupyter installed) |

## Reproduction

```bash
python3 -m src.selfcheck
python3 -m src.run_v2 --split training --budget 16 --workers 12 --collect \
    --output-dir outputs
python3 -m src.run_v2 --train-ranker --dataset outputs/ranker_rows_training.jsonl
python3 -m src.run_v2 --split evaluation --budget 16 --workers 12 \
    --ranker outputs/ranker_model.joblib
python3 notebooks/build_notebook_v2.py
```

## Data note

`data/comp_arc-prize-2026-arc-agi-2/` is **not present** on this machine; there
is no competition download, no `sample_submission.json`, and the 2026 test set
is not public. The submission is therefore built over the real task ids of the
shipped ARC-AGI-2 **evaluation** split (120 ids), using the competition schema
(`attempt_1`, `attempt_2` per task id, with `attempt_{slot}_{i}` extras for
tasks that have more than one test input).

## Limitations

* ARC-AGI-2 evaluation grids are larger and more compositional than training
  (median 18x19 versus 10x10, 5.2 versus 3.8 colours). Absolute accuracy stays
  low; the interesting quantity is the precision/recall trade-off.
* Candidate coverage, not ranking, is the binding constraint: on the training
  split the manual prior already selects a correct candidate whenever one
  exists, so the ranker cannot lift pass@1 above the candidate oracle.
* Panel and ray families cover single-step rules only; multi-step composition
  still exceeds the 16 s per-task budget.
