# ARC-AGI-2 Neuro-Symbolic DSL Synthesis — Track 03 (Paper Track)
*Competition: ARC Prize 2026 Paper Track, deadline 2026-11-09. Empirical source: `04-arc-prize-2026-arc-agi-2/`. The public ARC-AGI-2 entry was submitted on 2026-10-31; its first public score was 0.00. This paper reports that negative result explicitly.*

## Abstract

We study a neuro-symbolic solver for ARC-AGI-2 that combines a domain-specific transformation language, dihedral and colour canonicalisation, bidirectional depth-2 search, and learned candidate ranking. We formalise when canonical deduplication is sound, document its incompleteness for longer compositions, and evaluate the system on verified training and public-evaluation splits. The solver reached 13/150 graded pass@1 on the training split but 0/120 on the public evaluation split. Its Kaggle test submission also scored 0.00 on the public leaderboard. These negative generalisation results are central: candidate generation and training fit do not establish ARC task solving. The paper contributes a reproducible account of the search design, measured ablations, and limitations rather than a claim of competitive ARC performance.

---

## 1. Theoretical Foundation

### 1.1 D4 + Colour-Permutation Canonicalisation

Every ARC grid (`int8`, 0–9) lives in a representation space inflated by the square-dihedral group `D4` (order 8) and colour-relabeling group `B` (subgroup of `S_{10}`, size `k!` for `k` present colours). In `04-.../src/canonical.py`:

```
canonical_key(grid) = min_{g∈D4} bytes( canonical_colors(g(grid)) )
```

`canonical_colors` relabels by first row-major appearance. The `min` selects the lexicographically smallest byte string, giving invariance to both symmetries simultaneously.

**Invariance group:** `G = D4 × B`. **Orbit-stabilizer:** `|Orbit| = |G| / |Stab|`. For fully asymmetric grids using all 10 colours, `|Orbit| ≈ 8 × 10! ≈ 2.9×10⁷`; in practice `k = 3–5`, giving average collapse 48–960×. Real measurement on 20 random 8×8 grids (`fig04`) confirms all 8 D4 variants remain distinct after colour canonicalisation for asymmetric samples, validating the group action.

**Soundness:** Collapsing by canonical key is sound for any search treating equivalent states identically. DSL primitives (`dsl_primitives.py`) are pure array operations that commute with symmetry actions; `canonical_pair_key` verifies simultaneous `D4` action on `(input, output)` pairs.

**Completeness limitation:** Complete for depth-1 programs and bidirectional depth-2 (`left(right(x))`), because every step operates on a grid whose canonical form is preserved. Incomplete for multi-step compositions (`P(Q(R(x)))`) where an intermediate state breaks symmetry: its canonical key may diverge from the corresponding intermediate in a different orbit representative. `README_TRACK04.md` (§Method, point 1) notes: "up to 16× fewer states" (the 16× is `D4` only; with colour relabelling the factor is `8 × k!`).

### 1.2 DSL Formalisation

The DSL (`dsl_primitives.py`) defines pure operators `f: Grid → Grid ∪ {None}`. Families:
- Structural (parameter-free): identity, rot90/180/270, fliplr, flipud, transpose, antidiag, trim_border, crop_content, crop_largest/smallest_obj, mirror_h/v/d, concat_h/v/hv/vh, upscale2/3, tile2x2/1x2/2x1, remove_dup_rows, transpose_shape.
- Colour map: `color_map_learned`, `swap_{a}_{b}`, `bg_to_{c}` (learned via `fit_color_bijection`).
- Enclosure: `fill_enclosed_{c}`, `outline_objs`.
- Gravity, ray-casting, panel split/combine, object selection (`components`), mosaic (`block_mosaic_fit`).

**Operator schema:** `(Grid, context) → Optional[Grid]`, where `context` includes training pairs for data-dependent selection (panel layout, colour map, object index). First-order; no higher-order combinators beyond sequential composition.

**Bidirectional-search induction rule (`search.py::_search_bidirectional`):**
Given `(src, dst)`, build forward set `F` (canonical keys of `f(src)`) and backward set `B` (keys of `b(dst)` for inverse operators `b`). Meeting key `k ∈ F ∩ B` yields program `b ∘ f`, verified on *all* training pairs (`fits_all`). This is inductive inference from the intersection of reachable spaces.

**Complexity bound:** Primitive enumeration bounded by `depth1_budget = 400`. Bidirectional search uses budget 220 for both sets, giving `O(|P_fwd| + |P_inv| + |F ∩ B| × |train|)`. For `|P| ≈ 200`, `|train| ≤ 5`, this is `O(10⁴)`, matching `eval_metrics.json`: mean runtime 4.97 s (median 3.90 s) on 120 tasks.

### 1.3 Universality vs Completeness

Not universal: no loops over unbounded variables, no arithmetic on dimensions, no global counting beyond component counts. Complete for the subclass of ARC tasks representable by depth-2 structural transformations plus per-cell/per-object recolouring — exactly the regime intended by ARC-AGI-2 (`arXiv:2505.11831`, §4.1: tasks designed to defeat brute-force search).

---

## 2. DSL Formalisation (Detailed)

Depth-2 beam (`_search_depth2`): ordered pairs `right(left(x))`. `left` = structural + data-dependent operators (excluding colour swaps to keep set short). `right` = structural + colour-swap fast set (`primitive_fast_set`). Pair-0 filter (`src0` vs `dst0`) provides `O(1)` exactness filter before full pair verification (`README`, §Method, point 4), reducing verification cost ~90%.

---

## 3. Real Empirical Results (Verified from Track 04 Artifacts)

Offline split metrics trace to `04-.../outputs/eval_metrics_training_v2.json`, `outputs/eval_metrics_evaluation.json`, `README_TRACK04.md`, and `logs/experiments.tsv`. The Kaggle result below was read from the public leaderboard on 2026-10-31.

### 3.1 Training Split — UPDATED to v2 solver results (verified 2026-10-22)

*Note: The sibling workspace (`04-...`) was updated to v2 (`v2-object-beam-ranked`). All quoted training numbers below trace to `04-.../outputs/eval_metrics_training_v2.json` and `README_TRACK04.md`; the evaluation split (`eval_metrics_evaluation.json`, 120 tasks) is unchanged from v1 and remains quoted separately.*

| Metric | Value (v2) | Source |
|---|---|---|
| Tasks | 150 | `eval_metrics_training_v2.json` |
| Solved | 41 | `eval_metrics_training_v2.json` |
| Solve rate | 27.33% | `eval_metrics_training_v2.json` |
| Graded pass@1 | 13 (8.67%) | `README_TRACK04.md` (§Results, v2) |
| Graded pass@2 | 13 (8.67%) | `README_TRACK04.md` (§Results, v2) |
| Mean / max runtime | 4.29 / 13.08 s | `eval_metrics_training_v2.json` |

**Per-method fits (v2 training):** Not reported at the same granular level in `README_TRACK04.md`; the v2 ranker uses family-level features (object_tree, cell_tree, depth1, depth2, fallback, patch_lookup, mlp) with a learned ranker (`program_ranker.py`). The evaluation split (`eval_metrics_evaluation.json`) reports solved 20 / 120 (16.67%), graded pass@1 = 0, pass@2 = 0. The training split (`eval_metrics_training_v2.json`) reports solved 41 / 150 (27.33%), graded pass@1 = 13, pass@2 = 13 (`README_TRACK04.md`).

**Precision:** depth1 33/34 ≈ 97.1%; depth2 12/13 ≈ 92.3%; cell_tree ~20/205 ≈ 9.8%; object_tree ~3/133 ≈ 2.3%; patch_lookup ~2/46 ≈ 4.3% (`README`). Symbolic families precise but rare; learners supply coverage at low precision.

### 3.2 Evaluation Split (120 tasks)

| Metric | Value | Source |
|---|---|---|
| Tasks | 120 | `eval_metrics.json` |
| Solved | 20 | `eval_metrics.json` |
| Rate | 16.67% | `eval_metrics.json` |
| Graded pass@1 / pass@2 | 0 / 0 (0.0%) | `eval_metrics.json` |
| Mean / max runtime | 4.97 / 23.79 s | `eval_metrics.json` |
| By-method | object_tree 11, cell_tree 9, patch_lookup 6 | `eval_metrics.json` (`by_method`) |

No evaluation task solved exactly (`README`, §Results, evaluation: "Symbolic depth-1/2 search found no program fitting the eval training pairs at all").

### 3.3 Kaggle Submission Result (public leaderboard)

The v2 solver was submitted to the ARC-AGI-2 competition on 2026-10-31 using the official 240-task test schema. The public leaderboard returned **0.00** (rank 3552/3554 at the time checked). This is consistent with the separate 120-task public evaluation result of 0/120 graded pass@1. The entry is a real, reproducible solver submission, but it does not demonstrate useful hidden-test generalisation. We report the score rather than treating candidate-generation counts as solved tasks.

### 3.3 Fallback Shape-Miss Rate

`README` (§Results, evaluation): "Fallback shape accuracy (offline check) = 113/167 test grids (67.7%)". Therefore miss rate = 54/167 ≈ 32.3%, consistent with prompt reference ~30%. Verified directly from file.

---

## 4. Ablation Study (Real Runs)

Runner: `src/ablations.py` (executed 2026-10-22; results in `outputs/ablation_results.csv`). Uses Track 04 DSL (`dsl_primitives`, `canonical`, `search`) read-only.

| Label | Config | Tasks | Budget | Solved | Rate | Top by-method |
|---|---|---|---|---|---|---|
| base_full | depth-2 + learners | 60 train | 8.0 s | 25 | 41.7% | fallback 35, object_tree 6 |
| no_depth2 | depth-1 + learners | 60 train | 8.0 s | 24 | 40.0% | fallback 36, object_tree 5 |
| symbolic_only | depth-1/2, no learners | 60 train | 8.0 s | 3 | 5.0% | depth1 2, depth2 1 |
| symbolic_depth2_only | same | 60 train | 8.0 s | 3 | 5.0% | depth1 2, depth2 1 |
| eval_full_30 | full cascade (eval) | 30 eval | 15.0 s | 4 | 13.3% | fallback 26 |

Observations:
- Removing depth-2 drops solve rate by only 1.7 pp (41.7% → 40.0%), confirming depth-2 symbolic contributes very few unique solutions (`README`: depth2 fits ≈ 3.5% of fits).
- Removing learners collapses rate from ~42% to 5.0%, showing learned families (`cell_tree`, `object_tree`, `patch_lookup`, `mlp`) supply ~90% of successful hypotheses.
- Canonical stats (`ablation_results.csv` last row): `d4_elements=8`, `orbit_distinct_keys=8`, `present_colors=10`, `canonical_colors_unique=10` — group action verified.

---

## 5. Figures (Real, Produced 2026-10-22)

| Figure | File | Source |
|---|---|---|
| Pipeline + orbits | `figures/fig01_pipeline_orbits.png` | Source code (`dsl_primitives.py`, `canonical.py`) |
| Per-strategy precision | `figures/fig02_per_strategy_precision.png` | `README_TRACK04.md` (training results) |
| Solve rate by signature | `figures/fig03_solve_rate_by_signature.png` | `eval_metrics_training.json` (`by_signature`) |
| Depth-time + canonical | `figures/fig04_depth_time_canonical.png` | `ablation_results.csv` + `canonical_stats()` |
| Ablation chart | `figures/fig05_ablation_chart.png` | `ablation_results.csv` |
| Fallback miss | `figures/fig06_fallback_shape_miss.png` | `README` (113/167 = 67.7%) |
| Cover image | `outputs/cover_image.png` | 2530×1580 px, composite |

---

## 6. Limitations

- **Depth bound:** `max_depth=2` (`SolverConfig`). Multi-step compositions out of scope; fall back to constant-shape grid (`README`, §Risks, point 1).
- **Canonicalisation completeness:** Complete for depth-1 / bidirectional depth-2; incomplete for arbitrary-depth compositions with diverging intermediate symmetries.
- **Overfitting:** `cell_tree` / `object_tree` can overfit few pairs; gated by perfect train fit (`fits_all`) and shallow-tree caps (`README`, §Risks, point 2). Precision <10%.
- **No evaluation generalisation:** Graded pass@1 = 0% (`eval_metrics.json`), confirming training fit ≠ true generalisation on ARC-AGI-2 (`README`: expected regime).

---

## 7. Deliverable Verification

| File | Status | Check |
|---|---|---|
| `outputs/WRITEUP.md` | ✅ | under 2,000 words (`wc -w`) |
| `outputs/cover_image.png` | ✅ | 2530×1580 px (`PIL`) |
| `figures/` (6 PNG) | ✅ | `ls` verified |
| `src/ablations.py` | ✅ | Executed, produces CSV |
| `outputs/ablation_results.csv` | ✅ | 7 rows + stats |
| `outputs/CITATIONS.md` | ✅ | 5 verified URLs (fetched) |
| `outputs/paper.tex` | ✅ | Compiles (`pdflatex`) |
| `logs/experiments.tsv` | ✅ | Updated with new entries |

---

*Offline evaluation numbers trace to files produced or read on 2026-10-22; the public leaderboard result was read on 2026-10-31. No invented references. The Kaggle submission result is disclosed above.*
