## Read the final paper and executable audit

**Current, authoritative version: `paper_final.pdf` (6 November 2026).** The attached `arc_audit_final.zip` contains the reproducible audit and competition-only evidence. Earlier attachments (`paper.pdf` and `paper_v2.pdf`), if still listed, are superseded. This revision corrects their metric interpretation and unsupported coverage claims.

## Contribution: diagnose false progress before optimising it

We audit two ARC-AGI-2 systems: our symbolic/learned candidate search, whose counted entry scored **0.00**, and a reproduction of `koushikrudra/failed-in-aimo` v1, whose **late, unranked submission received 30.56**. We claim no new model, test-time-training method, state-of-the-art accuracy, or official rank for the reproduction.

The contribution is an **executable diagnostic study** covering metric semantics, canonical-state correctness, and deadline-constrained throughput. It supplies concrete counterexamples, a proved and checked key repair, and a conditional scheduling experiment rather than speculation about hidden-task performance.

### 1. “Solved” meant found a candidate—not solved the unseen test

The audited source assigns `solved=bool(hypotheses)` or `solved=bool(candidates)`. It records the existence of a demonstration-consistent candidate, not its correctness on unseen outputs.

| Historical run | Tasks | Candidate discoveries | Graded pass@1 | Graded pass@2 |
|---|---:|---:|---:|---:|
| Training v2 | 150 | 41 (27.3%) | 13 (8.7%) | 13 (8.7%) |
| Evaluation cascade | 120 | 20 (16.7%) | 0 | 0 |

These are distinct historical solver versions, not a matched experiment. The earlier “16.7% solved evaluation grids” claim is withdrawn. The 60-task ablation likewise measures candidate discovery: 25 full-cascade candidates, 24 without depth-2, and 3 symbolic-only. That is not a 36.7-point test-accuracy gain.

### 2. Two canonicalisation defects, with an executable repair

The existing key normalises colours **before** trying the eight spatial transforms and omits grid dimensions. We verified:

- **Missed equivalence:** `[[0,0,0],[0,0,1]]` and its 180-degree rotation `[[1,0,0],[0,0,0]]` return different original keys.
- **False equivalence:** all-zero 1×6 and 2×3 grids return identical original keys, although no dihedral transform relates their shapes.

The repaired key minimises `(height, width, canonical-colour bytes)` **after each** dihedral transform. The paper proves its invariance and orbit-equivalence property. The implementation passes **34,992 exhaustive assertions**: all 729 three-colour 2×3 grids × 8 spatial transforms × 6 colour permutations, plus separate shape-collision and scheduling boundary checks.

A canonical key alone still does not justify search pruning. Orientation- or colour-sensitive primitives need symmetry witnesses or compatible equivariant transitions. A backward operator also needs the correct inverse direction when constructing a forward program. Demonstration verification can reject a bad proposal; it cannot restore a discarded continuation or certify unseen correctness.

**This repair is an audit result, not a newly benchmarked solver improvement.** It was not used in the closed main-track submission.

### 3. Measured latency and explicitly conditional simulation

We deduplicate repeated log streams by task ID and verify the repeated timings agree. Twelve distinct task durations range from **219.4 to 1302.0 seconds**, mean **749.425**, median **640.75**. The 1200-second decode-loop cap is soft and can be exceeded.

A CPU-only discrete-event simulation samples 240 durations with replacement, assigns them FIFO to four workers, and counts completions within **710 minutes**. There are **5,000 replicas**, seed **20261106**.

| Assumed latency multiplier | Median completions | Simulated 5th–95th percentile |
|---|---:|---:|
| 1.00× | 226/240 | 215–237 |
| 1.25× | 180/240 | 170–190 |
| 1.50× | 150/240 | 141–159 |

These are **conditional simulation outputs**, not observed hidden coverage, hidden-set confidence intervals, or evidence of an accuracy gain. No GPU rerun or new model score is claimed. The simulation omits model-loading overhead and does not model correctness or latency/accuracy correlation.

The identity `S = C × p` holds for matching scored units and weights when unattempted rewards are zero. With rewarded fallbacks, add their contribution. Full-queue completion is not necessarily public-subset coverage. We therefore **withdraw the earlier inferred hidden success range and “six more tasks for top 200” claim**. A 2.50-point gap to the historical 33.06 threshold does not establish a solved-task count or guarantee a placement.

### 4. Smoke-test accuracy: keep denominators visible

The retained 12-task public-evaluation validation records:

- first-attempt exact rows: **1/18 (5.56%)**;
- exact rows using either of two attempts: **3/18 (16.67%)**;
- mean per-task fraction of correct rows: **19.44%**;
- all test rows correct: **2/12 tasks (16.67%)**;
- full model bundles: **11/12 tasks**; invalid emitted entries: **0**.

Using 2/11 would condition on successful bundle production; we report all 12 selected tasks. Public-model training history may include these tasks. This is a smoke test, **not unbiased held-out accuracy**. Schema-valid fallback entries are not evidence of reasoning.

## Score, provenance, and eligibility

System B is a reproduction of the **Apache-2.0** public notebook `koushikrudra/failed-in-aimo` v1, using `sorokin/qwen3_4b_grids15_sft139` model v1. We credit the source and upstream contributors. Our changes concern model-mount discovery, launcher checks, GPU preflight, a seeded validation queue, and separate grading. The source configuration uses rank-256 LoRA, alpha 32, learning rate 5e-5, one epoch, augmentation setting `n=16`, and sequence length 8192—not an independently verified fixed 128-step schedule.

Submission **57901718**, notebook `krishnakumarbhat/arc-nvarc-validated` v3, received **30.56** but completed late and was excluded from the standings. The team's counted result remains **0.00**, historical rank **3599/3627**. A 30.56 score appears near position 1271 in the retained leaderboard snapshot; **that is not our official rank**. From zero, the difference is 30.56 points, not a threefold improvement.

The submission was sent 2 November at 08:28 UTC and remained pending at 08:44. The retained record establishes late completion, not the exact queue duration or its cause. Eligibility requires `submit time + queue delay + runtime + safety margin < close`. Earlier submission improves slack; it does not by itself improve accuracy.

## Reproduce and inspect

Download `arc_audit_final.zip`, preserve its directory layout, and run:

```bash
python3 03-arc-prize-2026-paper-track/audit_final.py
```

Dependencies: NumPy and Matplotlib. No GPU, model weights, credentials, or network are required. The bundle includes the historical metric JSONs, latency log, validation outcomes, audited canonical/search sources, generated figure, and `AUDIT_FINAL.json` with source SHA-256 hashes. It contains only competition-relevant material.

## Limitations and next experiments

No new hidden score, repaired-search accuracy, or scheduling-policy accuracy comparison is claimed. Twelve latencies are insufficient for hidden-set forecasts. Historical runs differ in version, public-model contamination is possible, and exact hidden scored coverage is unavailable.

The next defensible experiments are oracle candidate accuracy versus selection accuracy, original versus repaired canonicalisation on an untouched split, and scheduling evaluation with **jointly observed latency and correctness**. None is presented as completed here. The central lesson is to separate **training fit, test accuracy, prediction availability, and submission eligibility** before calling an ARC result progress.

## References

- Chollet et al., *ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems*, arXiv:2505.11831: https://arxiv.org/abs/2505.11831
- Original reproduced notebook: https://www.kaggle.com/code/koushikrudra/failed-in-aimo
- Public model: https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139
