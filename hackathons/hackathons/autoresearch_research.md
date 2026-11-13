# Autoresearch Marathon: Multi-Track Kaggle Competition Optimization

## Topic
Maximise verified, submission-ready artefacts across 8 concurrent Kaggle/Devpost tracks
under three GPU tiers (local RTX 3050 4GB, Kaggle T4 x2, Colab Pro T4), with a strict
no-submission policy (everything staged under each track's `outputs/`).

## Tracks, metrics, current baselines
| Track | Metric (official where known) | Baseline (measured) | Target |
|---|---|---|---|
| 01 NFL Big Data Bowl 2027 | competition metric unknown -> regression proxy R2 / MAE | GroupKFold OOF R2=0.0355, MAE=1.635 | >0.25 R2 |
| 02 Gemma 4 Developer Agent Paper | localisation recall@k, patch pass rate | GG recall@5=0.318 vs flat 0.273, tokens -43% | recall@5 >0.45 |
| 03 ARC Prize 2026 Paper Track | rubric (accuracy/universality/theory) | writeup 1330 words, 6 figures, 5 verified citations | complete + ablations |
| 04 ARC-AGI-2 | exact match on eval set | train pass@1 6.8% (68/1000), eval pass@1 0/120 | eval pass@1 >0 |
| 05 Name That LLM | balanced accuracy (18 classes) | grouped CV 0.3906 (5-fold SGKF, group=prompt_id) | >0.46 |
| 06 Playground S6E10 | ROC-AUC | 0.959265 (lgbm .9576 / cat .9590 / xgb .9564) | >0.9613 |
| 07 OpenCV AI Competition | rubric (100 pts) + agentic/COOL awards | 55 tests pass, agent 35 FPS, detection ROC-AUC 0.9917 | 1-click AWS deploy + COOL evidence |
| 08 OpenZee OpenAir | unknown | recon only | identification |

## Protocol per iteration
1. Pick the highest-potential frontier node in `autoresearch_research_strategy_graph.jsonl`.
2. Formulate a hypothesis with a measurable metric.
3. Implement in the track's `src/` (one class per file, absolute imports, typed, docstrings).
4. Evaluate with the track's STRICT validation (never random KFold where the metric requires grouping).
5. Keep only if the metric improves; log to `autoresearch_research.jsonl`, the strategy graph,
   `variations.md`, and `experiments/worklog.md`.
6. Append the measurement to the track's `logs/experiments.tsv`.

## Hard constraints
- NEVER submit to Kaggle or Devpost.
- Kaggle GPU: max 44h/week, keep 1h emergency reserve.
- Colab Pro: billable from second 1 - always stop sessions; DNS wait-loop mandatory.
- Local RTX 3050 only for workloads under 3.2GB VRAM.
- Never modify /media/pope/virtbox/a_resume/.
- Track 05 validation: StratifiedGroupKFold(n_splits=5, groups=prompt_id, shuffle=True, random_state=42);
  never random KFold; never test-time Hungarian/prompt clustering.
