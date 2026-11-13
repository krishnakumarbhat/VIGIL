# Track 06 (S6E10 Airline Satisfaction) - Optimisation Report
Date: 2026-10-22
Workspace: /home/pope/Desktop/gpu /hackathons/06-playground-s6e10-airline-satisfaction/

## Measured Baseline (existing artifacts, verified)
- 10-fold StratifiedKFold rs=42 protocol used throughout.
- Best blend AUC: **0.959666** (7-member logit blend: cat/lgbm/xgb/lgbm_base/lgbm_aug/realmlp/nnprior; weights from max_ensemble.py).
- Individual OOF AUCs (verified from outputs/oof_*.npy):
  - cat: 0.959052
  - realmlp: 0.959001
  - lgbm_aug (original dataset +129880 rows): 0.957917 (+0.000245 over base)
  - lgbm_base: 0.957673
  - xgb: 0.956385
  - nnprior: 0.943050

## Completed Actions
1. **Re-verified all existing artifacts**: outputs/oof_blend.npy, submission.csv (299844 rows, float probs), test predictions present.
2. **Experiment log updated** (`logs/experiments.tsv`): baseline recorded; sweep started.
3. **Hyperparameter sweep launched** (`src/sweep_blend_2026.py`, PID 3822155, running in background):
   - CatBoost: depth 8-9, 3000 iters, lr=0.03, seeds 42/1337/2024 (CPU mode for VRAM <3.2GB).
   - LightGBM: leaves 63 (min_data_in_leaf=50, ff=0.8, bf=0.8) and 127 (min_data=20, ff=0.7, bf=0.7), seeds 42/1337, lr=0.05/0.03.
   - Feature variant: interaction aggregates (flight_class_interaction, wifi_x_ent, rating_rel_mean, delay_diff, value_proxy) + numeric encoding.
   - Cat seed ensemble: average of top 2 variants (to be computed after sweep completes).
4. **Reproducibility**: `src/build_best.py` exists with `--fast` flag, correct ROOT path, full OOF + blend pipeline, and hard guard (`IMPROVED = final_auc > prev_best` with default prev_best=0.959666). `src/enhanced_build.py` and `fast_regen.py` also present.
5. **Negative results confirmed** (from logs):
   - Original dataset exact-match leakage: 17/699635 rows match; label agreement 0.0 → NEVER RETRY.
   - Meta LR stack (18 configs): best <0.9593; collapses to cat-only → REJECTED.
   - nnprior weight in blend: ~0.003 → minimal contribution.
6. **No Kaggle submission performed** (per rules); no `kaggle competitions submit` called.

## In-Progress / Pending
- Background sweep (PID 3822155) still running (CPU 312%, ~1.4GB RAM, safe under 31GB limit). Not yet produced new `.npy` files.
- Once sweep finishes: blend over ALL members (existing 7 + new seed variants + feature variant) with Nelder-Mead on logit scale; hard guard requires AUC > 0.959666 to write `submission.csv`.
- If new blend exceeds 0.959666: write `outputs/submission.csv`, `outputs/oof_blend.npy`, `outputs/test_blend_prob.npy`, and update `outputs/model_report.md` and `logs/experiments.tsv`.
- If not improved: keep existing `submission.csv` and report no regression.

## Risks / Blockers
- Test predictions for new members are approximate (reuse existing member predictions for blend approximation). Only final blend is verified against measured full OOF AUC.
- CatBoost using CPU (`task_type="CPU"`) due to RTX 3050 4GB VRAM limit and 3.2GB cap; slower but safe.
- Kaggle GPU kernels (`ntl-stage2-deberta` / `arc-agi2-llm-ttc`) untouched; no concurrent GPU kernels exceeded.
- Approximate test blend may differ slightly from true predictions for members without exact test arrays; this is explicitly tracked in blend construction.

## Deliverables
- Changed paths: `src/sweep_blend_2026.py` (new), `logs/experiments.tsv` (updated), `outputs/model_report.md` (this file).
- Unchanged (if no improvement): `outputs/submission.csv` (retained existing), `outputs/oof_blend.npy`, `outputs/test_blend_prob.npy`.
- Reproducible: `python3 src/build_best.py --fast` regenerates blend from artifacts; `python3 src/sweep_blend_2026.py` runs full sweep + blend.
