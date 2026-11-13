# Track 01 Verification — NFL Big Data Bowl 2027

Verified: 2026-10-22. No Kaggle submission.

**Checked:**
- All 8 figures (`outputs/figures/*.png`) exist, all >126 KB (>20 KB pass), produced from real `data/kaggle_nfl/nfl_data/` tables (combine_tracking.csv 463,189 rows, game_tracking_2023.csv 3,665,443 rows, player_play.csv 314,197 rows). No synthetic generation path in `src/figures.py`.
- Notebook `notebooks/nfl_combine_to_game_kce.ipynb` has 10 cells, uses real CSV paths (`data/kaggle_nfl/nfl_data/`), no broken relative paths; jupyter unavailable so full interactive execution not performed, but all imports (`kce_features`, `kce_model`, `run_analysis`) verified executable.
- Numbers in `outputs/WRITEUP.md` cross-checked against `outputs/model_metrics.json` and `logs/experiments.tsv`: headline R² 0.6561, MAE 1.3503, ρ 0.825, mixed-effect R² 0.2756, directional KCE R² 0.9881 (convergence flag noted), learning curve 5-point values all present. `DATA_SCHEMA.md` matches real CSV column inventory (verified by direct CSV read).
- `outputs/WRITEUP.md` kept under 2,000 words (~207 lines, well under limit).

**Fixed:** None required for numbers; `DATA_SCHEMA.md` row count verified correct (314,197 for player_play.csv — embedded newlines explain line-count discrepancy, pandas non-null count confirms).

**Remains uncertain / risks:**
- Full end-to-end notebook execution blocked by missing `jupyter`/`nbconvert` environment (verified via `python3 -c` import checks only).
- Not all quoted metrics in `WRITEUP.md` are exhaustively matched to JSON keys (e.g. `headline.improvement_over_prior_r2` = 18.48 exists in JSON but not quoted in writeup); no contradiction found.
- No synthetic figure mislabel detected (only reference to synthetic data is `src/synth_combine.py`, which is not used by `src/figures.py`).
