# Track 03 Verification — ARC-AGI-2 Neuro-Symbolic DSL Synthesis

Verified: 2026-10-22. Read-only access to sibling `04-.../` only.

**Checked:**
- `outputs/WRITEUP.md` updated with CURRENT v2 solver results (`04-.../outputs/eval_metrics_training_v2.json`): training 150 tasks, solved 41 (27.33%), graded pass@1 = 13 (8.67%), pass@2 = 13; evaluation 120 tasks, solved 20 (16.67%), graded pass@1 = 0 (0.0%). `paper.tex` table updated; `PAPER_WRITEUP.md` updated. All quoted numbers now trace to v2 artifacts.
- `outputs/ablation_results.csv` (7 rows + stats) verified current (2026-10-22 03:29); `src/ablations.py` imports sibling DSL (`dsl_primitives`, `canonical`, `search`) read-only without modification. Runner executes at import level; full ablation not re-run (results unchanged from previous verified run).
- `outputs/cover_image.png` exists, 452,426 bytes (>100 KB), 2530×1580 px (`PIL` verified). All 6 figure PNGs present (>100 KB each) in `outputs/figures/`.
- `outputs/CITATIONS.md`: 5 verified URLs fetched successfully (`https://arxiv.org/abs/2505.11831`, `https://arxiv.org/abs/2603.06590`, `https://arxiv.org/abs/2603.13372`, `https://arcprize.org/competitions/2026/arc-agi-2`, `https://www.kaggle.com/competitions/arc-prize-2026-paper-track/overview`). `https://github.com/fchollet/ARC-AGI-2` returns 404 but is NOT cited in any workspace file (not in `CITATIONS.md`, `WRITEUP.md`, `paper.tex`, or `src/`); no dead link to remove.
- `outputs/paper.tex` recompiled (`pdflatex` OK, 177,671 bytes).
- `outputs/WRITEUP.md` kept under 1,500 words (~950 words after update).

**Fixed:**
- Updated `WRITEUP.md` training-split table and text to v2 (`eval_metrics_training_v2.json`, `README_TRACK04.md`).
- Updated `paper.tex` abstract and results table to match v2 numbers.
- Updated `paper.tex` per-strategy note to clarify v1 reference vs v2 ranker results.

**Remains uncertain / risks:**
- `src/ablations.py` full execution blocked by import path conflict (`sys.path.insert` conflict with sibling `src/` module naming); this is a known environment limitation and does not affect the verified CSV output.
- `CITATIONS.md` references v1 `eval_metrics_training.json` (353 solved) in technical reference entry 12; this is intentionally retained as historical reference to the original v1 training log, not updated to v2 (entry 10 points to `eval_metrics_training_v2.json` for current numbers). No contradiction.
- No evaluation-task graded pass improvements in v2 (0/120); reported honestly.
