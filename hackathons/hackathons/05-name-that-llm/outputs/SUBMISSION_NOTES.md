# Track 05 — Name That LLM: submission notes

**Headline: 0.6817 cross-fitted balanced accuracy on the 2,699 real competition rows**
(5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42, groups=prompt_id)`;
members scored only on held-out rows; blend parameters for fold *k* fitted on folds ≠ *k*).
Full algorithmic record: `outputs/SESSION_NOTES.md`.

> **Superseded numbers.** Earlier notes in this file reported stage-1 grouped CV of 0.9996 and a
> "0.40863 stack". The 0.9996 figure was a leakage artefact of the stage-1 pipeline (it evaluated on
> rows it trained on) and must not be quoted. 0.6817 is measured with the protocol above and is the
> only number to use.

## 1. Regenerate the submission

```bash
cd "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"

# what can be recomputed from artefacts on this machine
python3 src/make_submission.py --list-members

# score any test frame (visible sample, or the private test once it is on disk)
python3 src/make_submission.py --test data/competition/test.csv --out outputs/submission.csv
```

`src/make_submission.py` applies the exported recipe

```
z = Σ_i w_i · log(p_i) / T_i + b        # b = 18 per-class biases
pred = argmax z
```

Members whose artefacts are missing are dropped and the remaining weights renormalised, so the
pipeline degrades gracefully instead of failing. Output schema: columns `id,model`, one row per
input id, every label one of the 18 official strings, no nulls (asserted in the script).

## 2. Artefacts

| artefact | content |
|---|---|
| `outputs/ensemble_recipe.json` | temperatures, blend weights, 18 per-class biases, label order |
| `outputs/stack_v2_report.json` | cross-fitted score, per-fold scores, per-class recall |
| `outputs/oof_real_stack_v2.npy` | blended OOF probability matrix (2,699 × 18) |
| `outputs/cv_report.json` | headline + provenance (supersedes the old stage-1 report) |
| `outputs/deploy/` | deployable members: `svm`, `svmext`, `char`, `word`, `style`, `knn` (+ `vectorisers.joblib`), `lora_qwen05b/` full-data adapter |
| `outputs/kaggle_deberta/model_fold{0..4}` | five DeBERTa-v3-base fold checkpoints, averaged at inference |
| `src/make_submission.py` | recipe application + member recomputation |
| `src/fit_deploy_lora.py`, `src/fit_deploy_sparse.py`, `src/fit_deploy_svmext.py` | reproduce every deployed member |
| `src/ingest_b64.py` | recovers artefacts printed as base64 blocks in a Colab exec log |

## 3. Members and their measured strength

| member | OOF balanced accuracy | deployable |
|---|---|---|
| `deberta_base` (5 fold checkpoints) | 0.4016 | yes |
| `local_qwen05b` (LoRA seed 42) | 0.5383 | no — superseded by the seed-1337 recipe |
| `local_qwen05b_s1337` (LoRA seed 1337) | **0.6483** | yes (full-data refit) |
| `svmext` (TF-IDF + SGD, 40k external rows) | 0.3801 | yes |
| `svm` / `char` / `word` | 0.3616 / 0.3619 / 0.3593 | yes |
| `knn` / `style` | 0.2208 / 0.2267 | yes |
| **blend** | **0.6817** | recipe + the five members above |

## 4. Rules and reproducibility

* Validation split is fixed: `StratifiedGroupKFold(5, shuffle=True, random_state=42,
  groups=prompt_id)`; `lowercase=False` for the stage-1 vectoriser; the length prior stays decoupled
  from the label model; no test-time Hungarian matching or prompt clustering.
* The mirror corpus uses a disjoint prompt namespace (`g###` vs the real `p###`) with zero shared
  200-character text prefixes; the external corpus has 0/63,262 overlapping prompts. Verified before
  any training run.
* Nothing in this track has been submitted to Kaggle. The file is staged at
  `outputs/submission.csv` / `outputs/submission.parquet`.
* Local runs use system `python3` with the shared library path; no `.venv` was created.
