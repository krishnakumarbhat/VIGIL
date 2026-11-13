# S6E10 OOF Library — 19 prediction columns, one fixed fold split

Out-of-fold and test predictions for Playground S6E10 (Predicting Airline Satisfaction), all trained on
`StratifiedKFold(5, shuffle=True, random_state=42)` over `train.csv` in its original row order.
Because every column shares the same folds, you can stack them, or stack them with your own OOFs, and get an honest CV score.

| column(s) | model | OOF AUC |
|---|---|---|
| `lgb` | LightGBM, lr 0.02, 127 leaves | 0.961014 |
| `xgb` | XGBoost, depth 6, max_bin 1024 | 0.960911 |
| `cat` | CatBoost, depth 7 | 0.960901 |
| `cat_ctr` | CatBoost with Flight Distance, Age and the 4 categoricals as categorical features (ordered CTR) | 0.960980 |
| `lgb_route` | LightGBM on a route-centric view: raw + route profile + counts + original-model logit + 3 target encodings | 0.961035 |
| `lgb_ogteach` | `lgb` + the original-only RealMLP teacher logit as a feature | 0.961000 |
| `realmlp_s*` (3 seeds) | RealMLP on raw columns + categorical twins, flat_anneal, 3 epochs | 0.960954–0.960972 |
| `realmlp_fa4_s*` (3 seeds) | same, 4 epochs | 0.960962–0.960984 |
| `realmlp_og_s*` (2 seeds) | same + two original-only teacher logits (LightGBM, RealMLP) as inputs | 0.960870–0.960917 |
| `realmlp_v4_s*` (3 seeds) | RealMLP fed the engineered features + fold-safe target encodings, lin_cos_log_15, 6 epochs | 0.961133–0.961159 |
| `realmlp_v4og_s*` (2 seeds) | v4 + the original-only RealMLP teacher logit | 0.961149–0.961206 |

Logistic-regression stack with seed groups averaged into one member each (11 members): **0.961552 nested-CV AUC**.
History: v1 0.961410 → v2 0.961503 → v3 0.961552.

## Files
- `oof_predictions.parquet`: `id`, `fold`, `satisfaction` (label), one probability column per model
- `test_predictions.parquet`: `id`, one probability column per model (mean of the 5 fold models)

## Features
Raw columns, value counts over train+test, original-dataset target mean per value, distance/age digits,
predictions of models trained on the original dataset only (LightGBM, RealMLP — no competition label involved),
GPT-2 token keys of the numeric columns, the route profile (per-Flight-Distance mean of every other column, no target)
and fold-safe target encoding (exact distance, distance // 10, distance % 1000, age × class, distance × class).

Credits: token keys and the original-only model — [arhancanli12](https://www.kaggle.com/code/arhancanli12/s6e10-generator-fingerprints-lgbm-cv-0-9610);
route profile, the feature-fed RealMLP v4, the original-data RealMLP teacher and the CatBoost CTR idea —
[goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127);
RealMLP parameters — [Vladimir Demidov](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit).

Trees were early-stopped on their validation fold (patience 300), as in most public notebooks; that flatters each
tree OOF very slightly. RealMLPs train a fixed number of epochs.
