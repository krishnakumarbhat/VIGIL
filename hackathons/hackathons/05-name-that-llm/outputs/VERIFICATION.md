# Track 05 verification — 2026-10-23

Every claim below was re-executed on this machine after the models were rebuilt. Commands are
copy-pasteable from the track root.

## 1. No prompt or text leakage between training corpora and validation folds

```
real prompts 150 (namespace p###)  ·  mirror prompts 510 (g###)  ·  external prompts 63,262
prompt overlap real∩mirror   = 0
prompt overlap real∩external = 0
200-char text-prefix overlap real∩mirror   = 0
200-char text-prefix overlap real∩external = 0
```

## 2. Every member's OOF matrix covers the 2,699 real rows exactly once

```
local_qwen05b        rows 2699 unique 2699 OK
local_qwen05b_s1337  rows 2699 unique 2699 OK
svmext               rows 2699 unique 2699 OK
svm                  rows 2699 unique 2699 OK
char                 rows 2699 unique 2699 OK
word                 rows 2699 unique 2699 OK
style                rows 2699 unique 2699 OK
knn                  rows 2699 unique 2699 OK
```

No row is predicted by a model that trained on it, and no row is counted twice.

## 3. Row alignment

`outputs/kaggle_deberta/row_index_42.parquet` matches `data/competition/train.csv` row-for-row:
`id` order identical, `prompt_id` identical, `y` identical to the mapped `model` string. All member
matrices are indexed in that order (`np.load("..._idx_fold{i}.npy")`).

## 4. Recipe integrity

```
weights sum to 1.0
18 per-class biases, max |b| = 0.5 (cap 2.0)
labels = the 18 official strings, order fixed in outputs/ensemble_recipe.json
```

## 5. Submission builder

```
python3 src/make_submission.py --list-members
  reproducible : deberta_base, char, knn, local_qwen05b_s1337, style, svm, svmext, word
  missing      : local_qwen05b   (superseded by the seed-1337 recipe; weights renormalised)

python3 src/make_submission.py --test data/competition/test.csv --out outputs/submission.csv
  -> 3 rows, ids match test.csv order, labels all valid

python3 src/make_submission.py --test /tmp/opencode/test64.csv --out /tmp/opencode/sub64.csv
  -> 64 rows, 18 distinct predicted labels, ids aligned  (exercises batch > 1 through every member)
```

## 6. Artefact recovery path

`src/ingest_b64.py` round-trip tested on a random 5,000-byte payload: printed as a chunked base64
block and reconstructed byte-identical. This is the channel that survives a pruned Colab session
registry, which is what destroyed the 7B run's matrices.

## 7. Honest caveats

* Blend parameters are fitted on folds ≠ *k* (cross-fitted), but the base models behind fold *k*'s
  features trained on the other folds — standard stacking, with a small non-zero optimism.
* Fold-level blend scores span 0.4148–0.9481; the pooled 0.6817 is their mean because the folds are
  equal-sized. If the private test's prompt mix is harder than this training set's mix, expect lower.
* Nothing was submitted to Kaggle. `outputs/submission.csv` / `.parquet` are staged only.
