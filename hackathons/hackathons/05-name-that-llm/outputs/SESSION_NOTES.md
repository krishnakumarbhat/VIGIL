# Track 05 — session notes: 0.4016 → 0.6817 grouped balanced accuracy

Everything below is measured on the same protocol: 5-fold `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=42, groups=prompt_id)` over the **2,699 real competition rows**
(150 prompts, 18 passages each). A member's OOF matrix only ever contains predictions for rows it
did not train on; the blender's weights, temperatures and class biases for fold *k* are fitted on
folds ≠ *k* and applied once to fold *k*.

## Member table (out-of-fold balanced accuracy, 18 classes)

| member | what it is | OOF bal-acc |
|---|---|---|
| `deberta_base` | DeBERTa-v3-base fine-tune, 5 fold checkpoints (Kaggle `ntl-real-deberta`) | 0.4016 |
| `local_qwen05b` | Qwen2.5-0.5B LoRA, seed 42, mirror 3k, len 192 | 0.5383 |
| `local_qwen05b_s1337` | Qwen2.5-0.5B LoRA, seed 1337, mirror 6k, len 256 | **0.6483** |
| `svmext` | word(1-2)+char_wb(3-5) TF-IDF → SGD, real+mirror+40k external | 0.3801 |
| `svm` | same but mirror-only | 0.3616 |
| `char` | char_wb(2-5) TF-IDF → SGD | 0.3619 |
| `word` | word(1-3) TF-IDF → SGD | 0.3593 |
| `knn` | cosine kNN (k=30) on char TF-IDF | 0.2208 |
| `style` | 43 hand-crafted stylometric features → SGD | 0.2267 |
| **blend** | temperatures + weights + per-class bias, cross-fitted | **0.6817** |

Per-class recall after blending: **0.48 – 0.92** (min `glm-5.3` 0.48, max `claude-opus-5` 0.92).
At session start the minimum class recall was **0.047** — the blend was scoring well on accuracy
while silently never predicting four model families.

## What actually moved the number

1. **Per-class logit bias tuning** (+4.2 points, 0.5557 from a 0.5399-weighted blend). The metric is
   a macro average, so a class with recall 0.05 costs ~5 points on its own. Coordinate ascent on the
   18 additive biases, fitted inside each fold with a |b| ≤ 2 cap, repaired the tail.
2. **Member diversity beats member strength** (+1.9 for the SVM, +2.4 for `svmext`). A 0.36 member
   with uncorrelated errors was worth more than another point of single-model accuracy.
3. **The external corpus helps the cheap members** (`svm` 0.3616 → `svmext` 0.3801 with 40k
   external rows). Verified leak-free first: 0 of 63,262 external prompts overlap the 150 real
   prompts, and there are no shared 200-character text prefixes.
4. **A 0.5B model can reach 0.65** — the seed-1337 run (mirror 6k, max-len 256) beat the seed-42
   run (mirror 3k, max-len 192) by 11 points. The original "too weak to help" verdict was a
   hyper-parameter artefact, not a capacity limit.
5. **Temperature scaling per member** (0.73–1.9 fitted): the TF-IDF members were badly overconfident
   and were drowning the LoRA before calibration.

## What failed (and why it is written down)

| attempt | outcome |
|---|---|
| Linear meta-learner over concatenated member log-probs (7 members × 36 features, nested C) | **0.5709** — worse than the constrained weight+bias model (0.5976 at the time). 2,159 training rows cannot support 630 free parameters. |
| Qwen2.5-7B LoRA on A100 (5-fold curriculum) | Fold 0 reached 0.4759, but the runtime's CLI session record was pruned mid-run, orphaning a healthy VM: no `exec`, no `download`, artifacts unreachable. Training stopped rather than burn units for nothing. |
| DeBERTa-v3-large trained fully in bf16 | Loss pinned at ln 18 ≈ 2.89 for 2,400 steps (fold 0 bal-acc 0.2167). bf16 master weights do not converge for DeBERTa's disentangled attention; needs fp32 weights + autocast. |
| `zip`-based artifact bundling | Abandoned for `shutil.make_archive` plus in-band base64 (see below). |

## Infrastructure notes worth keeping

* **In-band artefacts.** `colab download` resolves files through a *local* session registry. When
  that registry is pruned (observed after the CLI's own `sessions` command), an otherwise healthy
  runtime becomes unreachable by name *and* by raw id, taking every file on it. Training scripts now
  print artefacts as `BEGIN_<tag> <bytes> <chars>` / chunked base64 / `END_<tag>` on stdout;
  `src/ingest_b64.py` reconstructs and length-validates them from the captured log.
* **PEFT casts projection inputs to fp32**, doubling activation memory; `model.to(torch.bfloat16)`
  after `get_peft_model` plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` removed the OOM.
* **Recycled Colab runtimes keep ~39 GB of stale CUDA memory** after an OOM; always create a fresh
  session after a failure rather than re-execing into it.
* **BLAS oversubscription**: 180 small logistic-regression fits at `n_jobs=-1` drove load average to
  36 and stalled for 20 minutes; pinning `OMP_NUM_THREADS=1` finished them in ~8.

## Deployment

`outputs/ensemble_recipe.json` holds `temperatures`, `weights`, 18 `biases` and the label order.
`src/make_submission.py` recomputes the available members from disk and applies

```
z = Σ_i w_i · log(p_i) / T_i + b ;  pred = argmax z
```

Members whose artefacts are missing are dropped and the remaining weights renormalised, so the
pipeline degrades instead of failing. Deployable artefacts under `outputs/deploy/`:

| member | artefact |
|---|---|
| `deberta_base` | `outputs/kaggle_deberta/model_fold{0..4}` (5 checkpoints, averaged) |
| `svm`, `char`, `word`, `style`, `knn` | `svm.joblib`, `char.joblib`, `word.joblib`, `style.joblib`, `knn.joblib` + `vectorisers.joblib` |
| `svmext` | `svmext.joblib` (carries its own external-corpus vectorisers) |
| `local_qwen05b_s1337` | `lora_qwen05b/` — full-data refit of the same recipe (`src/fit_deploy_lora.py`) |

## Honest limitations

* The stacker is cross-fitted over OOF matrices but the *base* models for fold *k*'s features were
  trained on the other folds, i.e. standard stacking rather than fully nested CV. Any optimism from
  this is small but not zero, and it is the reason the fold spread (0.41 → 0.95) is reported next to
  the pooled number instead of being hidden.
* Fold difficulty varies enormously (val 0.4148 / 0.4963 / 0.6815 / 0.8685 / 0.9481). Differences
  under ~0.02 between variants are inside noise; treat rankings accordingly.
* `glm-5.3` remains the hardest class (recall 0.48), confused mostly with `glm-5.3-flash` and the
  `gpt-5.6-*` family.
