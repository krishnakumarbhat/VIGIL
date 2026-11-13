"""Refit the stacked winner and regenerate Track 05 deliverables.

OOF finding (this iteration): a cross-fitted ``LogisticRegression(C=8,
class_weight=balanced)`` stacker over the OOF probabilities of
``svc_extra_claude_w50`` / ``nbsvm_all`` / ``svc35_w0`` reaches 0.40863 grouped
CV balanced accuracy vs the stage-1 simplex blend 0.39061 (+0.018) -> KEEP.

Run AFTER ``stack_probe.py`` (needs the three cached OOF matrices).
Writes ``outputs/final_model_stacked.joblib``, refreshed
``final_model_config.json``/``submission.*`` and appends the stacked update to
``SUBMISSION_NOTES.md``.
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntl_features import LABELS, load_competition_frame, load_external_frame  # noqa: E402
from finalize import build_reconstruction, log_experiment  # noqa: E402
from stack_model import StackedNTLModel  # noqa: E402

ROOT = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{ROOT}/outputs"
CACHE = f"{OUT}/cache"

CFG = dict(views=("char", "word", "shape", "sty"), char_ngram=(2, 5), word_ngram=(1, 2))
CFG35 = dict(views=("char", "word", "shape", "sty"), char_ngram=(3, 5), word_ngram=(1, 3))
META_C = 8.0
CV_KEEP = 0.40863
BASELINE = 0.39061
SWEEP = {"c2": 0.40344, "c3": 0.40622, "c4": 0.40626, "c8": 0.40863,
         "c16": 0.40858, "c32": 0.40834, "c1000": 0.40126}


def main() -> None:
    comp = load_competition_frame()
    ext = load_external_frame(max_per_class=1200)
    y = comp["y"].to_numpy()
    clean = (comp["split"] != "pilot_range").to_numpy()

    oof = {t: np.load(f"{CACHE}/oof_{t}.npy") for t in
           ("svc_extra_claude_w50", "nbsvm_all", "svc35_w0")}

    print("fitting StackedNTLModel ...", flush=True)
    model = (StackedNTLModel(CFG, CFG35, extra_claude_weight=0.5, meta_C=META_C)
             .fit(comp, ext, {"svc": oof["svc_extra_claude_w50"],
                              "nbsvm": oof["nbsvm_all"],
                              "svc35": oof["svc35_w0"]}))
    joblib.dump(model, f"{OUT}/final_model_stacked.joblib")

    in_cv = np.hstack(list(oof.values())).astype(np.float32)
    cv = model.meta.predict_proba(in_cv)
    print(f"in-sample meta clean={balanced_accuracy_score(y[clean], cv[clean].argmax(1)):.5f}",
          flush=True)

    json.dump({"recipe": "cross-fitted LR(C=8, class_weight=balanced) meta over "
                         "3-member OOF probabilities",
               "members": sorted(oof), "meta_C": META_C,
               "folded_cv_bal_acc_clean": CV_KEEP, "baseline": BASELINE,
               "reg_C_sweep": SWEEP},
              open(f"{OUT}/cv_stack_report.json", "w"), indent=1)
    log_experiment("bal_acc_stack_probs_c8", CV_KEEP,
                   "cross-fitted LR C=8 class_weight=balanced meta over "
                   "svc_extra_claude_w50+nbsvm_all+svc35_w0 OOF probs only, "
                   f"grouped 5-fold clean subset; baseline {BASELINE} -> KEEP; "
                   "reg sweep " + ";".join(f"{k}={v:.4f}" for k, v in SWEEP.items()))

    recon = build_reconstruction()
    P = model.predict_proba(recon["text"].tolist())
    sub = pd.DataFrame({"id": recon["id"].astype(str),
                        "model": [LABELS[i] for i in P.argmax(1)]})
    sub.to_parquet(f"{OUT}/submission.parquet", index=False)
    sub.to_csv(f"{OUT}/submission.csv", index=False)
    truth = recon["model"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    recon_acc = float(balanced_accuracy_score(truth, P.argmax(1)))
    print(f"submission rows={len(sub)} recon_bal_acc={recon_acc:.4f}")

    json.dump({"architecture": "StackedNTLModel", "members": ["svc", "nbsvm", "svc35"],
               "meta": "LogisticRegression", "meta_C": META_C,
               "feature_config": CFG, "cfg35": CFG35, "extra_claude_weight": 0.5,
               "cv_bal_acc_clean": CV_KEEP, "cv_baseline": BASELINE,
               "recon_bal_acc": recon_acc},
              open(f"{OUT}/final_model_config.json", "w"), indent=1)

    with open(f"{OUT}/SUBMISSION_NOTES.md", "a") as fh:
        fh.write(f"""

## Stacked update (second iteration)
Cross-fitted ``LogisticRegression(C=8, class_weight=balanced)`` meta-stacker over the OOF
probabilities of ``svc_extra_claude_w50`` + ``nbsvm_all`` + the new ``svc35_w0`` diversity
member (char 3-5 / word 1-3 view).  Grouped 5-fold CV (groups=prompt_id): **balanced
accuracy {CV_KEEP:.4f}** on the clean subset vs {BASELINE:.4f} for the stage-1 blend
(+0.018).  The regularisation sweep (C in 2..1000) peaks at C=8; kNN-vote features in
TF-IDF space and rank-averaged blends were implemented and rejected (both score below
baseline).  ``final_model.joblib`` keeps the stage-1 ensemble; ``final_model_stacked.joblib``
is this winner.  The shipped ``submission.parquet`` was regenerated from the stacked model.
""")


if __name__ == "__main__":
    main()
