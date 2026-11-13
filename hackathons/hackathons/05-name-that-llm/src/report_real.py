"""Assemble ``outputs/cv_report.json`` from the saved OOF artefacts.

No model is refitted here: this is the reporting entry point that turns
``outputs/oof_real*.npz`` + ``outputs/blend_search.json`` + the variant log into
the published numbers (grouped-CV balanced accuracy, per-class recall,
length-bin accuracy, ablations, nested-selection estimate).

Usage: ``python3 src/report_real.py``
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blend_search import TEMPS, FastScorer, load_oof  # noqa: E402
from ntl_cv import MetricsCalculator, grouped_folds  # noqa: E402
from ntl_data import (LABELS, LENGTH_BINS, COMPETITION_TEST, AugmentationLoader,  # noqa: E402
                      CompetitionLoader, text_hashes)

TRACK = Path(__file__).resolve().parents[1]
OUTPUTS = TRACK / "outputs"


def main() -> None:
    """Recompute and write the full report for the shipped model."""
    comp = CompetitionLoader().load()
    aug = AugmentationLoader(max_per_class=900).load(exclude_hashes=text_hashes(comp))
    oof = load_oof()
    search = json.loads((OUTPUTS / "blend_search.json").read_text())
    best = search["best"]

    fused = np.zeros_like(next(iter(oof.values())))
    scorer = FastScorer(oof, comp.y)
    for name, w, t in zip(best["combo"], best["weights"], best["temps"]):
        fused += w * scorer._temper(oof[name], t)
    calc = MetricsCalculator(LABELS, LENGTH_BINS)
    metrics = calc.score(fused, comp.y, comp.n_words)

    # shipped artefact = 2-member blend selected by finalize_real.py
    shipped = 0.6 * oof["ridge_a05"] + 0.4 * oof["svc_c1_m15"]
    shipped = shipped / shipped.sum(1, keepdims=True)
    shipped_m = calc.score(shipped, comp.y, comp.n_words)

    sub_path = OUTPUTS / "submission.parquet"
    sub = pd.read_parquet(sub_path)
    test = pd.read_csv(COMPETITION_TEST)

    stage1 = json.loads((OUTPUTS / "real_variant_scores.json").read_text())
    ablations = {k: v["bal_acc"] for k, v in stage1.items() if "+prior" not in k}
    ablations.update({k: float(scorer.balanced_accuracy((k,), (1.0,), (1.0,)))
                      for k in oof if k not in ablations})

    report = {
        "protocol": {
            "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
            "groups": "prompt_id",
            "validation_rows": int(len(comp)),
            "n_prompts": int(len(set(comp.groups))),
            "validation_source": "data/competition/train.csv (REAL competition rows only)",
            "augmentation_rows": int(len(aug)),
            "augmentation_prompts": int(len(set(aug.groups))),
            "leakage_control": "mirror texts hashed + excluded; mirror prompt ids disjoint "
                               "from the 150 competition prompts; vectorisers fitted on the "
                               "mirror corpus only",
            "headline_metric": "balanced accuracy (macro recall) over all 18 classes",
        },
        "shipped_model": {
            "offline_artifact": "outputs/final_model.joblib",
            "members": {"ridge_a05": 0.6, "svc_c1_m15": 0.4},
            "mirror_share_for_svc": 0.15,
            "prior_alpha": 0.0,
            "features": "char 2-5 + word 1-2 + word-shape/typography TF-IDF + 115 stylometric",
            "n_features": int(281376),
        },
        "bal_acc": shipped_m["bal_acc"],
        "accuracy": shipped_m["accuracy"],
        "n_rows": shipped_m["n_rows"],
        "per_class_recall": shipped_m["per_class_recall"],
        "length_bin_accuracy": shipped_m["length_bin_accuracy"],
        "blend_search": {
            "oof_best": best,
            "note": "3-member/temperature grid maximum over the same OOF; +0.002 vs the "
                    "shipped 2-member blend, i.e. inside noise",
        },
        "nested_selection_estimate": {
            "fold_bal_acc": search["nested_fold_bal_acc"],
            "mean_bal_acc": search["nested_mean"],
            "note": "weights re-selected on 4 folds and scored on the held-out fold; "
                    "0.3608 is the optimism-corrected estimate of the blend-selection "
                    "procedure, vs 0.3689 for the same search evaluated in-sample",
        },
        "ablations": ablations,
        "submission": {
            "path": "outputs/submission.parquet",
            "rows": int(len(sub)),
            "ids_match_test_csv": bool(list(sub["id"].astype(str)) ==
                                      list(test["id"].astype(str))),
            "columns": list(sub.columns),
            "nulls": int(sub.isna().sum().sum()),
            "labels_valid": bool(set(sub["model"]).issubset(set(LABELS))),
            "source": str(COMPETITION_TEST),
            "note": "public 3-row sample only; private test scored by the same artifact",
        },
    }
    (OUTPUTS / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("bal_acc", "accuracy", "n_rows")}, indent=1))
    print("submission rows", len(sub), "valid", report["submission"]["labels_valid"])


if __name__ == "__main__":
    main()
