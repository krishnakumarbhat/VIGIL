"""Track 05 n9 leg-2 anomaly drill-down.

Given stack_lenpriors_check showed the bin-permutation control does NOT collapse,
find the mechanism: compare prediction histograms / per-class recalls for
  A anchor 54-dim C=8
  B anchor 54-dim C=1000
  C V1 (real bins) C=32 m=10
  D control (permuted bins) C=32 m=10
  E pure-Gaussian-noise features C=32 (76-dim, noise replacing r+onehot)
  F noise C=8
If E/F reproduce ~0.5 the "gain" is an estimator-behavior channel (any extra
columns shift the fitted geometry), NOT bin-prior information -> the honest
takeaway is a different (possibly still real) mechanism that must be probed
directly, not a length-prior effect.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stack_lenpriors import (CLEAN, COMP, FOLDS, LABELS, NB, NC, S54, Y,  # noqa: E402
                             build_features, cross_fit_lr, log, score)


def run_tags(F: np.ndarray, C: float, name: str) -> dict:
    oof = cross_fit_lr(F, C)
    pred = oof.argmax(1)
    ba = balanced_accuracy_score(Y[CLEAN], pred[CLEAN])
    hist = np.bincount(pred[CLEAN], minlength=NC)
    recalls = []
    for c in range(NC):
        m = (Y[CLEAN] == c)
        recalls.append(float((pred[CLEAN][m] == c).mean()))
    log(f"{name} C={C} -> cleanBA {score(oof):.5f} "
        f"distinct_pred_classes={(hist > 0).sum()} "
        f"recall_min={min(recalls):.3f} recall_max={max(recalls):.3f}")
    return {"tag": name, "C": C, "clean_ba": float(ba),
            "pred_hist": [int(v) for v in hist],
            "per_class_recall": {l: round(r, 3) for l, r in zip(LABELS, recalls)}}


def main() -> dict:
    out = {}
    out["anchor_c8"] = run_tags(S54.copy(), 8.0, "anchor")
    out["anchor_c1000"] = run_tags(S54.copy(), 1000.0, "anchor_bigC")
    out["v1_c32"] = run_tags(build_features(10.0), 32.0, "v1_real_bins")

    mod = sys.modules["stack_lenpriors"]
    orig = mod.BIN.copy()
    mod.BIN = np.random.default_rng(11).permutation(orig)
    out["ctrl_c32"] = run_tags(build_features(10.0), 32.0, "ctrl_permuted_bins")
    mod.BIN = orig

    rng = np.random.default_rng(3)
    Noise = rng.standard_normal((len(COMP), NB + NC)) * 0.3
    Sn = np.hstack([S54, Noise]).astype(np.float64)
    out["noise_c32"] = run_tags(Sn, 32.0, "gaussian_noise")
    out["noise_c8"] = run_tags(Sn, 8.0, "gaussian_noise")

    json.dump(out, open("/tmp/opencode/lenprior_drill.json", "w"), indent=1)
    log("drill done")
    return out


if __name__ == "__main__":
    main()
