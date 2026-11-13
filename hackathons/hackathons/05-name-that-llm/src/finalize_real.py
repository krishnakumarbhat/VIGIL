"""Select the final Track 05 blend, refit on all real rows, and write the submission.

Steps
-----
1. Load the OOF probabilities produced by ``run_real_experiments.py``.
2. Pick the blend weights / prior strength that maximise grouped-CV balanced
   accuracy over the REAL competition rows (selection is reported explicitly).
3. Refit every selected member on all 2,699 competition rows (+ weighted mirror
   rows when the winning variant used them).
4. Persist ``outputs/final_model.joblib`` and score ``data/competition/test.csv``
   into ``outputs/submission.parquet``; write ``outputs/cv_report.json``.

Usage: ``python3 src/finalize_real.py``
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ntl_cv import MetricsCalculator  # noqa: E402
from ntl_data import (LABELS, LENGTH_BINS, COMPETITION_TEST, AugmentationLoader,  # noqa: E402
                      CompetitionLoader, text_hashes)
from ntl_features import FeatureStack  # noqa: E402
from ntl_members import (CentroidMember, LogisticMember, NBSVMMember, PriorCorrector,  # noqa: E402
                         RidgeMember, SVCMember)
from ntl_submission import EnsembleModel, SubmissionWriter  # noqa: E402
from run_real_experiments import (FeatureBundle, TSV, log_variant, mixed_weights)  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
OUTPUTS = TRACK / "outputs"

# tag prefix -> (factory, mirror share) ; share is re-derived from the tag
FACTORIES = {
    "svc_c05": lambda: SVCMember(C=0.5),
    "svc_c1": lambda: SVCMember(C=1.0),
    "svc_c025": lambda: SVCMember(C=0.25),
    "svc_c05_t3": lambda: SVCMember(C=0.5, temperature=3.0),
    "svc_c2": lambda: SVCMember(C=2.0),
    "svc_c4": lambda: SVCMember(C=4.0),
    "svc_c1_m15": lambda: SVCMember(C=1.0),
    "svc_c1_wordshape": lambda: SVCMember(C=1.0),
    "svc_c1_styonly": lambda: SVCMember(C=1.0),
    "logreg_c2": lambda: LogisticMember(C=2.0),
    "ridge_a05": lambda: RidgeMember(alpha=0.5),
    "ridge_a1": lambda: RidgeMember(alpha=1.0),
    "ridge_a2": lambda: RidgeMember(alpha=2.0),
    "nbsvm_c01": lambda: NBSVMMember(C=0.1),
    "nbsvm_c03": lambda: NBSVMMember(C=0.3),
    "nbsvm_char_c01": lambda: NBSVMMember(C=0.1),
    "centroid": lambda: CentroidMember(),
}

FULL_VIEWS = ("char", "word", "shape", "sty")
VIEWS_BY_TAG = {
    "svc_c1_wordshape": ("word", "shape", "sty"),
    "svc_c1_styonly": ("sty",),
    "nbsvm_char_c01": ("char",),
}
SHARE_BY_TAG = {"svc_c1_m15": 0.15}


def parse_tag(tag: str) -> tuple[str, float]:
    """Split ``name@share`` (or ``name+priorA``) into a base variant and share."""
    if "+prior" in tag:
        tag = tag.split("+prior")[0]
    base, _, share = tag.partition("@")
    return base, float(share) if share else 0.0


def pick_blend(oof: dict[str, np.ndarray], y: np.ndarray, n_words: np.ndarray,
               calculator: MetricsCalculator) -> tuple[dict[str, float], float, dict]:
    """Grid-search blend weights over single-model OOFs plus prior strength."""
    singles = [k for k in oof if "+prior" not in k]
    ranked = sorted(singles, key=lambda k: calculator.score(oof[k], y, n_words)["bal_acc"],
                    reverse=True)
    pool = ranked[:4]
    best = {"bal_acc": -1.0, "weights": {}, "alpha": 0.0, "metrics": {}}
    grid = [w for w in np.round(np.arange(0.0, 1.01, 0.2), 2)]
    for k in pool:
        for alpha in (0.0, 0.15, 0.25, 0.35):
            p = oof[k]
            corr = PriorCorrector(alpha=alpha).fit(y)
            m = calculator.score(corr.transform(p), y, n_words)
            if m["bal_acc"] > best["bal_acc"]:
                best = {"bal_acc": m["bal_acc"], "weights": {k: 1.0},
                        "alpha": alpha, "metrics": m}
    for a, b in itertools.combinations(pool, 2):
        for wa in (0.6, 0.7, 0.8, 0.9):
            p = wa * oof[a] / oof[a].sum(1, keepdims=True) + \
                (1 - wa) * oof[b] / oof[b].sum(1, keepdims=True)
            for alpha in (0.0, 0.15, 0.25):
                corr = PriorCorrector(alpha=alpha).fit(y)
                m = calculator.score(corr.transform(p), y, n_words)
                if m["bal_acc"] > best["bal_acc"]:
                    best = {"bal_acc": m["bal_acc"], "weights": {a: wa, b: 1 - wa},
                            "alpha": alpha, "metrics": m}
    return best["weights"], best["alpha"], best["metrics"]


def fit_member(base: str, share: float, bundle: FeatureBundle, comp, prior_alpha: float,
               prior: np.ndarray) -> object:
    """Fit one member on all competition rows (+ weighted mirror rows)."""
    from ntl_members import balanced_weights
    if share > 0:
        import scipy.sparse as sp
        sw_c, sw_a = mixed_weights(comp.y, bundle.aug.y, share)
        X = sp.vstack([bundle.Xc, bundle.Xa], format="csr")
        y = np.concatenate([comp.y, bundle.aug.y])
        sw = np.concatenate([sw_c, sw_a])
    else:
        X, y, sw = bundle.Xc, comp.y, balanced_weights(comp.y)
    member = FACTORIES[base]()
    member.fit(X, y, sw)
    return member


def infer_only() -> None:
    """Score ``$NTL_TEST_CSV`` (default competition test.csv) with the saved artefact."""
    test_path = Path(os.environ.get("NTL_TEST_CSV", COMPETITION_TEST))
    model = joblib.load(OUTPUTS / "final_model.joblib")
    test = pd.read_csv(test_path)
    writer = SubmissionWriter()
    sub = writer.build(test, model)
    out = Path(os.environ.get("NTL_OUT_DIR", OUTPUTS)) / "submission.parquet"
    writer.write(sub, out)
    print(f"scored {test_path} -> {out} ({len(sub)} rows)")


def main() -> None:
    """Run selection, refit, persist artefacts and write the public submission."""
    t0 = time.time()
    if "--infer-only" in sys.argv:
        infer_only()
        return
    comp = CompetitionLoader().load()
    aug = AugmentationLoader(max_per_class=900).load(exclude_hashes=text_hashes(comp))
    bundle = FeatureBundle(FULL_VIEWS, comp, aug).build()
    print(f"features ready {bundle.Xc.shape} ({time.time() - t0:.0f}s)", flush=True)

    oof: dict[str, np.ndarray] = {}
    for name in ("oof_real.npz", "oof_real_stage2.npz"):
        path = OUTPUTS / name
        if path.exists():
            oof.update({k: v for k, v in np.load(path).items() if not k.endswith("+prior")})
    calc = MetricsCalculator(LABELS, LENGTH_BINS)
    # blend pool = variants on the full feature set (single shared stack in the artefact)
    oof = {k: v for k, v in oof.items()
           if parse_tag(k)[0] in FACTORIES
           and VIEWS_BY_TAG.get(parse_tag(k)[0], FULL_VIEWS) == FULL_VIEWS}
    weights, alpha, metrics = pick_blend(oof, comp.y, comp.n_words, calc)
    tag = "blend(" + " + ".join(f"{w:.2f}*{k}" for k, w in weights.items()) + f") prior={alpha}"
    print("selected:", tag, "bal_acc=", round(metrics["bal_acc"], 4), flush=True)
    log_variant("05-name-that-llm", "bal_acc_real_final_blend", metrics["bal_acc"],
                f"{tag}; selected on real-train OOF; acc={metrics['accuracy']:.4f}")

    prior = np.bincount(comp.y, minlength=len(LABELS)).astype(np.float64)
    prior /= prior.sum()
    members = []
    for name, w in weights.items():
        base, share = parse_tag(name)
        share = max(share, SHARE_BY_TAG.get(base, 0.0))
        members.append((base, fit_member(base, share, bundle, comp, alpha, prior), w))
        print(f"  refit {name} (mirror share {share}) {time.time() - t0:.0f}s", flush=True)

    model = EnsembleModel(bundle.stack, members, prior_alpha=alpha, prior=prior)
    joblib.dump(model, OUTPUTS / "final_model.joblib", compress=3)
    print("saved final_model.joblib", flush=True)

    writer = SubmissionWriter()
    test = pd.read_csv(COMPETITION_TEST)
    sub = writer.build(test, model)
    writer.write(sub, OUTPUTS / "submission.parquet")
    print(sub.to_string(), flush=True)

    report = {
        "protocol": {
            "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
            "groups": "prompt_id",
            "validation_rows": int(len(comp)),
            "n_prompts": int(len(set(comp.groups))),
            "validation_source": "data/competition/train.csv (REAL competition rows)",
            "augmentation_rows": int(len(aug)),
            "augmentation_prompts": int(len(set(aug.groups))),
            "leakage_control": "mirror texts hashed and excluded; mirror prompt ids disjoint",
            "vectorisers_fitted_on": "mirror corpus only (never validation text)",
            "headline_metric": "balanced accuracy (macro recall)",
        },
        "bal_acc": metrics["bal_acc"],
        "accuracy": metrics["accuracy"],
        "n_rows": metrics["n_rows"],
        "per_class_recall": metrics["per_class_recall"],
        "length_bin_accuracy": metrics["length_bin_accuracy"],
        "final_blend": {"weights": weights, "prior_alpha": alpha},
        "variants": {k: v["bal_acc"] for k, v in
                     json.loads((OUTPUTS / "real_variant_scores.json").read_text()).items()},
        "submission": {
            "rows": int(len(sub)),
            "source": str(COMPETITION_TEST),
            "note": "public sample test.csv (3 rows); regenerate with private test via "
                    "python3 src/finalize_real.py or notebooks/inference_submission.ipynb",
        },
    }
    (OUTPUTS / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
