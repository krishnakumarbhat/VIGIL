"""Grouped-CV experiments on the REAL competition train.csv (Track 05).

Protocol
--------
* rows   : ``data/competition/train.csv`` (2,699 rows / 150 prompts / 18 models)
* split  : ``StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)``
* groups : ``prompt_id``  -> no prompt is ever in both train and validation
* metric : balanced accuracy (macro recall), the competition metric
* mirror : ``data/compact/external.parquet`` + extra-Claude passages, hashed and
           excluded against every competition text, used for TRAINING only

Usage
-----
    python3 src/run_real_experiments.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ntl_cv import GroupedCVRunner, MetricsCalculator  # noqa: E402
from ntl_data import (LABELS, LENGTH_BINS, WORD_COL, AugmentationLoader,  # noqa: E402
                      CompetitionLoader, text_hashes)
from ntl_features import FeatureStack  # noqa: E402
from ntl_members import (CentroidMember, LogisticMember, NBSVMMember,  # noqa: E402
                         PriorCorrector, RidgeMember, SVCMember, balanced_weights)

TRACK = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/opencode/ntl_real")
OUTPUTS = TRACK / "outputs"
LOGS = TRACK / "logs"
TSV = LOGS / "experiments.tsv"


def log_variant(track: str, metric: str, value: float, notes: str) -> None:
    """Append one row to ``logs/experiments.tsv`` (creating the header)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    new = not TSV.exists()
    with TSV.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("track\tmetric\tvalue\tnotes\n")
        fh.write(f"{track}\t{metric}\t{value:.5f}\t{notes}\n")


class FeatureBundle:
    """Fitted feature stack plus the competition/augmentation design matrices."""

    def __init__(self, views: tuple[str, ...], comp, aug, cache_dir: Path = CACHE) -> None:
        self.views = views
        self.comp = comp
        self.aug = aug
        tag = "-".join(views)
        self.cache = cache_dir / tag
        self.stack: FeatureStack | None = None
        self.Xc: sp.csr_matrix
        self.Xa: sp.csr_matrix

    def build(self) -> "FeatureBundle":
        """Fit the stack on augmentation text only, then transform both corpora."""
        self.cache.mkdir(parents=True, exist_ok=True)
        stack_path = self.cache / "stack.joblib"
        if stack_path.exists():
            self.stack = joblib.load(stack_path)
            self.Xc = sp.load_npz(self.cache / "Xc.npz")
            self.Xa = sp.load_npz(self.cache / "Xa.npz")
            return self
        t0 = time.time()
        self.stack = FeatureStack(views=self.views).fit(list(self.aug.texts))
        print(f"  [fit {self.views}] {time.time() - t0:.1f}s", flush=True)
        self.Xc = self.stack.transform(list(self.comp.texts))
        self.Xa = self.stack.transform(list(self.aug.texts))
        joblib.dump(self.stack, stack_path, compress=0)
        sp.save_npz(self.cache / "Xc.npz", self.Xc)
        sp.save_npz(self.cache / "Xa.npz", self.Xa)
        return self


def mixed_weights(y_comp: np.ndarray, y_aug: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights so each class puts ``share`` of its mass on augmentation rows."""
    sw_c = np.zeros(len(y_comp))
    sw_a = np.zeros(len(y_aug))
    for c in range(len(LABELS)):
        mc = y_comp == c
        ma = y_aug == c
        if mc.any():
            sw_c[mc] = (1.0 - share) / mc.sum()
        if ma.any() and share > 0:
            sw_a[ma] = share / ma.sum()
    scale = len(LABELS)
    return sw_c * scale, sw_a * scale


def make_fit_predict(member_factory, bundle: FeatureBundle, share: float, use_aug: bool):
    """Return a ``fit_predict(Xtr, ytr, Xva, fold)`` closure for one member."""

    def fit_predict(Xtr, ytr, Xva, fold):
        if use_aug and share > 0:
            sw_c, sw_a = mixed_weights(ytr, bundle.aug.y, share)
            X = sp.vstack([Xtr, bundle.Xa], format="csr")
            y = np.concatenate([ytr, bundle.aug.y])
            sw = np.concatenate([sw_c, sw_a])
        else:
            X, y, sw = Xtr, ytr, balanced_weights(ytr)
        member = member_factory()
        member.fit(X, y, sw)
        if use_aug and share > 0:
            del X
        return member.proba(Xva)

    return fit_predict


def main() -> None:
    """Run every registered variant and persist OOF matrices + metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, nargs="*", default=[0.0, 0.25, 0.4])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    comp = CompetitionLoader().load()
    aug = AugmentationLoader(max_per_class=900).load(exclude_hashes=text_hashes(comp))
    print(f"competition rows={len(comp)} prompts={len(set(comp.groups))}", flush=True)
    print(f"augmentation rows={len(aug)} prompts={len(set(aug.groups))}", flush=True)

    metrics = MetricsCalculator(LABELS, LENGTH_BINS)
    runner = GroupedCVRunner(comp.y, comp.groups)
    base_views = ("char", "word", "shape", "sty")
    bundle = FeatureBundle(base_views, comp, aug).build()

    # member registry: name -> (factory, uses augmentation by default)
    registry: dict[str, object] = {
        "svc_c05": lambda: SVCMember(C=0.5),
        "svc_c1": lambda: SVCMember(C=1.0),
        "svc_c025": lambda: SVCMember(C=0.25),
        "svc_c05_t3": lambda: SVCMember(C=0.5, temperature=3.0),
        "logreg_c2": lambda: LogisticMember(C=2.0),
        "ridge_a1": lambda: RidgeMember(alpha=1.0),
        "nbsvm_c01": lambda: NBSVMMember(C=0.1),
        "nbsvm_c03": lambda: NBSVMMember(C=0.3),
        "centroid": lambda: CentroidMember(),
    }

    oof_store: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    for name, factory in registry.items():
        for share in ([0.0, 0.4] if name.startswith(("svc_c05", "nbsvm_c01")) else [0.0]):
            if args.quick and share != 0.0:
                continue
            t0 = time.time()
            tag = f"{name}@{share:.2f}"
            use_aug = share > 0
            res = runner.run(make_fit_predict(factory, bundle, share, use_aug), bundle.Xc)
            sc = metrics.score(res.proba, comp.y, comp.n_words)
            oof_store[tag] = res.proba
            results[tag] = sc
            log_variant("05-name-that-llm", f"bal_acc_real_{tag}", sc["bal_acc"],
                        f"real comp train.csv n={len(comp)}; views={'+'.join(base_views)}; "
                        f"mirror_share={share}; acc={sc['accuracy']:.4f}; {time.time() - t0:.0f}s")
            print(f"{tag:22s} bal_acc={sc['bal_acc']:.4f} acc={sc['accuracy']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # prior correction on the best single member
    best_single = max(results.items(), key=lambda kv: kv[1]["bal_acc"])[0]
    for alpha in (0.15, 0.25, 0.4):
        corr = PriorCorrector(alpha=alpha).fit(comp.y)
        p = corr.transform(oof_store[best_single])
        sc = metrics.score(p, comp.y, comp.n_words)
        tag = f"{best_single}+prior{alpha}"
        oof_store[tag] = p
        results[tag] = sc
        log_variant("05-name-that-llm", f"bal_acc_real_{tag}", sc["bal_acc"],
                    f"prior correction alpha={alpha} on {best_single}; acc={sc['accuracy']:.4f}")
        print(f"{tag:22s} bal_acc={sc['bal_acc']:.4f}", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUTS / "oof_real.npz", **oof_store)
    with (OUTPUTS / "real_variant_scores.json").open("w") as fh:
        json.dump(results, fh, indent=1)
    print("best:", max(results.items(), key=lambda kv: kv[1]["bal_acc"])[0], flush=True)


if __name__ == "__main__":
    main()
