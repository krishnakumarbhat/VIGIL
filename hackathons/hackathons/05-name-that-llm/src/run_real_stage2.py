"""Stage-2 Track 05 experiments: C grid, feature-view ablations, char NB-SVM probe.

Everything here uses the same protocol as ``run_real_experiments.py``: real
competition ``train.csv`` only for validation, ``StratifiedGroupKFold(5,
shuffle=True, random_state=42)`` with groups ``prompt_id``, mirror rows for
training augmentation only.

Usage: ``python3 src/run_real_stage2.py``
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ntl_cv import GroupedCVRunner, MetricsCalculator  # noqa: E402
from ntl_data import (LABELS, LENGTH_BINS, AugmentationLoader, CompetitionLoader,  # noqa: E402
                      text_hashes)
from ntl_members import NBSVMMember, RidgeMember, SVCMember  # noqa: E402
from run_real_experiments import FeatureBundle, log_variant, make_fit_predict  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
OUTPUTS = TRACK / "outputs"

# (tag, views, member factory, mirror share)
VARIANTS = [
    ("svc_c2", ("char", "word", "shape", "sty"), lambda: SVCMember(C=2.0), 0.0),
    ("svc_c4", ("char", "word", "shape", "sty"), lambda: SVCMember(C=4.0), 0.0),
    ("ridge_a05", ("char", "word", "shape", "sty"), lambda: RidgeMember(alpha=0.5), 0.0),
    ("ridge_a2", ("char", "word", "shape", "sty"), lambda: RidgeMember(alpha=2.0), 0.0),
    ("svc_c1_m15", ("char", "word", "shape", "sty"), lambda: SVCMember(C=1.0), 0.15),
    ("svc_c1_wordshape", ("word", "shape", "sty"), lambda: SVCMember(C=1.0), 0.0),
    ("svc_c1_styonly", ("sty",), lambda: SVCMember(C=1.0), 0.0),
    ("nbsvm_char_c01", ("char",), lambda: NBSVMMember(C=0.1), 0.0),
]


def main() -> None:
    """Run the stage-2 grid and store OOF probabilities alongside stage 1."""
    comp = CompetitionLoader().load()
    aug = AugmentationLoader(max_per_class=900).load(exclude_hashes=text_hashes(comp))
    calc = MetricsCalculator(LABELS, LENGTH_BINS)
    runner = GroupedCVRunner(comp.y, comp.groups)
    bundles: dict[tuple, FeatureBundle] = {}
    oof_store: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}

    for tag, views, factory, share in VARIANTS:
        t0 = time.time()
        if views not in bundles:
            bundles[views] = FeatureBundle(views, comp, aug).build()
        bundle = bundles[views]
        res = runner.run(make_fit_predict(factory, bundle, share, share > 0), bundle.Xc)
        m = calc.score(res.proba, comp.y, comp.n_words)
        oof_store[tag] = res.proba
        scores[tag] = m["bal_acc"]
        log_variant("05-name-that-llm", f"bal_acc_real_{tag}", m["bal_acc"],
                    f"views={'+'.join(views)}; mirror_share={share}; n=2699 real rows; "
                    f"acc={m['accuracy']:.4f}; {time.time() - t0:.0f}s")
        print(f"{tag:24s} bal_acc={m['bal_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(OUTPUTS / "oof_real_stage2.npz", **oof_store)
    print("best stage2:", max(scores.items(), key=lambda kv: kv[1]), flush=True)


if __name__ == "__main__":
    main()
