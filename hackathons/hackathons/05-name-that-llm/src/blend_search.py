"""Blend search on saved OOF probabilities, with a nested-CV honesty check.

Free parameters are the member blend weights and an optional per-member
temperature (``p ** (1/T)`` renormalised).  A plain grid maximum over OOF is
optimistic, so the same search is also run leave-one-fold-out: select on 4 folds,
score the held-out fold, then average.

The scorer is a vectorised confusion-count recall (balanced accuracy) so the
grid stays in the seconds range.

Usage: ``python3 src/blend_search.py``
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ntl_cv import grouped_folds  # noqa: E402
from ntl_data import LABELS, CompetitionLoader  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
OUTPUTS = TRACK / "outputs"
STEPS = np.round(np.arange(0.0, 1.01, 0.1), 3)
TEMPS = (0.7, 1.0, 1.4)
K = len(LABELS)


def load_oof() -> dict[str, np.ndarray]:
    """Merge stage-1 and stage-2 OOF dictionaries, dropping prior variants."""
    out: dict[str, np.ndarray] = {}
    for name in ("oof_real.npz", "oof_real_stage2.npz"):
        path = OUTPUTS / name
        if path.exists():
            out.update({k: v for k, v in np.load(path).items() if "+prior" not in k})
    return out


class FastScorer:
    """Vectorised macro-recall scorer over cached temperature variants."""

    def __init__(self, oof: dict[str, np.ndarray], y: np.ndarray) -> None:
        self.y = y
        self.class_counts = np.bincount(y, minlength=K).astype(np.float64)
        self.cache = {
            (name, t): self._temper(p, t)
            for name, p in oof.items() for t in TEMPS
        }

    @staticmethod
    def _temper(p: np.ndarray, t: float) -> np.ndarray:
        """Sharpen/flatten a probability matrix and renormalise."""
        q = np.power(np.clip(p, 1e-9, None), 1.0 / t)
        return q / q.sum(1, keepdims=True)

    def balanced_accuracy(self, combo: tuple[str, ...], weights: tuple[float, ...],
                          temps: tuple[float, ...], rows: np.ndarray | None = None) -> float:
        """Macro recall of the blended argmax (optionally restricted to ``rows``)."""
        p = None
        for name, w, t in zip(combo, weights, temps):
            term = self.cache[(name, t)] * w
            p = term if p is None else p + term
        idx = slice(None) if rows is None else rows
        pred = p[idx].argmax(1)
        y = self.y[idx]
        cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
        recalls = np.diag(cm) / np.maximum(cm.sum(1), 1)
        return float(recalls.mean())


class BlendSearcher:
    """Grid search over (member subset, weights, temperatures)."""

    def __init__(self, scorer: FastScorer) -> None:
        self.scorer = scorer

    def search(self, pool: list[str], rows: np.ndarray | None = None,
               max_members: int = 3) -> tuple[dict, float]:
        """Return the best configuration and its score."""
        best = {"combo": (), "weights": (), "temps": ()}
        best_score = -1.0
        for size in range(1, max_members + 1):
            for combo in itertools.combinations(pool, size):
                for weights in itertools.product(STEPS, repeat=size):
                    if abs(sum(weights) - 1.0) > 1e-6:
                        continue
                    for temps in itertools.product(TEMPS, repeat=size):
                        s = self.scorer.balanced_accuracy(combo, weights, temps, rows)
                        if s > best_score:
                            best_score, best = s, {"combo": combo, "weights": weights,
                                                   "temps": temps}
        return best, best_score


def main() -> None:
    """Run the OOF grid plus a leave-one-fold-out estimate of the grid's optimism."""
    comp = CompetitionLoader().load()
    oof = load_oof()
    scorer = FastScorer(oof, comp.y)
    ranked = sorted(oof, key=lambda k: scorer.balanced_accuracy((k,), (1.0,), (1.0,)),
                    reverse=True)
    pool = ranked[:6]
    best, score = BlendSearcher(scorer).search(pool)
    print(f"OOF-best {best['combo']} w={best['weights']} T={best['temps']} -> {score:.4f}",
          flush=True)

    folds = grouped_folds(comp.y, comp.groups)
    nested = []
    for tr, va in folds:
        cfg, _ = BlendSearcher(scorer).search(pool, rows=tr)
        nested.append(scorer.balanced_accuracy(cfg["combo"], cfg["weights"], cfg["temps"], va))
    print("nested folds", np.round(nested, 4), "mean", round(float(np.mean(nested)), 4),
          flush=True)

    payload = {
        "best": {"combo": list(best["combo"]), "weights": list(best["weights"]),
                 "temps": list(best["temps"]), "oof_bal_acc": score},
        "nested_fold_bal_acc": [float(x) for x in nested],
        "nested_mean": float(np.mean(nested)),
        "pool": pool,
    }
    (OUTPUTS / "blend_search.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
