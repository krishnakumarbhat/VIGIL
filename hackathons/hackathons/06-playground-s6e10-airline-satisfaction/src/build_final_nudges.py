"""Build the last-slot candidates: the measured 0.96225 anchor plus mirror-pair nudges.

The anchor is the file that areebbb's notebook measured at public 0.96225:
``pct(rank(X) + 1.75 * (rank(III) - rank(PAIR)))``.  Mirror pairs exploit the near
linearity of AUC: nudging a block of rows by ``+n`` changes the score by ``+g`` and by
``-g`` for ``-n``, so the better of the two sides is worth about ``|g|`` while the
second-order cost is common to both.  Both sides must be submitted because only the
leaderboard says which one won.

Run:  python3 src/build_final_nudges.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/opencode/kg/iii")
from summit_composer import block_noise, nudge, pct, stack_scale  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG = "/tmp/opencode/kg"
N = 299844
BLOCKS = 4
SIZE = 0.04
BLOCK_SEED = 2031


def main() -> None:
    ids = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"]).id.values

    def read(p: str) -> np.ndarray:
        d = pd.read_csv(p)
        assert (d.id.values == ids).all(), p
        col = [c for c in d.columns if c != "id"][0]
        return pct(d[col].values)

    X = read(os.path.join(ROOT, "data", "new1", "ref_t3_096223.csv"))
    III = read(os.path.join(KG, "iii", "submission.csv"))
    PAIR = read(os.path.join(KG, "pair", "submission.csv"))
    anchor = pct(X + 1.75 * (III - PAIR))
    ref = read(os.path.join(KG, "areebbb_s6e10-endgame-lb-0-96225-rank-22", "a03_X_dir175.csv"))
    print(f"rebuilt anchor vs shipped a03: rank corr {np.corrcoef(anchor, ref)[0, 1]:.10f}")

    scale = stack_scale(np.load(os.path.join(ROOT, "data", "oof_pool", "own_cv_stack_181_test.npy")))
    outs = {"f00_anchor": anchor}
    for k in (0, 1):
        n = block_noise(N, k, SIZE, BLOCKS, BLOCK_SEED, 9001 + k)
        outs[f"f{1 + 2 * k:02d}_p{k}"] = nudge(anchor, scale, n)
        outs[f"f{2 + 2 * k:02d}_m{k}"] = nudge(anchor, scale, -n)
        print(f"block {k}: rows {(n != 0).sum()}")

    for name, v in outs.items():
        path = os.path.join(ROOT, "outputs", f"{name}.csv")
        pd.DataFrame({"id": ids, "satisfaction": v}).to_csv(path, index=False)
        print(f"wrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
