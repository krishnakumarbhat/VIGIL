"""Stage 1 of the real-data pipeline: build and cache every derived table.

Run: ``python3 src/build_tables.py``

Produces (all from the REAL Kaggle CSVs, never synthetic):
  outputs/cache/combine_sessions.csv   one row per drill attempt (6,310)
  outputs/cache/combine_cuts.csv       one row per detected cut
  outputs/cache/combine_player.csv     one row per athlete (510) with KCE/DBR
  outputs/cache/play_features.csv      one row per (game, play, tracked athlete)
  outputs/cache/derived_sep.csv        tracking-derived separation (proxy check)

Position-derived kinematics use Savitzky-Golay differentiation throughout; no
quantity is taken from the unsigned `a` column.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kce_features import (  # noqa: E402
    aggregate_player_features,
    build_combine_tables,
    build_play_features,
    derive_separation,
    side_split_features,
)

DATA = os.path.join("data", "kaggle_nfl", "nfl_data")
CACHE = os.path.join("outputs", "cache")

GT_DTYPES = {"game_id": "int32", "play_id": "int32", "nfl_id": "int32",
             "x": "float32", "y": "float32", "s": "float32", "a": "float32",
             "dis": "float32", "o": "float32", "dir": "float32", "event": "category"}
PP_DTYPES = {"game_id": "int32", "play_id": "int32", "nfl_id": "int32"}


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    force = "--force" in sys.argv

    ct = pd.read_csv(os.path.join(DATA, "combine_tracking.csv"))
    ct = ct[ct.entity_type == "PLAYER"].copy()
    log(f"combine_tracking loaded ({len(ct):,} PLAYER rows)", t0)
    tab = build_combine_tables(ct)
    sessions, cuts = tab["sessions"], tab["cuts"]
    player = aggregate_player_features(sessions, cuts)
    player = side_split_features(player, cuts)
    sessions.to_csv(os.path.join(CACHE, "combine_sessions.csv"), index=False)
    cuts.to_csv(os.path.join(CACHE, "combine_cuts.csv"), index=False)
    player.to_csv(os.path.join(CACHE, "combine_player.csv"), index=False)
    log(f"sessions={len(sessions):,} cuts={len(cuts):,} players={len(player):,}", t0)

    need = ["play_features.csv", "derived_sep.csv"]
    if force or any(not os.path.exists(os.path.join(CACHE, f)) for f in need):
        gt = pd.read_csv(os.path.join(DATA, "game_tracking_2023.csv"), dtype=GT_DTYPES)
        pp = pd.read_csv(os.path.join(DATA, "player_play.csv"), low_memory=False,
                         dtype=PP_DTYPES)
        log(f"game_tracking loaded ({len(gt):,} rows) + player_play ({len(pp):,})", t0)
        pf = build_play_features(gt, pp)
        pf.to_csv(os.path.join(CACHE, "play_features.csv"), index=False)
        log(f"play_features={len(pf):,} rows", t0)
        dv = derive_separation(gt, pp)
        dv.to_csv(os.path.join(CACHE, "derived_sep.csv"), index=False)
        log(f"derived_sep={len(dv):,} rows", t0)
    else:
        log("play_features/derived_sep cache reused (pass --force to rebuild)", t0)
    log("done", t0)


if __name__ == "__main__":
    main()
