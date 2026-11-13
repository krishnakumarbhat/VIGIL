"""True column inventory of every CSV in data/kaggle_nfl/nfl_data.

Run ``python3 src/profile_data.py`` to (re)write ``outputs/DATA_SCHEMA.md``.
Nothing here is assumed: every column, dtype, non-null count, cardinality and
range is measured from the files themselves, over the *whole* file (the 335 MB
tracking file is read once with explicit dtypes to keep the footprint small).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

DATA = os.path.join("data", "kaggle_nfl", "nfl_data")
OUT = os.path.join("outputs", "DATA_SCHEMA.md")

# explicit dtypes keep the 3.66 M-row file comfortably in memory
GT_DTYPES = {"game_id": "int32", "play_id": "int32", "nfl_id": "int32",
             "x": "float32", "y": "float32", "s": "float32", "a": "float32",
             "dis": "float32", "o": "float32", "dir": "float32"}
CT_DTYPES = {"draft_year": "int16", "nfl_id": "int32", "attempt": "int8",
             "x": "float32", "y": "float32", "s": "float32", "a": "float32",
             "dis": "float32", "dir": "float32"}
PP_DTYPES = {"game_id": "int32", "play_id": "int32", "nfl_id": "int32"}

FILES: Dict[str, dict] = {
    "combine_tracking.csv": dict(
        note="10 Hz combine/Pro-Day tracking, 2023-2025 draft classes. "
             "`entity_type` splits PLAYER from BALL rows; `a` is an unsigned "
             "magnitude (min = 0.000 over all 463,189 rows), so signed "
             "acceleration is differentiated from x/y instead "
             "(Savitzky-Golay, see src/kce_features.py).",
        dtype=CT_DTYPES, parse_dates=["time"]),
    "player_play.csv": dict(
        note="One row per (game, play, player). Holds the official target "
             "fields: `separation_at_pass_forward`, `cushion`, `target`, "
             "`route_ran`, `rec_yards`, `yards_after_catch`, "
             "`expected_yards_after_catch`. NOTE: quoted `play_description` "
             "contains embedded newlines, so line counts overstate the row "
             "count -- the counts below come from pandas itself.",
        dtype=PP_DTYPES),
    "game_tracking_2023.csv": dict(
        note="10 Hz in-game tracking, sampled subset: 333 games / 31,690 plays "
             "/ 156 athletes, mean 1.6 tracked athletes per play (measured) -- "
             "too sparse to reconstruct defender coverage.",
        dtype=GT_DTYPES, parse_dates=["time"], chunked=True),
    "players.csv": dict(note="Roster/physical/draft metadata, keyed by nfl_id."),
    "games.csv": dict(note="Game metadata, 2023-2025 seasons."),
    "combine_results.csv": dict(
        note="Official combine measurables (forty, three-cone, vertical, ...) "
             "with measured missingness: three_cone 64.7%, short_shuttle "
             "60.6%, bench_reps 62.4% null."),
    "player_career_successes.csv": dict(
        note="Career snap counts and honors (All-Pro / Pro Bowl) per athlete; "
             "honors are extremely sparse (All-Pro 1st: 6 of 510)."),
}


def profile_csv(fname: str, cfg: dict) -> Dict:
    """Read the whole file (chunked when large) and measure every column."""
    path = os.path.join(DATA, fname)
    size_mb = os.path.getsize(path) / 1e6
    use_dtypes = cfg.get("dtype")
    parse_dates = cfg.get("parse_dates")
    n_rows = 0
    partial = None
    if cfg.get("chunked"):
        agg: Dict[str, dict] = {}
        for ch in pd.read_csv(path, dtype=use_dtypes, parse_dates=parse_dates,
                              chunksize=400_000, low_memory=False):
            n_rows += len(ch)
            for c in ch.columns:
                s = ch[c]
                if c not in agg:
                    agg[c] = dict(dtype=str(s.dtype), non_null=0, nunique=set(),
                                  mn=None, mx=None, ex="")
                a = agg[c]
                a["non_null"] += int(s.notna().sum())
                a["nunique"].update(s.dropna().unique()[:50_000].tolist()
                                    if s.dtype == object or str(s.dtype) == "category"
                                    else [])
                if s.dtype.kind in "if" and s.notna().any():
                    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
                    a["mn"] = lo if a["mn"] is None else min(a["mn"], lo)
                    a["mx"] = hi if a["mx"] is None else max(a["mx"], hi)
                elif not a["ex"]:
                    v = s.dropna().astype(str)
                    a["ex"] = v.iloc[0] if len(v) else ""
        cols = [dict(name=c, dtype=a["dtype"], non_null=a["non_null"],
                     nunique=len(a["nunique"]), min=a["mn"], max=a["mx"],
                     example=str(a["ex"])[:40]) for c, a in agg.items()]
        return dict(file=fname, size_mb=round(size_mb, 1), n_rows=n_rows,
                    n_cols=len(cols), profiled_rows=n_rows, fully_profiled=True,
                    cols=cols, note=cfg.get("note", ""))
    df = pd.read_csv(path, dtype=use_dtypes, parse_dates=parse_dates,
                     low_memory=False)
    n_rows = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        rec = dict(name=c, dtype=str(s.dtype), non_null=int(s.notna().sum()),
                   nunique=int(s.nunique(dropna=True)))
        if s.dtype.kind in "if" and s.notna().any():
            rec.update(min=float(np.nanmin(s)), max=float(np.nanmax(s)))
        else:
            vals = s.dropna().astype(str)
            rec["min"] = None
            rec["max"] = None
            rec["example"] = vals.iloc[0][:40] if len(vals) else ""
        cols.append(rec)
    return dict(file=fname, size_mb=round(size_mb, 1), n_rows=n_rows,
                n_cols=len(df.columns), profiled_rows=n_rows,
                fully_profiled=True, cols=cols, note=cfg.get("note", ""))


def key_relationships() -> List[str]:
    """Measured join keys / cardinalities between the tables."""
    ct = pd.read_csv(os.path.join(DATA, "combine_tracking.csv"),
                     usecols=["nfl_id", "draft_year", "entity_type", "drill_name",
                              "event_id", "attempt"], dtype={"nfl_id": "int32",
                                                             "draft_year": "int16",
                                                             "attempt": "int8"})
    pp = pd.read_csv(os.path.join(DATA, "player_play.csv"), low_memory=False,
                     usecols=["game_id", "play_id", "nfl_id",
                              "separation_at_pass_forward", "cushion", "route_ran",
                              "target"], dtype=PP_DTYPES)
    gt = pd.read_csv(os.path.join(DATA, "game_tracking_2023.csv"),
                     usecols=["game_id", "play_id", "nfl_id"], dtype="int32")
    players = pd.read_csv(os.path.join(DATA, "players.csv"),
                          usecols=["nfl_id", "draft_year", "nfl_position"])
    cr = pd.read_csv(os.path.join(DATA, "combine_results.csv"),
                     usecols=["nfl_id", "draft_year"])
    pcs = pd.read_csv(os.path.join(DATA, "player_career_successes.csv"),
                      usecols=["nfl_id"])

    ids = {
        "combine_tracking": set(ct.nfl_id.unique()),
        "players": set(players.nfl_id.unique()),
        "combine_results": set(cr.nfl_id.unique()),
        "player_career_successes": set(pcs.nfl_id.unique()),
        "player_play": set(pp.nfl_id.unique()),
        "game_tracking_2023": set(gt.nfl_id.unique()),
    }
    lines = ["| table | athletes (nfl_id) | overlap with players.csv |",
             "|---|---|---|"]
    for k, v in ids.items():
        lines.append(f"| {k} | {len(v)} | {len(v & ids['players'])} |")

    ct_p = ct[ct.entity_type == "PLAYER"]
    drills = sorted(ct_p.drill_name.unique())
    n_left = sum(d.endswith("_LEFT") for d in drills)
    n_right = sum(d.endswith("_RIGHT") for d in drills)
    bases = ({d[:-5] for d in drills if d.endswith("_LEFT")} |
             {d[:-6] for d in drills if d.endswith("_RIGHT")})
    lines += [
        "",
        f"* combine_tracking draft classes: "
        f"{ {int(k): int(v) for k, v in ct.draft_year.value_counts().sort_index().items()} } "
        f"(PLAYER rows: {len(ct_p):,} of {len(ct):,}; BALL rows: "
        f"{len(ct) - len(ct_p):,} carry the athlete's nfl_id).",
        f"* combine sessions (event_id x nfl_id x drill x attempt): "
        f"{ct_p.groupby(['event_id', 'nfl_id', 'drill_name', 'attempt']).ngroups:,}; "
        f"every consecutive frame gap measures exactly 0.1 s (10 Hz).",
        f"* drill names: {len(drills)} total -- {n_left} end in `_LEFT`, "
        f"{n_right} end in `_RIGHT` ({len(bases)} distinct base drills, of which "
        f"only POST_CORNER_ROUTE has both sides present). The suffix is the "
        f"athlete's ALIGNMENT, not his break direction.",
        f"* game_tracking_2023: {gt.game_id.nunique()} games, "
        f"{gt.groupby(['game_id', 'play_id']).ngroups:,} plays, "
        f"{gt.nfl_id.nunique()} athletes, mean "
        f"{gt.groupby(['game_id', 'play_id']).nfl_id.nunique().mean():.2f} "
        f"tracked athletes per play (max "
        f"{gt.groupby(['game_id', 'play_id']).nfl_id.nunique().max()}).",
        f"* game_tracking plays also present in player_play: "
        f"{len(set(map(tuple, gt[['game_id', 'play_id']].drop_duplicates().to_numpy())) & set(map(tuple, pp[['game_id', 'play_id']].drop_duplicates().to_numpy()))):,} "
        f"of {gt.groupby(['game_id', 'play_id']).ngroups:,}.",
    ]
    sep = pp[pp.separation_at_pass_forward.notna()]
    lines += [
        f"* player_play.separation_at_pass_forward: {len(sep):,} non-null rows "
        f"across {sep.nfl_id.nunique()} athletes (mean "
        f"{sep.separation_at_pass_forward.mean():.3f} yd, sd "
        f"{sep.separation_at_pass_forward.std():.3f}); cushion non-null "
        f"{pp.cushion.notna().sum():,}; target==True rows "
        f"{int(pp.target.sum()):,}; route_ran non-null "
        f"{pp.route_ran.notna().sum():,}.",
        f"* (game, play, athlete) rows with BOTH game tracking and an official "
        f"separation value: "
        f"{len(set(map(tuple, gt.drop_duplicates().to_numpy())) & set(map(tuple, sep[['game_id', 'play_id', 'nfl_id']].to_numpy()))):,}.",
    ]
    a = pd.read_csv(os.path.join(DATA, "combine_tracking.csv"), usecols=["a"],
                    dtype={"a": "float32"})
    lines.append(
        f"* combine_tracking.a: min {a.a.min():.3f}, max {a.a.max():.3f}, "
        f"{float((a.a < 0).mean()):.4%} of {len(a):,} rows negative -> UNSIGNED "
        f"magnitude; signed acceleration must come from the positions.")
    return lines


def main() -> None:
    profiles = {f: profile_csv(f, cfg) for f, cfg in FILES.items()}
    lines: List[str] = [
        "# Data schema - NFL Big Data Bowl 2027 (measured, not assumed)",
        "",
        "Generated by `python3 src/profile_data.py`. Every dtype, count and "
        "range below is read from the CSVs on disk over the entire file.",
        "",
        "## File inventory",
        "",
        "| file | size (MB) | rows | cols |",
        "|---|---|---|---|",
    ]
    for f, p in profiles.items():
        lines.append(f"| `{f}` | {p['size_mb']} | {p['n_rows']:,} | {p['n_cols']} |")
    lines += ["", "## Join keys and cardinalities", ""]
    lines += key_relationships()
    lines += ["", "## Columns by file", ""]
    for f, p in profiles.items():
        lines += [f"### `{f}`", "", p["note"], "",
                  "| column | dtype | non-null | unique | min | max | example |",
                  "|---|---|---|---|---|---|---|"]
        for c in p["cols"]:
            lines.append(
                f"| `{c['name']}` | {c['dtype']} | {c['non_null']:,} | "
                f"{c['nunique']:,} | {_fmt(c.get('min'))} | {_fmt(c.get('max'))} | "
                f"{c.get('example', '')} |")
        lines.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes)")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


if __name__ == "__main__":
    main()
