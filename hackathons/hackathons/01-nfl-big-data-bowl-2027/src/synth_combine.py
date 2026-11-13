"""Schema-compatible synthetic generator for NFL BDB 2027 combine tracking.

Produces realistic 10 Hz combine sensor tracking so the KCE feature/model
pipeline is fully testable when the official Kaggle data (403-blocked for our
token; rules not accepted) is unavailable. No public mirror containing
2027 combine tracking was found (HF "combine tracking" search empty;
GitHub code search requires auth). The generator is NOT a substitute for
Kaggle data -- it only enables offline development and unit testing.

SCHEMA (combine_tracking.csv, per competition data description):
    game_play   : str   unique drill-attempt id (e.g. "2025_WR_0012")
    frame       : int   10 Hz frame index within the attempt
    nfl_id      : int   prospect NFL id (also used on BALL rows as the athlete)
    x, y        : float position in yards (combine coordinate frame)
    s           : float speed (yards/s)
    a           : float acceleration (yards/s^2, signed along dir)
    dis         : float cumulative distance (yards)
    dir, orientation : float degrees
    event       : str   drill phase marker (e.g. "start", "cut", "finish")
    drill       : str   drill name incl. alignment side, e.g. "SPEED_OUT_LEFT"

The drill name's LEFT/RIGHT suffix denotes the receiver's ALIGNMENT SIDE,
not the break direction. Inside vs outside break is recovered from the sign
of the 2D velocity cross product v_pre x v_post.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

DT = 0.1  # 10 Hz


@dataclass
class DrillSpec:
    name: str
    alignment: str  # "LEFT" or "RIGHT"
    n_frames: int
    cut_frame: int  # frame at which the cut is initiated
    break_side: str  # "INSIDE" or "OUTSIDE" relative to alignment
    base_speed: float  # yards/s pre-cut
    cut_sharpness: float  # deg direction change at cut


DRILLS: List[DrillSpec] = [
    DrillSpec("SPEED_OUT_LEFT", "LEFT", 90, 40, "OUTSIDE", 9.5, 45.0),
    DrillSpec("SPEED_OUT_RIGHT", "RIGHT", 90, 40, "OUTSIDE", 9.5, 45.0),
    DrillSpec("POST_CORNER_ROUTE_LEFT", "LEFT", 110, 55, "INSIDE", 9.0, 60.0),
    DrillSpec("POST_CORNER_ROUTE_RIGHT", "RIGHT", 110, 55, "INSIDE", 9.0, 60.0),
    DrillSpec("COMEBACK_ROUTE_LEFT", "LEFT", 100, 45, "INSIDE", 8.5, 55.0),
    DrillSpec("COMEBACK_ROUTE_RIGHT", "RIGHT", 100, 45, "INSIDE", 8.5, 55.0),
    DrillSpec("DAGGER_ROUTE_LEFT", "LEFT", 95, 42, "OUTSIDE", 8.0, 50.0),
    DrillSpec("DAGGER_ROUTE_RIGHT", "RIGHT", 95, 42, "OUTSIDE", 8.0, 50.0),
    DrillSpec("CORNER_ROUTE_LEFT", "LEFT", 105, 50, "OUTSIDE", 8.5, 55.0),
    DrillSpec("CORNER_ROUTE_RIGHT", "RIGHT", 105, 50, "OUTSIDE", 8.5, 55.0),
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_combine_tracking(
    n_prospects: int = 200,
    attempts_per_prospect: int = 4,
    seed: int = 42,
    out_dir: str = "data/synthetic",
) -> str:
    rng = _rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    rows: List[dict] = []
    nfl_ids = rng.integers(100_000, 999_999, size=n_prospects)
    positions = ["WR", "TE", "DB", "OL", "DL"]
    schools = ["State U", "Tech", "College A", "College B", "University C"]

    attempt_id = 0
    abilities: dict = {}
    for p_idx, nfl_id in enumerate(nfl_ids):
        position = positions[p_idx % len(positions)]
        school = schools[p_idx % len(schools)]
        draft_year = 2023 + (p_idx % 3)
        athlete_quickness = rng.uniform(0.7, 1.3)  # athlete ability factor
        abilities[int(nfl_id)] = float(athlete_quickness)

        for _ in range(attempts_per_prospect):
            drill = random.choice(DRILLS)
            attempt_id += 1
            game_play = f"{draft_year}_{drill.alignment}_{attempt_id:05d}"

            # Pre-cut trajectory: roughly linear run at base speed
            v_pre = drill.base_speed * athlete_quickness
            # Post-cut direction depends on alignment + break side
            heading_pre = 90.0  # degrees, running upfield (simplified)
            if drill.break_side == "INSIDE":
                heading_post = heading_pre - drill.cut_sharpness * rng.uniform(0.8, 1.2)
            else:
                heading_post = heading_pre + drill.cut_sharpness * rng.uniform(0.8, 1.2)
            v_post = v_pre * rng.uniform(0.55, 0.85)  # speed loss through cut

            x0 = rng.uniform(5, 15)
            y0 = rng.uniform(20, 40)

            for frame in range(drill.n_frames):
                t = frame * DT
                if frame < drill.cut_frame:
                    hd = np.radians(heading_pre)
                    vv = v_pre
                else:
                    hd = np.radians(heading_post)
                    vv = v_post
                # add small process noise
                vv += rng.normal(0, 0.15)
                dx = vv * np.cos(hd) * DT
                dy = vv * np.sin(hd) * DT
                x0 += dx
                y0 += dy
                speed = max(vv, 0.1)
                accel = (vv - (v_pre if frame < drill.cut_frame else v_post)) / DT if frame in (drill.cut_frame, drill.cut_frame + 1) else rng.normal(0, 0.5)
                rows.append({
                    "game_play": game_play,
                    "frame": frame,
                    "nfl_id": int(nfl_id),
                    "position": position,
                    "school": school,
                    "draft_year": int(draft_year),
                    "x": round(float(x0), 4),
                    "y": round(float(y0), 4),
                    "s": round(float(speed), 4),
                    "a": round(float(accel), 4),
                    "dis": round(float(frame * speed * DT), 4),
                    "dir": round(float(np.degrees(hd) % 360), 2),
                    "orientation": round(float(np.degrees(hd) % 360), 2),
                    "event": "cut" if frame == drill.cut_frame else ("start" if frame == 0 else "run"),
                    "drill": drill.name,
                })

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "combine_tracking.csv")
    df.to_csv(path, index=False)
    # Persist the latent per-athlete ability so generate_aux_tables() can build a
    # career-success target that actually shares signal with the kinematics.
    pd.DataFrame({
        "nfl_id": list(abilities.keys()),
        "latent_ability": list(abilities.values()),
    }).to_csv(os.path.join(out_dir, "latent_ability.csv"), index=False)
    print(f"wrote {path}: {len(df)} rows, {df['game_play'].nunique()} attempts")
    return path


def generate_aux_tables(out_dir: str = "data/synthetic", seed: int = 42) -> None:
    rng = _rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    n_prospects = 200
    nfl_ids = rng.integers(100_000, 999_999, size=n_prospects)
    positions = ["WR", "TE", "DB", "OL", "DL"]
    players = pd.DataFrame({
        "nfl_id": nfl_ids.astype(int),
        "name": [f"Prospect {i}" for i in range(n_prospects)],
        "position": [positions[i % len(positions)] for i in range(n_prospects)],
        "school": [f"School {i % 5}" for i in range(n_prospects)],
        "draft_year": [2023 + (i % 3) for i in range(n_prospects)],
    })
    players.to_csv(os.path.join(out_dir, "players.csv"), index=False)

    # Career-success target: partly explained by the same latent athletic ability
    # that drives on-field combine kinematics (v_pre = base_speed * ability), plus
    # idiosyncratic career noise -> a learnable but far-from-perfect signal, which
    # is what pipeline validation needs. Clearly labeled synthetic everywhere.
    ability_path = os.path.join(out_dir, "latent_ability.csv")
    if os.path.exists(ability_path):
        ability = pd.read_csv(ability_path)["latent_ability"].to_numpy(float)
        a_min, a_max = ability.min(), ability.max()
        a01 = (ability - a_min) / max(a_max - a_min, 1e-9)
    else:  # pragma: no cover - fallback keeps the table schema intact
        a01 = rng.uniform(0, 1, size=n_prospects)
    career_score = np.clip(
        0.70 * a01 + 0.30 * rng.uniform(0, 1, size=n_prospects), 0.0, 1.0
    )
    career = pd.DataFrame({
        "nfl_id": nfl_ids.astype(int),
        "career_success_score": career_score.round(4),
        "games_played": rng.integers(0, 34, size=n_prospects),
        "snaps": rng.integers(0, 1200, size=n_prospects),
    })
    career.to_csv(os.path.join(out_dir, "player_career_successes.csv"), index=False)

    combine_results = pd.DataFrame({
        "nfl_id": nfl_ids.astype(int),
        "drill": [d.name for d in DRILLS for _ in range(n_prospects // len(DRILLS) + 1)][:n_prospects],
        "result": rng.uniform(3.5, 5.5, size=n_prospects).round(3),
    })
    combine_results.to_csv(os.path.join(out_dir, "combine_results.csv"), index=False)
    print("aux tables written to", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prospects", type=int, default=200)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/synthetic")
    args = ap.parse_args()
    generate_combine_tracking(args.n_prospects, args.attempts, args.seed, args.out_dir)
    generate_aux_tables(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
