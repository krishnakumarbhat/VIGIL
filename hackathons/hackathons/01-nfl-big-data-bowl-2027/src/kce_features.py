"""Kinematic features from REAL NFL combine and game tracking (Savitzky-Golay based).

Why this module exists
----------------------
The `a` column shipped in ``combine_tracking.csv`` is an *unsigned magnitude*:
over all 463,189 rows ``min(a) == 0.0`` and ``mean(a) == 2.81`` (measured, see
``outputs/DATA_SCHEMA.md``).  Any "deceleration" taken from it is therefore
identically zero, which is why every player came out with DBR == 0 in the first
pass of this pipeline.  Signed acceleration is *not* recoverable from that
column -- it has to be differentiated out of the (x, y) positions themselves.

Here every kinematic quantity is derived from position by Savitzky-Golay
smoothing differentiation (``scipy.signal.savgol_filter``, window = 11 frames =
1.1 s, 3rd order polynomial, dt = 0.1 s), which suppresses the ~1 Hz position
noise that makes naive finite differences useless at 10 Hz:

    v(t) = d/dt  SG[x](t)      a(t) = d2/dt2 SG[x](t)
    a_longitudinal = (v . a) / |v|      <- SIGNED: negative when decelerating
    jerk = d/dt a_longitudinal

Definitions used throughout
---------------------------
cut        direction change > 35 deg between the 3-frame pre-window and the
           3-frame post-window of the smoothed heading, with mean speed in both
           windows >= 1.0 yd/s (standing athletes do not "cut").
cross      z-component of the 2-D velocity cross product v_pre x v_post.
           Positive cross == left turn (coordinate convention confirmed in the
           competition discussion 749574: "Combine x/y is a normal frame where
           a positive cross product means a left turn").
ALIGNMENT SIDE  the LEFT/RIGHT suffix carried by the drill *name*
           (``SPEED_OUT_LEFT``, ``POST_CORNER_ROUTE_RIGHT`` ...).  It says which
           side the athlete is aligned/started on.  **It is NOT the break
           direction** -- break direction comes from the sign of `cross`, and
           INSIDE/OUTSIDE is derived from the *pair* (turn direction,
           alignment side): turn == alignment side => OUTSIDE break, otherwise
           INSIDE.  Drill names without a side suffix get alignment "NONE" and
           break side "UNALIGNED".
KCE        Kinematic Cut Efficiency = |chord| / path length over the +/-0.5 s
           window around a cut (1.0 = no motion wasted, mechanically decreasing
           in cut angle, so it is always modelled with cut angle as a control).
DBR        Deceleration-to-Burst Ratio = peak |longitudinal deceleration| in the
           0.6 s before the cut / peak longitudinal acceleration in the 0.6 s
           after it, both from the Savitzky-Golay derivative, with a burst floor
           of 1.0 yd/s^2 (a near-zero burst made per-player DBR blow up to ~135
           in an early run; see logs/experiments.tsv).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------------------------------------------- constants --
DT = 0.1                 # tracking is exactly 10 Hz (measured: every dt == 0.1 s)
SG_WINDOW = 11           # frames (1.1 s) smoothing window
SG_POLY = 3              # polynomial order
CUT_ANGLE_DEG = 35.0     # direction change that counts as a cut
CUT_MIN_SPEED = 1.0      # yd/s: below this a "cut" is just standing still
CUT_PRE = 3              # frames in the pre/post heading windows
CUT_POST = 3
CUT_MIN_GAP = 5          # frames between two accepted cuts (0.5 s)
CUT_WINDOW = 5           # frames either side of the cut for KCE
BURST_FLOOR = 1.0        # yd/s^2 floor under the DBR denominator
DECEL_WINDOW = 6         # frames (0.6 s) around the cut for DBR
STOP_SPEED = 1.0         # yd/s threshold for stop / start counting

_ALIGN_RE = re.compile(r"_(LEFT|RIGHT)$")


# ------------------------------------------------------------- kinematics ---
def savgol_kinematics(x: np.ndarray, y: np.ndarray, dt: float = DT,
                      window: int = SG_WINDOW, polyorder: int = SG_POLY) -> Dict[str, np.ndarray]:
    """Differentiate a 10 Hz (x, y) trajectory with a Savitzky-Golay filter.

    Returns velocity components, *signed* longitudinal/lateral acceleration,
    jerk, heading (rad, unwrapped) and turn rate (deg/s).  Short sessions fall
    back to smaller windows so that sessions as short as 3 frames still work.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:                                    # degenerate: cannot differentiate
        z = np.zeros(n)
        return dict(vx=z, vy=z, ax=z, ay=z, speed=z, a_long=z, a_lat=z,
                    jerk=z, heading=z, turn_rate=z)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        w = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)
    p = min(polyorder, w - 1)
    # positions -> velocity (1st derivative) and acceleration (2nd derivative)
    vx = savgol_filter(x, w, p, deriv=1, delta=dt, mode="interp")
    vy = savgol_filter(y, w, p, deriv=1, delta=dt, mode="interp")
    ax = savgol_filter(x, w, p, deriv=2, delta=dt, mode="interp")
    ay = savgol_filter(y, w, p, deriv=2, delta=dt, mode="interp")
    speed = np.hypot(vx, vy)
    safe = np.maximum(speed, 1e-6)
    # signed longitudinal (tangential) acceleration: <0 while decelerating
    a_long = (vx * ax + vy * ay) / safe
    # signed lateral acceleration (centripetal): |v x a| / |v|
    a_lat = (vx * ay - vy * ax) / safe
    # jerk = d/dt of the longitudinal acceleration
    if n >= 5:
        jerk = savgol_filter(a_long, min(w, n if n % 2 == 1 else n - 1), p,
                             deriv=1, delta=dt, mode="interp")
    else:
        jerk = np.gradient(a_long, dt)
    heading = np.unwrap(np.arctan2(vy, vx))
    if n >= 5:
        turn_rate = savgol_filter(np.degrees(heading),
                                  min(w, n if n % 2 == 1 else n - 1), p,
                                  deriv=1, delta=dt, mode="interp")
    else:
        turn_rate = np.degrees(np.gradient(heading, dt))
    # heading is meaningless while standing still: blank the turn rate there
    turn_rate = np.where(speed >= CUT_MIN_SPEED, turn_rate, 0.0)
    return dict(vx=vx, vy=vy, ax=ax, ay=ay, speed=speed, a_long=a_long,
                a_lat=a_lat, jerk=jerk, heading=heading, turn_rate=turn_rate)


def alignment_side(drill_name: str) -> str:
    """LEFT / RIGHT / NONE suffix of the drill NAME = where the athlete aligned.

    Alignment side is an attribute of the drill setup, never the break
    direction; break direction is computed separately from the cross product.
    """
    m = _ALIGN_RE.search(str(drill_name))
    return m.group(1) if m else "NONE"


def drill_base(drill_name: str) -> str:
    """Drill name with the LEFT/RIGHT alignment suffix stripped."""
    return _ALIGN_RE.sub("", str(drill_name))


def break_side(turn: str, align: str) -> str:
    """Map (turn direction, alignment side) -> INSIDE / OUTSIDE / UNALIGNED.

    Convention (validated empirically in ``turn_sign_by_drill``): a LEFT-aligned
    athlete whose break turns him LEFT is going away from the formation, i.e.
    to the OUTSIDE; the same turn from a RIGHT-aligned athlete is also outside.
    Athletes aligned on a drill with no side get UNALIGNED.
    """
    if align == "NONE":
        return "UNALIGNED"
    return "OUTSIDE" if turn == align else "INSIDE"


# -------------------------------------------------------------------- cuts --
def detect_cuts(kin: Dict[str, np.ndarray], drill_name: str = "",
                min_speed: float = CUT_MIN_SPEED,
                angle_deg: float = CUT_ANGLE_DEG) -> pd.DataFrame:
    """Detect direction changes > 35 deg in a single session and label them.

    The pre/post heading means are compared on the *smoothed* heading; the
    cross product of the mean pre/post velocity vectors supplies the turn
    direction, and the drill-name suffix supplies the alignment side.
    """
    heading = np.degrees(kin["heading"])
    speed = kin["speed"]
    vx, vy = kin["vx"], kin["vy"]
    a_long = kin["a_long"]
    n = heading.size
    rows: List[dict] = []
    lo, hi = CUT_PRE + 1, n - CUT_POST - 1
    if hi <= lo:
        return pd.DataFrame(rows)
    dtheta = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for i in range(lo, hi):
        pre = heading[i - CUT_PRE:i]
        post = heading[i + 1:i + 1 + CUT_POST]
        d = float(np.mean(post) - np.mean(pre))
        d = (d + 180.0) % 360.0 - 180.0
        dtheta[i] = d
        ok[i] = (speed[i - CUT_PRE:i].mean() >= min_speed and
                 speed[i + 1:i + 1 + CUT_POST].mean() >= min_speed)
    cand = np.where(ok & (np.abs(dtheta) >= angle_deg))[0]
    if cand.size == 0:
        return pd.DataFrame(rows)
    # non-maximum suppression so one bend is one cut
    order = cand[np.argsort(-np.abs(dtheta[cand]))]
    taken: List[int] = []
    for i in order:
        if all(abs(int(i) - j) >= CUT_MIN_GAP for j in taken):
            taken.append(int(i))
    taken.sort()
    align = alignment_side(drill_name)
    for i in taken:
        pre_s, post_s = slice(i - CUT_PRE, i), slice(i + 1, i + 1 + CUT_POST)
        vpx, vpy = vx[pre_s].mean(), vy[pre_s].mean()
        vnx, vny = vx[post_s].mean(), vy[post_s].mean()
        cross = float(vpx * vny - vpy * vnx)
        turn = "LEFT" if cross > 0 else "RIGHT"
        # KCE over the +/-0.5 s cut window: chord / path length
        a0, a1 = max(0, i - CUT_WINDOW), min(n - 1, i + CUT_WINDOW)
        xs, ys = kin_x(kin, a0, a1), kin_y(kin, a0, a1)
        chord = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if a1 > a0 else 0.0
        path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) if a1 > a0 else 0.0
        kce = chord / path if path > 1e-9 else np.nan
        # DBR from signed Savitzky-Golay longitudinal acceleration
        d0, d1 = max(0, i - DECEL_WINDOW), i
        b0, b1 = i + 1, min(n, i + 1 + DECEL_WINDOW)
        decel = float(np.max(np.clip(-a_long[d0:d1], 0, None))) if d1 > d0 else 0.0
        burst = float(np.max(np.clip(a_long[b0:b1], 0, None))) if b1 > b0 else 0.0
        dbr = decel / max(burst, BURST_FLOOR)
        app = float(speed[max(0, i - CUT_PRE):i].mean())
        exi = float(speed[i + 1:min(n, i + 1 + CUT_POST)].mean())
        rows.append(dict(
            cut_frame=i, cut_time=i * DT, turn_deg=float(dtheta[i]), cross=cross,
            turn=turn, align_side=align, break_side=break_side(turn, align),
            drill=drill_name, drill_base=drill_base(drill_name),
            kce=kce, dbr=dbr, decel_peak=decel, burst_peak=burst,
            approach_speed=app, exit_speed=exi,
            speed_retention=exi / app if app > 1e-6 else np.nan,
            turn_rate_peak=float(np.max(np.abs(kin["turn_rate"][max(0, i - 3):i + 4]))),
            cut_angle_deg=abs(float(dtheta[i])),
        ))
    return pd.DataFrame(rows)


def kin_x(kin: Dict[str, np.ndarray], a: int, b: int) -> np.ndarray:
    """Slice helper: positions are stored back in the dict by the caller."""
    return np.asarray(kin["x"][a:b + 1], dtype=float)


def kin_y(kin: Dict[str, np.ndarray], a: int, b: int) -> np.ndarray:
    return np.asarray(kin["y"][a:b + 1], dtype=float)


# ------------------------------------------------------------ session stats --
def session_stats(kin: Dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
                  dt: float = DT) -> dict:
    """Per-session (one drill attempt) summary statistics."""
    speed = kin["speed"]
    a_long = kin["a_long"]
    jerk = kin["jerk"]
    tr = np.abs(kin["turn_rate"])
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(x) > 1 else 0.0
    net = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if len(x) > 1 else 0.0
    moving = speed >= STOP_SPEED
    stops = int(np.sum((~moving[1:]) & moving[:-1]))     # stopped -> moving
    starts = int(np.sum(moving[1:] & (~moving[:-1])))    # moving -> stopped
    imax = int(np.argmax(speed))
    peak = float(speed[imax])
    # lateral displacement relative to the session's mean travel direction
    if len(x) > 2 and path > 1e-9:
        ux, uy = (x[-1] - x[0]) / max(net, 1e-9), (y[-1] - y[0]) / max(net, 1e-9)
        lat = float(np.max(np.abs((x - x[0]) * uy - (y - y[0]) * ux)))
    else:
        lat = 0.0
    return dict(
        n_frames=int(len(x)),
        mean_speed=float(np.mean(speed)),
        max_speed=peak,
        p95_speed=float(np.percentile(speed, 95)),
        time_to_peak_speed=imax * dt,
        frac_above_80pct_peak=float(np.mean(speed >= 0.8 * peak)) if peak > 0 else 0.0,
        peak_accel=float(np.max(a_long)),
        peak_decel=float(np.max(np.clip(-a_long, 0, None))),
        mean_jerk=float(np.mean(np.abs(jerk))),
        p95_jerk=float(np.percentile(np.abs(jerk), 95)),
        mean_turn_rate=float(np.mean(tr)),
        max_turn_rate=float(np.max(tr)) if tr.size else 0.0,
        path_length=path,
        net_displacement=net,
        path_efficiency=net / path if path > 1e-9 else np.nan,
        lateral_displacement=lat,
        stop_count=stops,
        start_count=starts,
        moving_frac=float(np.mean(moving)),
        duration_s=float(len(x) * dt),
    )


# ------------------------------------------------------ combine aggregation --
def build_combine_tables(ct: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Turn raw combine tracking into session, player-drill and player tables.

    ``ct`` must already be filtered to entity_type == 'PLAYER'.  Alignment side
    is attached at every level; it is a drill-name property (setup), never a
    statement about which way the athlete broke.
    """
    ct = ct.copy()
    ct["time"] = pd.to_datetime(ct["time"])
    ct = ct.sort_values(["event_id", "nfl_id", "drill_name", "attempt", "time"])
    key = ["event_id", "nfl_id", "drill_name", "attempt"]
    ses_rows, cut_frames = [], []
    for k, g in ct.groupby(key, sort=False):
        x = g["x"].to_numpy(float)
        y = g["y"].to_numpy(float)
        if len(x) < 3:
            continue
        kin = savgol_kinematics(x, y)
        kin["x"], kin["y"] = x, y
        st = session_stats(kin, x, y)
        st.update(event_id=k[0], nfl_id=int(k[1]), drill_name=k[2], attempt=int(k[3]))
        st["align_side"] = alignment_side(k[2])
        st["drill_base"] = drill_base(k[2])
        st["drill_type"] = g["drill_type"].iloc[0]
        st["draft_year"] = int(g["draft_year"].iloc[0])
        ses_rows.append(st)
        c = detect_cuts(kin, drill_name=k[2])
        if len(c):
            for col in ("event_id", "nfl_id", "attempt", "drill_name"):
                c[col] = st[col]
            cut_frames.append(c)
    sessions = pd.DataFrame(ses_rows)
    cuts = pd.concat(cut_frames, ignore_index=True) if cut_frames else pd.DataFrame()

    # per-session cut aggregates folded back onto the session rows
    if len(cuts):
        per_ses = cuts.groupby(key, sort=False).agg(
            n_cuts=("kce", "size"), mean_kce=("kce", "mean"), max_kce=("kce", "max"),
            mean_dbr=("dbr", "mean"), max_dbr=("dbr", "max"),
            mean_cut_turn_deg=("cut_angle_deg", "mean"), max_cut_turn_deg=("cut_angle_deg", "max"),
            mean_speed_retention=("speed_retention", "mean"),
            mean_decel_peak=("decel_peak", "mean"), mean_burst_peak=("burst_peak", "mean"),
            n_inside=("break_side", lambda s: int((s == "INSIDE").sum())),
            n_outside=("break_side", lambda s: int((s == "OUTSIDE").sum())),
        ).reset_index()
        sessions = sessions.merge(per_ses, on=key, how="left")
    for c in ("n_cuts", "n_inside", "n_outside"):
        sessions[c] = sessions[c].fillna(0).astype(int) if c in sessions else 0

    return dict(sessions=sessions, cuts=cuts,
                player_drill=aggregate_player_features(sessions, cuts, by="drill_base"))


_ID_COLS = ("event_id", "nfl_id", "attempt", "draft_year", "game_id", "play_id")
_COUNT_COLS = ("n_frames", "n_cuts", "n_inside", "n_outside", "stop_count",
               "start_count", "duration_s")


def aggregate_player_features(sessions: pd.DataFrame, cuts: pd.DataFrame,
                              by: Optional[str] = None) -> pd.DataFrame:
    """Aggregate session rows up to player (optionally player x drill) level.

    Only numeric *measurements* are averaged; ids, counts and the drill labels
    are handled explicitly so the group keys survive the aggregation.
    """
    if sessions.empty:
        return pd.DataFrame()
    stat_cols = [c for c in sessions.columns
                 if sessions[c].dtype.kind in "if" and c not in _ID_COLS]
    mean_cols = [c for c in stat_cols if c not in _COUNT_COLS]
    sum_cols = [c for c in stat_cols if c in _COUNT_COLS]
    keys = ["nfl_id"] + ([by] if by else []) + (["align_side"] if by else [])
    spec = {**{c: "mean" for c in mean_cols}, **{c: "sum" for c in sum_cols}}
    grp = sessions.groupby(keys, sort=False)
    out = grp.agg(spec).reset_index()
    out["n_sessions"] = grp.size().values
    if "drill_type" in sessions.columns:
        out = out.merge(sessions.groupby(keys, sort=False)["drill_type"].first()
                        .reset_index(), on=keys, how="left")
    if "draft_year" in sessions.columns:
        out = out.merge(sessions.groupby(keys, sort=False)["draft_year"].first()
                        .reset_index(), on=keys, how="left")
    return out


def side_split_features(player: pd.DataFrame, cuts: pd.DataFrame) -> pd.DataFrame:
    """Add INSIDE/OUTSIDE- and alignment-specific aggregates to a player table.

    Alignment side (drill-name LEFT/RIGHT) and break side (cross product) are
    kept as *separate* feature families so a model can use either without the
    two being conflated.
    """
    if not len(cuts):
        return player
    out = player.copy()
    inside = cuts[cuts["break_side"] == "INSIDE"].groupby("nfl_id").agg(
        n_inside_cuts=("kce", "size"), mean_kce_inside=("kce", "mean"),
        mean_dbr_inside=("dbr", "mean")).reset_index()
    outside = cuts[cuts["break_side"] == "OUTSIDE"].groupby("nfl_id").agg(
        n_outside_cuts=("kce", "size"), mean_kce_outside=("kce", "mean"),
        mean_dbr_outside=("dbr", "mean")).reset_index()
    left = cuts[cuts["align_side"] == "LEFT"].groupby("nfl_id").agg(
        n_align_left=("kce", "size"), mean_kce_align_left=("kce", "mean")).reset_index()
    right = cuts[cuts["align_side"] == "RIGHT"].groupby("nfl_id").agg(
        n_align_right=("kce", "size"), mean_kce_align_right=("kce", "mean")).reset_index()
    for part in (inside, outside, left, right):
        out = out.merge(part, on="nfl_id", how="left")
    for c in out.columns:
        if c.startswith("n_"):
            out[c] = out[c].fillna(0)
    # rate features: share of *aligned* breaks that went outside (UNALIGNED
    # cuts have no drill-name side, so they must not dilute the ratio)
    aligned = out.get("n_inside_cuts", 0) + out.get("n_outside_cuts", 0)
    out["frac_breaks_outside"] = np.where(
        aligned > 0, out.get("n_outside_cuts", 0) / aligned, np.nan)
    out["inside_outside_kce_gap"] = out.get("mean_kce_inside") - out.get("mean_kce_outside")
    return out


def turn_sign_by_drill(cuts: pd.DataFrame) -> pd.DataFrame:
    """Empirical check of the alignment convention: turn direction per drill.

    A LEFT-suffixed OUT/CORNER-style drill should show a dominant turn
    direction; if the two sides mirror each other the LEFT/RIGHT suffix really
    encodes physical alignment in the tracking frame, which is what the
    INSIDE/OUTSIDE mapping in :func:`break_side` assumes.
    """
    if not len(cuts):
        return pd.DataFrame()
    g = cuts.groupby("drill").agg(
        n=("turn", "size"),
        frac_left_turn=("turn", lambda s: float((s == "LEFT").mean())),
        mean_turn_deg=("turn_deg", "mean"),
        frac_outside=("break_side", lambda s: float((s == "OUTSIDE").mean()) if (s != "UNALIGNED").all() else np.nan),
        n_unaligned=("break_side", lambda s: int((s == "UNALIGNED").sum())),
    ).reset_index()
    g["align_side"] = g["drill"].map(alignment_side)
    return g.sort_values("n", ascending=False)


# ------------------------------------------------------------- play features --
def play_event_times(gt: pd.DataFrame) -> pd.DataFrame:
    """Per-play event timestamps (ball_snap / pass_forward / pass_arrived)."""
    if not pd.api.types.is_datetime64_any_dtype(gt["time"]):
        gt = gt.assign(time=pd.to_datetime(gt["time"]))
    ev = gt[gt["event"].notna() & (gt["event"].astype(str) != "nan")]
    piv = ev.pivot_table(index=["game_id", "play_id"], columns="event",
                         values="time", aggfunc="min", observed=True)
    out = pd.DataFrame(index=piv.index)
    for name, col in (("t_snap", "ball_snap"), ("t_pf", "pass_forward"),
                      ("t_arr", "pass_arrived"), ("t_caught", "pass_outcome_caught")):
        out[name] = piv[col] if col in piv.columns else pd.NaT
    return out.reset_index()


def build_play_features(gt: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    """Per (game, play, player) kinematic features computed BEFORE pass forward.

    Only frames in [ball_snap, pass_forward] are used, so nothing downstream of
    the target instant can leak into the features.  Receivers are identified by
    joining player_play.route_ran; rows without a snap+pass_forward pair are
    dropped because the measurement window would be ill-defined.
    """
    times = play_event_times(gt)
    gt = gt.assign(time=pd.to_datetime(gt["time"]))
    gt = gt.merge(times, on=["game_id", "play_id"], how="left")
    gt = gt[gt["t_snap"].notna() & gt["t_pf"].notna()].copy()
    gt = gt[(gt["time"] >= gt["t_snap"]) & (gt["time"] <= gt["t_pf"])].copy()
    keep = ["game_id", "play_id", "nfl_id", "time", "x", "y"]
    gt = gt[keep].sort_values(["game_id", "play_id", "nfl_id", "time"])
    meta = times.merge(
        pp[["game_id", "play_id", "nfl_id", "route_ran", "separation_at_pass_forward",
            "cushion", "target", "rec_yards", "yards_after_catch", "team_abbr",
            "lined_up_position", "receiver_alignment", "pass_result", "pass_length",
            "in_motion_at_ball_snap", "expected_yards_after_catch", "yards_gained",
            "down", "yards_to_go", "quarter", "play_direction", "offense_formation",
            "expected_points_added"]],
        on=["game_id", "play_id"], how="inner")
    rows = []
    for k, g in gt.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        x = g["x"].to_numpy(float)
        y = g["y"].to_numpy(float)
        if len(x) < 6:
            continue
        kin = savgol_kinematics(x, y)
        kin["x"], kin["y"] = x, y
        st = session_stats(kin, x, y)
        st.pop("duration_s", None)
        # release: first frame at which the athlete is actually moving
        spd = kin["speed"]
        mov = np.where(spd >= CUT_MIN_SPEED)[0]
        st["release_time_s"] = float(mov[0] * DT) if mov.size else np.nan
        st["release_burst"] = float(np.max(kin["a_long"][:max(6, (mov[0] if mov.size else 0) + 6)])) if len(x) > 6 else np.nan
        st["speed_at_pf"] = float(spd[-1])
        st["a_long_at_pf"] = float(kin["a_long"][-1])
        st["total_turn_deg"] = float(np.sum(np.abs(kin["turn_rate"])) * DT)
        st["snap_to_pf_s"] = float(len(x) * DT)
        # route-level cut count within the pre-pass window
        cuts = detect_cuts(kin, drill_name="")
        st["n_pre_pass_cuts"] = int(len(cuts))
        st["has_pre_pass_cut"] = int(len(cuts) > 0)
        if len(cuts):
            st["mean_cut_turn_deg"] = float(cuts["cut_angle_deg"].mean())
            st["mean_pre_pass_kce"] = float(cuts["kce"].mean())
            st["mean_pre_pass_dbr"] = float(cuts["dbr"].mean())
        else:
            st["mean_cut_turn_deg"] = 0.0
            st["mean_pre_pass_kce"] = np.nan
            st["mean_pre_pass_dbr"] = np.nan
        st["game_id"], st["play_id"], st["nfl_id"] = int(k[0]), int(k[1]), int(k[2])
        rows.append(st)
    feats = pd.DataFrame(rows)
    out = feats.merge(meta, on=["game_id", "play_id", "nfl_id"], how="inner",
                      suffixes=("", "_meta"))
    return out


def derive_separation(gt: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    """Tracking-derived separation: distance to the nearest opponent.

    PROXY -- the official ``separation_at_pass_forward`` in player_play already
    exists, so this is built as an independent *check*: for plays where at
    least one opponent is also tracked we take the minimum distance from the
    receiver to any player on the other team at the pass_forward (and, where
    available, pass_arrived) frame.  It is validated against the official field
    before use as a target (Pearson/Spearman reported in model_metrics.json).
    """
    times = play_event_times(gt)
    gt = gt.assign(time=pd.to_datetime(gt["time"]))
    gt = gt.merge(times, on=["game_id", "play_id"], how="left")
    gt = gt.merge(pp[["game_id", "play_id", "nfl_id", "team_abbr", "route_ran",
                      "separation_at_pass_forward", "cushion", "lined_up_position"]],
                  on=["game_id", "play_id", "nfl_id"], how="left")
    team = gt[["game_id", "play_id", "nfl_id", "team_abbr"]].drop_duplicates()
    gt = gt[gt["team_abbr"].notna()]
    rows = []
    grouped = gt.groupby(["game_id", "play_id"], sort=False)
    for (gid, pid), g in grouped:
        if g["route_ran"].isna().all() or g["t_pf"].isna().all():
            continue
        t_pf = g["t_pf"].iloc[0]
        t_arr = g["t_arr"].iloc[0] if pd.notna(g["t_arr"].iloc[0]) else None
        rcv_ids = g.loc[g["route_ran"].notna(), "nfl_id"].unique()
        for t_label, t_val in (("pf", t_pf), ("catch", t_arr)):
            if t_val is None or pd.isna(t_val):
                continue
            f = g[(g["time"] - t_val).abs() <= pd.Timedelta(0.15)]
            if f.empty:
                continue
            for rid in rcv_ids:
                r = f[f["nfl_id"] == rid]
                if r.empty:
                    continue
                rx, ry = float(r["x"].iloc[0]), float(r["y"].iloc[0])
                rteam = r["team_abbr"].iloc[0]
                opp = f[(f["nfl_id"] != rid) & (f["team_abbr"] != rteam)]
                if opp.empty:
                    continue
                d = np.hypot(opp["x"].to_numpy(float) - rx, opp["y"].to_numpy(float) - ry)
                near = int(np.argmin(d))
                rows.append(dict(game_id=int(gid), play_id=int(pid), nfl_id=int(rid),
                                 moment=t_label, derived_sep=float(np.min(d)),
                                 n_opponents_tracked=int(len(opp)),
                                 nearest_opp_position=str(
                                     opp["lined_up_position"].iloc[near]),
                                 nearest_opp_id=int(opp["nfl_id"].iloc[near]),
                                 official_sep_pf=r["separation_at_pass_forward"].iloc[0],
                                 cushion=r["cushion"].iloc[0],
                                 route_ran=r["route_ran"].iloc[0],
                                 team_abbr=rteam))
    return pd.DataFrame(rows)


# -------------------------------------------------------- normalisation utils --
def add_normalised_features(df: pd.DataFrame, feature_cols: Sequence[str],
                            group_col: str = "position_group") -> pd.DataFrame:
    """Add body-size and position normalisations.

    * ``*_bls``   : value expressed in body-lengths per second (speed) or
                    body-lengths per second^2 (acceleration) using the athlete's
                    combine height -- a size-free way to compare a 6'5" receiver
                    with a 5'10" slot.
    * ``z_*``     : z-score of the feature inside the athlete's position group,
                    so "fast" always means "fast for his position".
    """
    out = df.copy()
    if "combine_height" in out.columns:
        height_yd = out["combine_height"].astype(float) / 36.0
        for c in feature_cols:
            if c in out.columns and ("speed" in c or c.startswith("max_speed")):
                out[f"{c}_bls"] = out[c] / height_yd
            if c in out.columns and "accel" in c:
                out[f"{c}_bls"] = out[c] / height_yd
    if group_col in out.columns:
        for c in feature_cols:
            if c not in out.columns or out[c].dtype.kind not in "if":
                continue
            g = out.groupby(group_col)[c]
            mu, sd = g.transform("mean"), g.transform("std")
            out[f"z_{c}"] = (out[c] - mu) / sd.replace(0, np.nan)
    return out
