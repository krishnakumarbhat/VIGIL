# From Combine Cuts to Game Performance: Predicting NFL Separation

**Track 01, NFL Big Data Bowl 2027.** Every number in this document is measured
on the real local tables and is reproducible with `python3 src/run_analysis.py`
(> `outputs/model_metrics.json`, `outputs/figures/`, `logs/experiments.tsv`).
Companion public Kaggle Notebook: https://www.kaggle.com/code/krishnakumarbhat/nfl-bdb-2027-combine-kce-analysis

## 1. The question, and the target (exact definitions)

The prior run of this track was not predictive: GroupKFold OOF **R² = 0.0355,
MAE = 1.635** — player-level combine features pushed against a play-level
target, i.e. the same athlete-level row repeated thousands of times. The work
below rebuilds features and targets so that the prediction question is well
posed, and reports eight target formulations side by side.

**Primary (real) target — `separation_at_pass_forward`.** Official `player_play`
field, play level: yards of separation the receiver has at the instant the ball
leaves the quarterback's hand. Rows with a non-null value in our play frame:
**9,359**, across 156 athletes and 333 games. Features are pre-snap context,
pre-pass route kinematics from `game_tracking`, and combine athleticism. This is
the target statement used for every headline claim below.

**Headline by measured OOF R² — `sep_gain` = `separation_at_pass_forward` −
`cushion`.** Both terms are official `player_play` fields, so the target is real,
but it is *not* the same question: `cushion` (the defender's pre-snap gap) is
algebraically a term of the target and is also offered as a feature. The size of
that algebraic contribution is therefore measured, not hidden: the same rows and
folds are re-run with `cushion` removed (`sep_gain_no_cushion`), and both numbers
are printed in every summary below. `sep_gain` answers "how much separation did
the route *create* after the snap", which is the football question this track is
actually about.

**Other real targets.** `player_mean_sep` and `player_mean_yac` (athlete-level
means of official fields), `career_snaps_log` (log career offensive snaps),
`yards_after_catch` (official, play level, targeted+caught plays).

**Proxies — labelled as such and validated before use.** `derived_sep_pf` and
`derived_sep_catch` are distances to the nearest *tracked* opponent at the
pass-forward / catch frame, derived from `game_tracking_2023.csv`. Both are
validated against the official field; both **fail** it (Pearson r ≈ 0.03, n ≈
4,100, stored under `derived_sep*_validation_vs_official`) because
`game_tracking` samples only **1.6 athletes per play**, so the nearest *tracked*
opponent is usually not the covering defender. They are reported as negative
results and are excluded from headline selection (`eligible_for_headline`).

## 2. Data (measured inventory)

`outputs/DATA_SCHEMA.md` is generated from the files themselves. Seven CSVs:
`combine_tracking.csv` 463,189 rows × 14 (510 athletes, 6,310 sessions, 10 Hz,
draft classes 2023–2025), `player_play.csv` 314,197 × 64,
`game_tracking_2023.csv` 3,665,443 × 12 (333 games, 31,690 plays, 156
athletes), `players.csv` / `combine_results.csv` / `player_career_successes.csv`
510 rows each, `games.csv` 1,002. The join that matters is
`nfl_id` across all of them; `player_play` shares 497 athletes with
`players.csv`, `game_tracking` only 156.

**The acceleration fix (why the first attempt had no signal in DBR).**
`combine_tracking.a` is an unsigned magnitude: min = 0.000, **0.0000% of
463,189 rows are negative**. Any signed deceleration read from it is therefore
identically zero, so DBR (deceleration-to-burst ratio) was 0 for every athlete.
All kinematics are now differentiated from positions with
`scipy.signal.savgol_filter` (window 11 frames = 1.1 s, polynomial 3, `deriv`
1 and 2, `mode="interp"`), which restores the sign. Measured on the 5,618
detected cuts: DBR mean **0.913**, median 0.903, max **6.475**; the shipped
column gives **0.000** on the same cuts (figure 3b).

## 3. Features

* **Per-session combine kinematics** (6,310 sessions → 510 athlete rows):
  speed, accel, jerk, turn rate, path efficiency, stops/starts, lateral
  displacement, time-to-peak, speed retention, KCE/DBR aggregates, plus
  position/height/weight normalised speeds (body lengths per second).
* **Cuts**: turning rate > 35° in the smoothed heading → 5,618 cut events,
  509 athletes, mean turn 87.3°. Each cut carries **KCE** (chord / path length
  over ±0.5 s) and **DBR** from the signed Savitzky-Golay longitudinal
  acceleration.
* **Alignment side vs break direction (football reasoning).** `LEFT`/`RIGHT`
  suffixes in the drill *name* describe where the athlete lines up — 11 `_LEFT`
  and 10 `_RIGHT` drills. Break direction is a separate physical quantity: the
  sign of the 2-D velocity cross product `v_pre × v_post` (z-component).
  Inside/outside is the *combination*: an inside break is a break away from the
  alignment side. Measured: 796 aligned cuts (170 LEFT, 626 RIGHT) → **470
  INSIDE, 326 OUTSIDE**, 4,822 cuts from drills with no side. Left-suffixed
  drills contain both predominantly-left and predominantly-right breaks, so
  alignment side is used as a *feature*, never as the break label, and is
  documented as such in the schema.
* **Per-play kinematics** (30,959 rows): pre-pass speed profile,
  `speed_at_pf`, release time, path efficiency, cut counts, pre-pass KCE/DBR,
  snap-to-pass duration, total turn.
* **Pre-snap context**: cushion, route, alignment, motion, down, distance,
  quarter, play direction, formation, lined-up position, position group.
* **Forbidden**: any outcome field (`separation_*`, `yards_after_catch`,
  `pass_result`, `yards_gained`, `expected_*`, `target`, the derived proxies)
  is hard-excluded from the feature list.

## 4. Models and validation protocol

* **Gradient boosting**: LightGBM 4.7.0 (400 trees, lr 0.05) with
  scikit-learn `HistGradientBoostingRegressor` run on identical folds as an
  independent check; train-fold median imputation, one-hot categoricals.
* **Splits**: `GroupKFold(5)` **by athlete** (`nfl_id`) and, for play-level
  targets, a second split **by `game_play_id`**, so no athlete and no play
  crosses a fold. Every reported score is out-of-fold.
* **Brackets**: each model is read against (a) a mean dummy regressor on the
  same folds and (b) a **shuffled-label null** through the identical CV.
* **Mixed effects** (`statsmodels MixedLM`): fixed effects as above with a
  player random intercept, a game random intercept, player + team variance
  component, player + game variance component, and a *population-level* OOF
  score (fixed effects only, so held-out players get no fitted intercept).
* **Learning curve** and **per-position OOF breakdown** are computed for the
  headline target and stored in `model_metrics.json`.

## 5. Results (every number out-of-fold, GroupKFold 5)

**Targets, LightGBM OOF (HistGB in parentheses; mean dummy ≈ 0; shuffled null
≈ −0.05…−0.15 on every row).**

| target | rows | by_player R² / MAE / ρ | by_game_play R² / MAE / ρ |
|---|---|---|---|
| `sep_at_pass_forward` (official) | 9,359 | 0.2613 / 1.4092 / 0.468 (HGB 0.2633) | **0.2820 / 1.3864 / 0.490** |
| … kinematics only | 9,359 | 0.1155 / 1.5901 / 0.258 | — |
| … pre-snap only | 9,359 | 0.0927 / 1.5813 / 0.339 | — |
| `sep_gain` (headline) | 8,715 | 0.6463 / 1.3776 / 0.818 | **0.6561 / 1.3503 / 0.825** (HGB 0.6576) |
| `sep_gain` w/o `cushion` feature | 8,715 | 0.0756 / 2.2792 / 0.318 | 0.1074 / 2.2383 / 0.356 |
| `career_snaps_log` | 510 | 0.5979 / 1.1806 / 0.760 | — |
| `yards_after_catch` (official) | 1,259 | −0.0582 / 4.6050 / 0.220 | 0.0457 / 4.2897 / 0.357 |
| `derived_sep_pf` (**proxy**, rejected) | 4,105 | 0.0502 / 7.1198 / 0.261 | — |
| `derived_sep_catch` (**proxy**, rejected) | 3,159 | 0.0129 / 7.9649 / 0.231 | — |
| `player_mean_sep` | 145 | −0.0717 / 0.3519 / 0.225 | — |
| `player_mean_sep` (preregistered 17 feats) | 145 | −0.1263 / 0.3544 / 0.301 | — |
| `player_mean_yac` | 122 | −0.1105 / 1.3129 / 0.193 | — |
| legacy player-features-on-play-target | 46,737 | −0.0027 / 1.6844 / 0.038 | — |

Three findings. (i) **The official target is predictable, modestly.**
`separation_at_pass_forward` reaches R² 0.26 (unseen athletes) / 0.28 (unseen
plays), MAE ≈ 1.39 yd against a 1.71 yd mean baseline, ρ ≈ 0.47–0.49; the
shuffled null sits at −0.057 / −0.066, so the score is real. The ablation rows
localise the signal: pre-pass route kinematics (0.1155) add roughly as much as
pre-snap context (0.0927), and together they reach 0.2613. (ii) **The headline
`sep_gain` is high for a partly algebraic reason, and the size of that reason is
measured**: R² falls from 0.6561 to **0.1074** when the `cushion` feature is
removed (by_player 0.6463 → 0.0756), i.e. the pre-snap gap plus its algebraic
coupling explains most of the 0.66, while ~0.11 R² of *created* separation is
genuinely predicted. Both numbers, not just the flattering one, are in
`model_metrics.json`. (iii) **Athlete-level targets fail.** Predicting career
snap volume works (R² 0.598), but predicting an athlete's *mean* separation or
YAC from combine kinematics is at or below the mean dummy (−0.07, −0.11, −0.13)
with ρ ≤ 0.30 — combine movement quality does not transfer to "who separates on
Sundays" at n = 122–145 athletes. The tracking-derived proxies stay at 0.05 /
0.01 and are excluded from headline selection.

**MixedLM (`statsmodels`, REML, 8,697 rows).** Player random intercept:
Intercept 1.6308 (p = 4e−6), `cushion` **+0.1928 yd per yd** (p < 1e−16),
`speed_at_pf` −0.0853 (p = 1.1e−5), `max_speed` +0.1128 (p = 2.8e−4),
Group Var 0.0111 (p = 0.005), in-sample R² 0.2039. Game random intercept:
Group Var 0.0190 (p = 1.6e−5), R² 0.2169; player+team variance component: team
Var 0.0108 (p = 0.0049); player+game: R² 0.2756. The **population-level OOF
score** (fixed effects only — a *new* athlete gets no fitted intercept) is
R² 0.1869, MAE 1.4673, ρ 0.4015, 5/5 folds converged, 0 fallbacks. Reading:
about 0.19 R² generalises to unseen players, and the football content is
"pre-snap cushion and top-end speed dominate; holding max speed constant,
arriving *faster* at pass-forward slightly hurts (β = −0.085), consistent with
route timing rather than sprinting".

**Directional KCE model.** `kce ~ cut_angle_deg + approach_speed +
speed_retention + C(align_side)`, player random intercept, 5,618 cuts / 509
athletes: cut angle −0.00618 KCE/degree (p < 1e−16), speed retention +0.01255
(p < 1e−16), approach speed −0.00112 (p = 0.15, n.s.), alignment side RIGHT
+0.00801 (p = 0.0037), NONE −0.00301 (p = 0.23), Group Var 0.0234 (p = 0.007),
in-sample R² 0.9881. Honest caveat: this fit is flagged `converged = false`
(statsmodels reports a boundary solution on a scale-separated problem), and its
alignment-side effect is ~0.03 SD — practically nothing. Together with the
INSIDE/OUTSIDE counts (470/326, figure 1) this supports the design decision:
alignment side routes the athlete, it does not describe the break.

**Learning curve (headline, by_game_play).** 20% → R² 0.6213 (1,752 rows),
40% → 0.6453, 60% → 0.6425, 80% → 0.6511, 100% → 0.6561 (8,715): still rising
slowly, so more plays would help, but data volume is not the bottleneck.

**Per-position.** Separation rows are skill positions by construction (WR/TE/RB,
n = 8,715, R² 0.6463 / 0.6561); figure 8 supplements this with cut-quality
scores for all four position groups.

## 6. Visual evidence

The Kaggle Writeup media gallery embeds three selected figures: the inside/outside
cut-direction analysis, the target-specific feature-importance plot, and the
out-of-fold prediction/ablation plot. Five additional plots remain available in
the linked notebook's public source dataset, including KCE/DBR distributions,
turn-rate, learning-curve, and per-position diagnostics.

## 7. Limitations

Game tracking covers **one season and ~1.6 athletes per play**, which is why the
derived separation proxies failed validation (r = 0.0334 / 0.0141) and why the
official field is used for every claim. `sep_gain` is an official-derived target
whose headline score is cushion-inflated (ablation: 0.1074 without the
feature). Athlete-level separation/YAC prediction is a negative result at these
sample sizes. The directional KCE fit is flagged unconverged. Combine athletes
(510) overlap game tracking for only 156, and only skill positions carry
separation.
