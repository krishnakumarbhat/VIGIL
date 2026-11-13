# NFL Big Data Bowl 2027 - Consolidated Evidence File (Track 01)

Research date: 2026-10-22. All quotes below were retrieved live via websearch/webfetch
on this date. Client-rendered Kaggle pages could not be fetched in full; for those,
quotes come from search-engine index snippets of the exact URLs (marked [snippet]).

## 1. Competition overview and exact metric

- Kaggle page title/tagline (retrieved from the Kaggle overview page and its
  mlcontests mirror): "Uncover non-obvious linkages between Combine sensor tracking
  and regular-season NFL game performance".
- Judging rubric (retrieved verbatim from the Kaggle overview mirror [snippet]):
  "| Writeup Score (20%) | Is the Writeup well-written? Is the Writeup easy to
  follow? Is the motivation (metric, player evaluation, etc) clearly defined? |
  | Data Visualization Score (20%) | Are the charts and tables provided
  accessible? Are the charts and tables accurate? Are the charts and tables
  innovative? |"
- The overview also states: "Does not award Points or Medals" - i.e., this is a
  JUDGED hackathon (NFL analyst review), NOT a leaderboard competition. We could
  not find any public numeric leaderboard metric (no RMSE/MAE target published).
  The remainder of the rubric (methodology/insight, presumably ~60%) was not fully
  retrievable from the client-rendered page. **Conclusion: there is no official
  numeric competition metric; our model reports a clearly labeled PROXY
  (GroupKFold MAE / R^2 on a held-out career-success target).**
- Prize pool: $100,000 total ($45k Open Track, $45k University Track, $10k Grand
  Prize for finalists presenting at the 2027 NFL Scouting Combine) [reconscribe.com,
  mlcontests mirror].
- Deadline: January 6, 2027 (i-programmer.info: "The competition is open now, and
  final submissions are due by January 6, 2027").

## 2. Submission format

From reconscribe.com (retrieved via websearch, verbatim):
- "The simplest route is a Kaggle Notebook attached to the competition" ... "A free
  Kaggle account and a public Kaggle Notebook, which runs on Kaggle's machines
  inside your browser, are enough."
- "You can use at most nine tables or figures, and Data Visualization is worth 20%
  of the score, so sketch each chart's single message first and cut anything that
  does not earn its place."
- Submission = public Kaggle notebook + writeup reviewed by NFL analysts; no
  prediction CSV leaderboard upload. (Our rules: NEVER submit to Kaggle; this
  workspace only prepares artifacts.)

## 3. Dataset schema (what is publicly documented)

From reconscribe.com (verbatim): "The Data tab holds 9 CSV files, about 2.31 GB in
total, licensed under CC BY-NC 4.0 ... The files are players, combine_results,
combine_tracking, player_career_successes, player_play, games, and game_tracking
files for 2023, 2024 and 2025."

Coverage (verbatim): "The data covers 510 rookies drafted or signed from 2023 to
2025, 6,310 Combine drill attempts tracked at 10 Hz, and 1,002 NFL regular-season
and playoff games."

Host-flagged quirks (verbatim): "13 of the 510 players never reached the field, and
on BALL rows in combine_tracking the nfl_id is the athlete doing the drill. Decide
early how your analysis handles both."

Sensor details (i-programmer.info, verbatim): "NFL combine sensor tracking uses
Radio Frequency Identification (RFID) chips placed in players' shoulder pads to
record real-time movement data every tenth of a second during drills. The system
measures continuous movement, speed, acceleration, deceleration, and distance every
0.1 seconds."

Position groups (operations.nfl.com, verbatim): "The dataset will focus on five
position groups: wide receivers, tight ends, defensive backs, offensive linemen
and defensive linemen."

Exact combine_tracking column list is not fully public in retrievable sources; the
task brief specifies x/y/player_id/frame/game_play/drill and discussion titles
confirm nfl_id + BALL-frame semantics. Our synthetic generator emits
[game_play, frame, nfl_id, x, y, s, a, dis, dir, orientation, event, drill] to be
schema-compatible; adjust to the real columns when data is accessible.

External join key (grindingthemocks.substack.com, verbatim): mock-draft dataset
field "nfl_id - Prospect's NFL ID (this is used in Big Data Bowl Data)".

## 4. Verbatim-grounded key insights

### 4.1 Cut-direction cross-product (discussion 749574)
Direct fetch of https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/749574
returned ONLY the client-rendered shell (title "NFL Big Data Bowl 2027 | Kaggle",
~5.9KB HTML, zero discussion text) - the page body is JS-rendered. Search-index
snippet of that exact URL (retrieved verbatim via websearch):
> "Combine sensor tracking and regular-season NFL game performance. Combine x/y is
> a normal frame where a positive cross product means a left turn"

This independently confirms our core method: sign(v_pre x v_post) gives turn
direction; combined with the drill name's ALIGNMENT side (LEFT/RIGHT), it recovers
inside vs outside break direction.

### 4.2 Discussion 747720 (pre-draft collegiate temporal strictness data)
Direct fetch of .../discussion/747720 also returned only the JS shell; the search
index exposed only the page title/tagline, no thread body. **No verbatim body text
for 747720 could be retrieved; we do not fabricate any.** Related verbatim threads
retrieved from the live discussion index:
- "Data note: 2024 back-pedal reps are truncated, and 2025 combine tracking is a
  different export" (Yumeko) - temporal/export heterogeneity across years.
- "Attempt Count in Combine Tracking Data" (ConorM; last comment by host Tom Bliss)
- "combine_tracking.csv: nfl_id on BALL frames - are we reading this correctly?
  (plus two small notes)" (timersevenwang; host Muhammad Jawad1894 replied)
- "Dataset Row Count Discrepancies Between Data Description and Downloaded CSV
  Files" (WWJY)
- "Clarification on Pass-Rush Attempts and Missing Pressure-Time Values" (Ridhi Jain14)

### 4.3 Null results are acceptable (discussion 749298, host Tom Bliss) [snippet]
> "Yes that passes - in general, we will not grade based purely off what specific
> results are found."
> "Top speed or similar metrics are definitely fine to be used in that context."
And the thread context: "finding 'no relationship' or a null result is still a
valid statistical finding. As long as your methodology, feature engineering, and
feature extraction from the Combine sensor tracking and regular-season data are
rigorous and well-documented, it holds analytical value."

### 4.4 What the organizers want (operations.nfl.com) [snippet]
> "Entering our ninth-annual competition, we are continuing to elevate the Big Data
> Bowl by giving participants the opportunity to evaluate current and former NFL
> Draft prospects," said Mike Lopez ... "The competition will uncover new insights
> into what drives success at the professional level..."

### 4.5 Code page (notebooks tab)
Fetch of .../code returned the JS shell only; no notebook titles were retrievable.

## 5. Public-mirror search result (STEP 2)

- Kaggle API v1 (/api/v1/competitions/list) returns 401 without auth; competition
  data endpoints 403/404 for our token (rules not accepted). NOT bypassed.
- Hugging Face datasets API: search "nfl" (50 results) and "combine tracking"
  (0 results) - NO dataset contains 2027 combine tracking.
- GitHub repo search: 76 repos mention "nfl big data bowl tracking"; the 2027 ones
  (usufalbaz/...-ktd, maxquirk/big-data-bowl, miguelvallejovillegas/..., etc.) are
  code/writeup repos (0-6 stars), none hosts the gated data; GitHub code search API
  requires auth (401).
- Conclusion: **no public mirror of the 2027 combine tracking exists (2026-10-22).**
  Per the task's fallback rule, the pipeline is built and validated against
  src/synth_combine.py (schema-compatible 10 Hz synthetic generator), and every
  number/figure produced from it is labeled SYNTHETIC.
