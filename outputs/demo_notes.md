# VIGIL-5 — demo notes

**Clip:** `data/demo/demo.mp4` — 12.0 s, 640×360, 24 fps, 288 frames.
**Output:** `outputs/demo_annotated.mp4` (same length, overlays + state banner).
**Telemetry:** `outputs/telemetry/demo.jsonl` — one validated
`vigil.telemetry/1.0` record per frame.
**Run summary:** `outputs/demo_summary.json` (run id
`run-20261022T140409Z-27336c`, generated `2026-10-22T14:04:58Z`).
**Backend:** `yunet+sface` (both model files present; no fallback used).

> Every number on this page is read from the two files above. Regenerate the
> trace table with `python3 scripts/make_agent_trace.py`; the script asserts the
> counts against the telemetry before writing `outputs/agent_trace.md`.

## What to look for

The clip is synthetic: licence-clean portraits are composited onto a generated
concourse, with two enrolled employees (`employee_a`, `employee_b`) and one
unscripted identity (`visitor_c`). Nothing in the clip is real surveillance
footage.

| Event | Frames | Time | Observed system behaviour |
|---|---|---|---|
| Occlusion | 150–172 | 6.25–7.17 s | At frame 166 the face is behind the pillar: YuNet returns 0 detections while motion z = 40.27. The agent runs a forensic recovery scan, then an ROI zoom; verification rejects the zoom and rolls the frame back. |
| Fast entry | 188–210 | 7.83–8.75 s | Face enters at the right edge; tracking keeps the identity, no adaptive action needed. |
| Low light | 236–248 | 9.83–10.33 s | Sharpness drops; quality reason code is raised, but the face stays identifiable, so the budget is not spent. |
| Blur / unknown visitor | 252–268 | 10.50–11.17 s | The unknown visitor is detected at quality 0.39–0.60 and stays `unknown`; no escalation fires (see below). |

## Final demo metrics (measured, not estimated)

| Metric | Value |
|---|---|
| Frames processed | 288 |
| Wall-clock | 48.045 s (5.99 FPS end-to-end including encode) |
| Mean / p50 / p95 / max latency | 158.62 / 140.40 / 321.69 / 461.15 ms |
| Face records | 288 (161 `employee_a`, 72 `employee_b`, 55 unknown) |
| Adaptive frames / steps | 1 frame / 2 steps (frame 166 only) |
| Actions | 1 `forensic_inspection`, 1 `zoom_roi`, 1 `rollback` |
| Forensic-inspection frames | 1 |
| Human escalations | 0 |
| Degraded frames | 0 |
| CPU-budget-exhausted frames | 35 |

Reason codes: 212 `baseline`, 35 `budget_pressure`, 28 `unknown_identity`,
27 `low_resolution`, 17 `low_quality`, 9 `motion_anomaly`,
9 `no_face_with_motion`, 4 `low_confidence`, 1 `spoof_suspect`.

Per-event coverage (frames / detections / forensic frames / adaptive steps):

- Occlusion 150–172: 23 frames, 28 detections, 1 forensic frame, 2 adaptive steps.
- Fast entry 188–210: 23 frames, 17 detections, 0 forensic frames, 0 adaptive steps.
- Low light 236–248: 13 frames, 3 detections, 0 forensic frames, 0 adaptive steps.
- Blur / unknown 252–262: 11 frames, 9 detections, 0 forensic frames, 0 adaptive steps.

All four scripted events are hit by the full agent (`event_recall = 1.0`); the
static 320 px baseline hits none (`event_recall = 0.0`). The headline benchmark
numbers, including the per-configuration table, are in
`outputs/benchmarks.json` and `outputs/DEVPOST_SUBMISSION.md`.

## Why the demo does not escalate

The escalation policy requires **both** a persistent unknown identity
(≥10 consecutive frames with quality ≥0.40) **and** an active critical cue
(spoof suspicion, motion anomaly, or scene degradation), plus a 120-frame
cooldown. The unknown visitor appears at frame 252, after the only motion
anomaly window (frame 166, active until 211) has expired, and the liveness
signal is not suspect. The agent therefore keeps the identity `unknown` and does
not manufacture a human review.

The escalation path itself is exercised deterministically in
`outputs/agent_trace_escalation.jsonl` (11 synthetic frames, escalation at
frame 9) and by
`tests/test_agent.py::test_persistent_unknown_near_anomaly_escalates_and_survives_cloud_failure`.
See `outputs/agent_trace.md` §2. This is the conservative behaviour we want:
identity ambiguity alone never pages a human.

## Annotated video legend

- **Green box + name**: matched enrolled identity, cosine ≥ threshold.
- **Amber box + `unknown`**: face detected, no gallery match above threshold
  (visible on the visitor from frame 252 onward).
- **Cyan corners**: the current agent action is a forensic inspection or an ROI
  zoom (visible at frame 166).
- **Banner text**: agent state path (`SEE → REASON → ACT → VERIFY`), backend,
  latency, motion z-score and face count.
- **Red incident tag**: human escalation. The capability exists, but **no
  escalation fires in this clip** (`demo_summary.json` → `"incidents": []`).

## Reproduce

```bash
cd "/home/pope/Desktop/gpu /hackathons/07-opencv-ai-competition-2026"

# one command: raw clip + annotated MP4 + telemetry + summary
python3 src/generate_demo_video.py \
  --out data/demo/demo.mp4 \
  --annotated-out outputs/demo_annotated.mp4 \
  --telemetry outputs/telemetry/demo.jsonl \
  --summary outputs/demo_summary.json

# or re-annotate an existing clip only
python3 -m src.engine \
  --video data/demo/demo.mp4 \
  --gallery data/demo/gallery \
  --out outputs/demo_annotated.mp4 \
  --telemetry outputs/telemetry/demo.jsonl \
  --summary outputs/demo_summary.json

# trace tables + escalation trace (asserts the counts below)
python3 scripts/make_agent_trace.py

# tests (61 pass on the build host)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests

# full benchmark (3 repetitions; writes outputs/benchmarks.json + CSV + experiments.tsv)
python3 benchmarks/run_eval.py
```

Quick checks:

```bash
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open('outputs/telemetry/demo.jsonl')]
print(len(rows))                                            # 288
print(sum(1 for r in rows if r['agent']['forensic']))       # 1
print([r['frame_index'] for r in rows if r['agent']['escalated']])  # []
PY
```

## Caveats

- Synthetic actors and a synthetic scene; this is a behaviour demonstration,
  not a field validation. Recognition scores on 60 pairs are a regression
  check, not an open-world accuracy claim.
- The liveness signal is a texture/motion/moiré proxy, not certified PAD.
- Cloud calls in this run used the deterministic `OfflineCloud`; the video was
  produced with no AWS credentials and no network. See `aws/README.md` for the
  live path.
- The run was taken under heavy shared-host load (load average ~47), which is
  why the end-to-end latency is 158 ms rather than the 36–45 ms measured in
  `benchmarks.json` on a quiet host. Both numbers are real; they measure
  different conditions and are labelled as such.
- Rendering adds wall-clock time on top of the engine loop, so the end-to-end
  FPS is not directly comparable with `benchmarks.json`.
