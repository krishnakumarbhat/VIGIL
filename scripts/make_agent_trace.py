"""Build the Agentic-Vision trace artifacts from measured telemetry.

Two traces are emitted, both reproducible by running this script:

1. ``outputs/agent_trace.md`` — the real demo run, including the one frame where
   an OpenCV 5 perception result changed a later decision (frame 166: motion
   anomaly with no detection -> forensic inspection -> ROI zoom -> rollback).
2. ``outputs/agent_trace_escalation.jsonl`` — an eleven-frame deterministic
   scenario in which a persistent unknown identity next to a motion anomaly
   reaches the human-escalation policy.  This exercises the human-control path
   that the twelve-second demo does not reach.

The script asserts the counts it prints against the files on disk, so the
markdown cannot drift from the telemetry again.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.agent import AutonomousVisionAgent  # noqa: E402
from src.cloud import OfflineCloud  # noqa: E402
from src.config import VisionConfig  # noqa: E402
from src.perception.motion import MotionReport  # noqa: E402
from tests.test_agent import FakeMotion, FakePipeline, _frame  # noqa: E402
from tests.conftest import make_hit  # noqa: E402

OUT = ROOT / "outputs"
DEMO = OUT / "telemetry" / "demo.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def demo_section() -> tuple[str, dict]:
    rows = load(DEMO)
    assert len(rows) == 288, f"expected 288 demo frames, found {len(rows)}"
    kinds = Counter(a["kind"] for r in rows for a in r["agent"]["actions"])
    adaptive = sum(1 for r in rows if r["agent"]["adaptive_steps"])
    steps = sum(r["agent"]["adaptive_steps"] for r in rows)
    forensic = sum(1 for r in rows if r["agent"]["forensic"])
    escalated = [r["frame_index"] for r in rows if r["agent"]["escalated"]]
    rollbacks = kinds.get("rollback", 0)
    ids = Counter(f["identity"] for r in rows for f in r["faces"])
    budget = sum(1 for r in rows if r["agent"]["budget_exhausted"])
    assert escalated == [], "demo unexpectedly escalates; update the narrative"
    assert kinds == Counter({"forensic_inspection": 1, "zoom_roi": 1, "rollback": 1}), kinds

    def row(r: dict) -> str:
        faces = r["faces"]
        ident = ",".join(str(f["identity"] or "unknown") for f in faces) or "-"
        q = ",".join(f"{f['quality']:.2f}" for f in faces) or "-"
        acts = ",".join(a["kind"] for a in r["agent"]["actions"]) or "-"
        path = "→".join(r["agent"]["state_path"])
        return (
            f"| {r['frame_index']} | {len(faces)} | {ident} | {q} | "
            f"{r['motion']['anomaly_z']:.2f} | {path} | {acts} | {r['verification']['notes']} |"
        )

    header = (
        "| frame | faces | identity | quality | motion z | state path | actions | verification |\n"
        "|---:|---:|---|---:|---:|---|---|---|\n"
    )
    win_a = "\n".join(row(r) for r in rows if 160 <= r["frame_index"] <= 172)
    win_b = "\n".join(row(r) for r in rows if 252 <= r["frame_index"] <= 268)

    f166 = next(r for r in rows if r["frame_index"] == 166)
    excerpt = {
        "frame_index": f166["frame_index"],
        "motion": f166["motion"],
        "agent": f166["agent"],
        "verification": f166["verification"],
    }

    section = f"""
## 1. Measured demo trace — where an OpenCV 5 result changes a decision

Run `{rows[0]['run_id']}` · `{rows[0]['backend']}` · 288 frames · source
`data/demo/demo.mp4`. Every value below is read from
`outputs/telemetry/demo.jsonl`; the counts are asserted by
`scripts/make_agent_trace.py`.

| Measured quantity | Value |
|---|---:|
| Frames | 288 |
| Frames with at least one adaptive action | {adaptive} |
| Adaptive steps (total) | {steps} |
| Action kinds | {", ".join(f"{k}×{v}" for k, v in sorted(kinds.items()))} |
| Forensic-inspection frames | {forensic} |
| Rollbacks | {rollbacks} |
| Human escalations | {len(escalated)} |
| CPU-budget-exhausted frames | {budget} |
| Face records | {sum(len(r['faces']) for r in rows)} (employee_a {ids['employee_a']}, employee_b {ids['employee_b']}, unknown {ids[None]}) |

**The causal chain.** Frame 166 is the clearest case. Motion analysis reports
`z=40.27` while YuNet returns **no face at 640 px**. That perception result
raises two reason codes (`motion_anomaly`, `no_face_with_motion`), and those
codes change the agent's next action: a forensic recovery scan at a lower
detector threshold, then an ROI zoom. Verification rejects the zoom (no new
face, IoU 0), and the agent rolls the frame back to the initial result instead
of accepting a hallucinated detection.

```json
{json.dumps(excerpt, indent=2)}
```

Frames 160–172 (the decision window):

{header}{win_a}

Frames 252–268 (blur / low light / persistent unknown identity):

{header}{win_b}

Why no escalation in this clip: the policy requires **both** a persistent
unknown identity (≥10 consecutive frames, quality ≥0.40) **and** a critical cue
inside the anomaly window (spoof suspicion, motion anomaly, or scene
degradation), plus a 120-frame cooldown. At frame 263 the last motion anomaly
(frame 166, window until 211) has expired and liveness is not suspect, so the
agent keeps the identity unknown and does not invent a human review. Section 2
shows the policy firing when both conditions genuinely hold.
"""
    return section, {"kinds": dict(kinds), "escalated": escalated, "rows": rows}


def escalation_section() -> tuple[str, list[dict]]:
    cfg = VisionConfig()
    face = make_hit(box=(70, 50, 110, 110), score=0.9, identity=None, quality=0.8, liveness=0.05)
    motion = FakeMotion(MotionReport(moving=True, anomaly_z=5.0, roi=(40, 30, 160, 160)))
    agent = AutonomousVisionAgent(
        cfg, FakePipeline([face]), motion=motion, cloud=OfflineCloud(cfg), run_id="run-escalation-trace"
    )
    frame = _frame()
    records: list[dict] = []
    for idx in range(11):
        outcome = agent.process(frame, frame_index=idx, source="synthetic:persistent_unknown")
        records.append(json.loads(outcome.telemetry.model_dump_json()))

    fired = [r["frame_index"] for r in records if r["agent"]["escalated"]]
    assert len(fired) == 1, f"expected exactly one escalation, got {fired}"
    at = fired[0]
    assert records[at]["agent"]["actions"][-1]["kind"] == "escalate_human"
    assert "ESCALATE" in records[at]["agent"]["state_path"]

    (OUT / "agent_trace_escalation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    table = "\n".join(
        f"| {r['frame_index']} | {r['faces'][0]['identity'] or 'unknown'} | "
        f"{r['faces'][0]['liveness']:.2f} | {r['motion']['anomaly_z']:.1f} | "
        f"{'→'.join(r['agent']['state_path'])} | "
        f"{','.join(a['kind'] for a in r['agent']['actions']) or '-'} | "
        f"{r['agent']['escalated']} |"
        for r in records
    )
    section = f"""
## 2. Human-control trace — the escalation path, exercised

Deterministic eleven-frame scenario (`scripts/make_agent_trace.py`), synthetic
frame, no model weights: an unknown face (`liveness=0.05`, i.e. spoof-suspect)
in front of a motion anomaly (`z=5.0`). One validated record per frame is in
`outputs/agent_trace_escalation.jsonl`.

| frame | identity | liveness | motion z | state path | actions | escalated |
|---:|---|---:|---:|---|---|---|
{table}

At frame {at} the two independent conditions are both true — ten consecutive
unknown frames **and** an active critical cue — so the agent appends
`escalate_human`, writes a `human_review` reason, and calls the cloud incident
hook. The hook is wrapped, so a network failure cannot take down the local
loop (covered by
`tests/test_agent.py::test_persistent_unknown_near_anomaly_escalates_and_survives_cloud_failure`).

What this proves for human control:

- identity ambiguity alone is **not** enough to escalate (frame 263–268 stay
  unknown without a review);
- escalation requires persistence *and* a critical cue, so it is not a random
  false alarm generator;
- a 120-frame cooldown prevents escalation storms;
- the incident record carries a digest, not an embedding, and no raw frame
  leaves the device by default.
"""
    return section, records


def main() -> None:
    demo_md, demo = demo_section()
    esc_md, records = escalation_section()
    header = f"""# VIGIL-5 — agent trace (measured)

Generated by `scripts/make_agent_trace.py`. Regenerate with:

```bash
cd "{ROOT}"
python3 scripts/make_agent_trace.py
```

Artifacts: `outputs/telemetry/demo.jsonl` (288 real frames),
`outputs/agent_trace_escalation.jsonl` (11-frame synthetic policy trace).
"""
    (OUT / "agent_trace.md").write_text(header + demo_md + esc_md)
    print("wrote outputs/agent_trace.md")
    print("demo action counts:", demo["kinds"], "| demo escalations:", len(demo["escalated"]))
    print("escalation frames:", [r["frame_index"] for r in records if r["agent"]["escalated"]])


if __name__ == "__main__":
    main()
