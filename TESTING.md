# VIGIL-5 — build, run, test

Everything below was executed on the submission host (Ubuntu 22.04, x86_64,
Python 3.10.12, OpenCV 5.0.0.93). The AWS path is offline-validated only; see
`aws/README.md`.

## 1. Environment

```bash
git clone <repository-url> vigil-5
cd vigil-5
python3 -m venv .venv && . .venv/bin/activate   # optional; a container works too
pip install -r requirements.txt
```

Model weights (OpenCV Zoo, ~38 MB) are **not** committed:

```bash
mkdir -p data/models
curl -L -o data/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o data/models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

If a model is missing the pipeline degrades through documented fallbacks and
records the backend in every telemetry row; it does not crash.

## 2. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

Expected on the build host: **61 passed** (no failures, no skips).

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required only because a third-party pytest
plugin installed in the shared interpreter of the build host fails to import
(`langsmith`). The project itself has no `langsmith` dependency; a clean
environment runs `python3 -m pytest` directly. `scripts/run_all.sh` exports the
variable for you.

Coverage by file:

| File | Covers |
|---|---|
| `tests/test_agent.py` | state path, resolution ladder, forensic inspection, escalation + cloud-failure isolation, cooldown |
| `tests/test_perception.py` | detection, quality, liveness proxy, identity matching |
| `tests/test_engine.py` | end-to-end engine on the demo clip, telemetry schema |
| `tests/test_submission_artifacts.py` | report/video/evidence artefacts exist and agree |
| `tests/test_aws_assets.py` | Dockerfiles, task definitions, Terraform, Lambda handler (static) |
| `tests/test_benchmarks.py` | benchmark harness and CSV/JSON writers |
| `tests/test_cloud.py` | offline cloud, incident recording, failure isolation |
| `tests/test_motion_quality.py` | motion anomaly, quality gates |

## 3. Reproduce the demo and its evidence

```bash
python3 src/generate_demo_video.py \
  --out data/demo/demo.mp4 \
  --annotated-out outputs/demo_annotated.mp4 \
  --telemetry outputs/telemetry/demo.jsonl \
  --summary outputs/demo_summary.json

python3 scripts/make_agent_trace.py          # asserts counts, writes agent_trace.md
```

Expected: 288 telemetry records, 1 forensic frame, 3 action records
(`forensic_inspection`, `zoom_roi`, `rollback`), 0 escalations in the demo and
exactly 1 escalation at frame 9 of the synthetic trace.

## 4. Reproduce the performance numbers

```bash
python3 benchmarks/run_eval.py               # exit 0, writes benchmarks.json/.csv
```

Expected headline values on an idle 4-core host: YuNet detection ROC-AUC
0.9917, full agent 20.22 FPS / 45.04 ms CPU p50, fused pipeline 24.81 FPS /
36.21 ms p50, event recall 1.00 vs 0.00 for the static baseline. Timings are
host-sensitive; AUC and recall are deterministic. The JSON retains per-repetition
spread and the load average that was measured.

## 5. Local optimisation evidence (x86 proxies, not COOL)

```bash
python3 benchmarks/cool_local.py --out outputs/cool_local.json
python3 benchmarks/arch_probe.py --out outputs/arch_probe_x86_64.json
```

The aarch64 probe is emulated; see `outputs/cool_evidence.json` →
`arm64_execution`. **No Graviton/COOL result is claimed.**

## 6. One command

```bash
bash scripts/run_all.sh          # tests + benchmark + demo + levers + evidence
```

Artifacts land in `outputs/`. The script fails fast on any non-zero step.
