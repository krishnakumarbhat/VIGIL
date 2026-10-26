#!/usr/bin/env bash
# One command that reproduces the submission artefacts.
#
#   bash scripts/run_all.sh              tests + benchmark + demo + local levers + evidence
#   bash scripts/run_all.sh --with-arm64 additionally run the QEMU aarch64 probe
#                                        (slow, downloads a cached toolchain)
#
# The ARM probe is emulated and is never treated as Graviton evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_ARM=0
[[ "${1:-}" == "--with-arm64" ]] && WITH_ARM=1

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export VIGIL_BENCH_REPS="${VIGIL_BENCH_REPS:-3}"

echo "== 1/5 tests =="
python3 -m pytest -p no:cacheprovider

echo "== 2/5 benchmark =="
python3 benchmarks/run_eval.py

echo "== 3/5 annotated demo =="
python3 src/generate_demo_video.py \
  --out data/demo/demo.mp4 \
  --annotated-out outputs/demo_annotated.mp4 \
  --telemetry outputs/telemetry/demo.jsonl \
  --summary outputs/demo_summary.json

echo "== 4/5 measured optimisation levers =="
python3 benchmarks/cool_local.py --out outputs/cool_local.json
python3 benchmarks/arch_probe.py --out outputs/arch_probe_x86_64.json

if [[ "$WITH_ARM" == "1" || -f outputs/arch_probe_aarch64.json ]]; then
  if [[ "$WITH_ARM" == "1" ]]; then
    echo "== 5/5 aarch64 probe (QEMU, emulated) =="
    bash scripts/run_arm64_emulated.sh --out outputs/arch_probe_aarch64.json
  fi
  echo "== building evidence =="
  python3 scripts/build_cool_evidence.py
else
  echo "== 5/5 evidence skipped: no aarch64 probe =="
  echo "   run: bash scripts/run_all.sh --with-arm64"
fi

echo
echo "Artefacts:"
ls -lh outputs/benchmarks.json outputs/benchmarks.csv outputs/demo_annotated.mp4 \
  outputs/cool_local.json outputs/cool_evidence.json 2>/dev/null || true
