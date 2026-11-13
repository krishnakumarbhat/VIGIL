#!/usr/bin/env bash
# Re-blend automatically once the seed-7 LoRA member has all five folds on disk.
set -uo pipefail
T="/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
LOG="$T/logs/blend_when_ready.log"
exec >> "$LOG" 2>&1
echo "=== watch start $(date -u +%FT%TZ) ==="
for i in $(seq 1 90); do
  if [ -f "$T/outputs/oof_local_qwen05b_s7_fold4.npy" ]; then
    echo "$(date -u +%FT%TZ) seed-7 member complete -> re-blending"
    cd "$T" && python3 src/stack_v2.py 2>&1 | tail -25
    echo "$(date -u +%FT%TZ) stack_v2 rc=$?"
    cd "$T" && python3 src/make_submission.py --test data/competition/test.csv \
       --out outputs/submission.csv 2>&1 | grep -avE "Loading weights|it/s" | tail -12
    echo "$(date -u +%FT%TZ) make_submission rc=$?"
    cd "/home/pope/Desktop/gpu /hackathons" && python3 gen_master_status.py 2>&1 | tail -2
    exit 0
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) timeout: seed-7 member did not complete"
