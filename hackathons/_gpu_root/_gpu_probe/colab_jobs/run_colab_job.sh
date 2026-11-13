#!/usr/bin/env bash
# Generic Colab runner for heavy GPU jobs.
#   ACCOUNT=pro|old GPU_TYPE=A100|T4 MODEL=Qwen/Qwen2.5-14B-Instruct SCRIPT=<local.py> \
#   ARGS="a b c" OUTDIR=<local dir> bash run_colab_job.sh
set -uo pipefail
CG="/home/pope/Desktop/gpu /_gpu_probe"
ACCOUNT="${ACCOUNT:-pro}"
GPU_TYPE="${GPU_TYPE:-T4}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SCRIPT="${SCRIPT:-$CG/colab_jobs/arc_llm_ttc.py}"
OUTDIR="${OUTDIR:-/home/pope/Desktop/gpu /hackathons/04-arc-prize-2026-arc-agi-2/outputs/pro_artifacts}"
UPLOADS="${UPLOADS:-}"
TAG="${TAG:-$(basename "$SCRIPT" .py)}"
S="${TAG}-$(date +%s)"
LOGDIR="${LOGDIR:-/home/pope/Desktop/gpu /hackathons/_gpu_probe/logs}"
mkdir -p "$LOGDIR" "$OUTDIR"
exec > >(tee -a "$LOGDIR/colab_${ACCOUNT}_${TAG}.log") 2>&1
echo "=== $(date -u +%FT%TZ) $ACCOUNT session $S gpu=$GPU_TYPE model=$MODEL script=$SCRIPT ==="

source "$CG/use-colab.sh" "$ACCOUNT"
CREATED=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if colab --config "$COLAB_CONFIG" new -s "$S" --gpu "$GPU_TYPE"; then CREATED=1; break; fi
  echo "create failed (attempt $attempt); backoff 150s"; sleep 150
done
[ "$CREATED" = "1" ] || { echo "SESSION_FAILED"; exit 97; }

# DNS wait: *.prod.colab.dev publishes its A record 10-45s after `colab new`
for i in $(seq 1 40); do
  HOST=$(python3 - "$COLAB_CONFIG" "$S" <<'PYEOF'
import json, sys
cfg, name = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(cfg)); s = d.get(name) or {}
    print((s.get("url") or "").replace("https://", ""))
except Exception:
    print("")
PYEOF
)
  IP=$(python3 -c "
import socket
try: print(socket.getaddrinfo('$HOST', 443, socket.AF_INET)[0][4][0])
except Exception: print('')" 2>/dev/null)
  [ -n "$IP" ] && { echo "DNS ready $HOST -> $IP"; break; }
  echo "dns wait $i"; sleep 3
done

# uploads with retry (the Colab upload endpoint intermittently returns HTTP 500)
if [ -n "$UPLOADS" ]; then
  IFS=',' read -ra UPS <<< "$UPLOADS"
  for pair in "${UPS[@]}"; do
    local_src="${pair%%:*}"; remote_dst="${pair#*:}"
    for u in 1 2 3 4 5; do
      if colab --config "$COLAB_CONFIG" upload -s "$S" "$local_src" "$remote_dst"; then break; fi
      echo "upload retry $u for $local_src"; sleep 20
    done
  done
fi

echo "=== EXEC START $(date -u +%FT%TZ) ==="
colab --config "$COLAB_CONFIG" exec -s "$S" -f "$SCRIPT" --timeout "${TIMEOUT:-14000}"
echo "=== EXEC DONE $(date -u +%FT%TZ) ==="

mkdir -p "$OUTDIR"
for artifact in /content/out.zip /content/out/submission.json /content/out/eval_metrics.json \
                /content/out/oof_by_source.json /content/out/run_meta.json \
                /content/out/oof_real_fold0.npy /content/out/oof_real_idx_fold0.npy \
                /content/out/oof_real_y_fold0.npy /content/out/fold_status.json \
                /content/out/cv_report.json /content/out/_manifest.json; do
  colab --config "$COLAB_CONFIG" download -s "$S" "$artifact" "$OUTDIR/" || true
done
# Pro sessions are precious (A100 cannot be respawned, T4 quota is limited):
# never stop a Pro session - keep it alive for the next job in the pipeline.
if [ "$ACCOUNT" = "pro" ]; then
  echo "PRO SESSION KEPT ALIVE: $S (no stop)"
else
  echo "KEPTALIVE_SESSION" > /dev/null || true       
fi
echo "=== finished $(date -u +%FT%TZ) ==="
