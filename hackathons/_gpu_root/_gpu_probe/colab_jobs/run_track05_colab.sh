#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pope/Desktop/gpu "
H="/home/pope/Desktop/gpu /hackathons"
CG="/home/pope/Desktop/gpu /_gpu_probe"
JOB="${JOB:-track05}"
S="${JOB}-$(date +%s)"
LOGDIR="$H/05-name-that-llm/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/colab_${JOB}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u +%FT%TZ) session $S ==="
ACCT="${COLAB_ACCOUNT:-pro}"
source "$CG/use-colab.sh" "$ACCT"
CREATED=0
for attempt in 1 2 3 4 5 6; do
  if colab --config "$COLAB_CONFIG" new -s "$S" --gpu "${COLAB_GPU_TYPE:-T4}"; then CREATED=1; break; fi
  echo "session create failed (attempt $attempt) - backing off"; sleep 90
done
[ "$CREATED" = "1" ] || exit 97
for i in $(seq 1 40); do
  H2=$(python3 - "$COLAB_CONFIG" "$S" <<'PYEOF'
import json,sys
cfg,name=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(cfg)); s=d.get(name) or {}; print((s.get("url") or "").replace("https://",""))
except Exception: print("")
PYEOF
)
  ip=$(python3 -c "
import socket
try: print(socket.getaddrinfo('$H2',443,socket.AF_INET)[0][4][0])
except Exception: print('')" 2>/dev/null)
  [ -n "$ip" ] && { echo "DNS ready -> $ip"; break; }
  echo "dns wait $i"; sleep 3
done
colab --config "$COLAB_CONFIG" upload -s "$S" "$H/05-name-that-llm/data/compact/train_all_v2.parquet" /content/train_all_v2.parquet
colab --config "$COLAB_CONFIG" upload -s "$S" "$H/05-name-that-llm/data/compact/labels.json" /content/labels.json
echo "=== SANITY EXEC (local file path) ==="
colab --config "$COLAB_CONFIG" exec -s "$S" -f "$CG/colab_jobs/sanity.py" --timeout 180 2>&1 | tail -12
echo "=== MAIN EXEC $(date -u +%FT%TZ) ==="
colab --config "$COLAB_CONFIG" exec -s "$S" -f "$CG/colab_jobs/track05_colab_job.py" --timeout ${TIMEOUT:-18000}
echo "=== MAIN EXEC DONE $(date -u +%FT%TZ) ==="
mkdir -p "$H/05-name-that-llm/outputs/colab_artifacts"
colab --config "$COLAB_CONFIG" download -s "$S" /content/out.zip "$H/05-name-that-llm/outputs/colab_artifacts/" || true
colab --config "$COLAB_CONFIG" stop -s "$S" || true
echo "=== finished $(date -u +%FT%TZ) ==="
