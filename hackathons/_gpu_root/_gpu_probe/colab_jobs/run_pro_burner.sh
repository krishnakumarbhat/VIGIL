#!/usr/bin/env bash
set -uo pipefail
H="/home/pope/Desktop/gpu /hackathons"
CG="/home/pope/Desktop/gpu /_gpu_probe"
JOB="${JOB:-pro7b}"
S="${JOB}-$(date +%s)"
LOGDIR="$H/05-name-that-llm/logs"; mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/colab_${JOB}.log") 2>&1
echo "=== $(date -u +%FT%TZ) PRO session $S gpu=${GPU_TYPE:-A100} ==="
source "$CG/use-colab.sh" pro
CREATED=0
for attempt in 1 2 3 4 5 6 7 8; do
  if colab --config "$COLAB_CONFIG" new -s "$S" --gpu "${GPU_TYPE:-A100}"; then CREATED=1; break; fi
  echo "create failed (attempt $attempt); backoff"; sleep 120
done
[ "$CREATED" = "1" ] || { echo "PRO_SESSION_FAILED"; exit 97; }
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
echo "=== EXEC START $(date -u +%FT%TZ) ==="
colab --config "$COLAB_CONFIG" exec -s "$S" -f "$CG/colab_jobs/track05_colab_job_v2.py" --timeout ${TIMEOUT:-17000}
echo "=== EXEC DONE $(date -u +%FT%TZ) ==="
mkdir -p "$H/05-name-that-llm/outputs/colab_pro_artifacts"
colab --config "$COLAB_CONFIG" download -s "$S" /content/out.zip "$H/05-name-that-llm/outputs/colab_pro_artifacts/" || true
colab --config "$COLAB_CONFIG" stop -s "$S" || true
echo "=== finished $(date -u +%FT%TZ) ==="
