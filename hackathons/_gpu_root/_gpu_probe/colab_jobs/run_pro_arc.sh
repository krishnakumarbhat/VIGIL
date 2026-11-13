#!/usr/bin/env bash
set -uo pipefail
H="/home/pope/Desktop/gpu /hackathons"
CG="/home/pope/Desktop/gpu /_gpu_probe"
ARC="$H/04-arc-prize-2026-arc-agi-2/data/competition"
S="proarc-$(date +%s)"
exec > >(tee -a "$H/04-arc-prize-2026-arc-agi-2/logs/colab_pro_arc.log") 2>&1
echo "=== $(date -u +%FT%TZ) PRO ARC session $S gpu=${GPU_TYPE:-T4} ==="
source "$CG/use-colab.sh" pro
CREATED=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if colab --config "$COLAB_CONFIG" new -s "$S" --gpu "${GPU_TYPE:-T4}"; then CREATED=1; break; fi
  echo "create failed (attempt $attempt); backoff 120s"; sleep 120
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
  sleep 3
done
colab --config "$COLAB_CONFIG" upload -s "$S" "$ARC/arc-agi_evaluation_challenges.json" /content/arc/eval_ch.json
colab --config "$COLAB_CONFIG" upload -s "$S" "$ARC/arc-agi_evaluation_solutions.json" /content/arc/eval_sol.json
colab --config "$COLAB_CONFIG" upload -s "$S" "$ARC/arc-agi_test_challenges.json" /content/arc/test_ch.json
echo "=== EXEC START $(date -u +%FT%TZ) ==="
colab --config "$COLAB_CONFIG" exec -s "$S" -f "$CG/colab_jobs/arc_llm_ttc.py" --timeout ${TIMEOUT:-15000}
echo "=== EXEC DONE $(date -u +%FT%TZ) ==="
mkdir -p "$H/04-arc-prize-2026-arc-agi-2/outputs/pro_artifacts"
colab --config "$COLAB_CONFIG" download -s "$S" /content/out.zip "$H/04-arc-prize-2026-arc-agi-2/outputs/pro_artifacts/" || true
colab --config "$COLAB_CONFIG" stop -s "$S" || true
echo "=== finished $(date -u +%FT%TZ) ==="
