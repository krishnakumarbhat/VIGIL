#!/usr/bin/env bash
# Periodically pull /content/out.zip from the live Colab Pro session so partial
# LoRA fold artefacts survive even if the session dies.
set -uo pipefail
CG="/home/pope/Desktop/gpu /_gpu_probe"
H="/home/pope/Desktop/gpu /hackathons"
DEST="$H/05-name-that-llm/outputs/colab_pro_artifacts"
mkdir -p "$DEST"
source "$CG/use-colab.sh" pro
for i in $(seq 1 200); do
  for S in $(python3 - <<'PY'
import json
try:
    d = json.load(open("/home/pope/.config/colab-cli/sessions.pro.json"))
    print(" ".join(d.keys()))
except Exception:
    print("")
PY
); do
    [ -z "$S" ] && continue
    TS=$(date -u +%H%M%S)
    colab --config "$COLAB_CONFIG" download -s "$S" /content/out.zip "$DEST/out_${TS}.zip" >/dev/null 2>&1 \
      && echo "$(date -u +%FT%TZ) pulled out.zip from $S" \
      || echo "$(date -u +%FT%TZ) no out.zip yet from $S"
  done
  sleep 300
done
