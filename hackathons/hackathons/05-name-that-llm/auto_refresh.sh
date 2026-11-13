#!/usr/bin/env bash
# Auto-refresh Track 05: when the Colab Pro LoRA / Kaggle DeBERTa artefacts land,
# rebuild the cross-fitted stack and regenerate outputs/submission.parquet + cv_report.json.
# Runs every 10 minutes, forever (until stopped).
set -uo pipefail
T="/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
LOG="$T/logs/auto_refresh.log"
exec >> "$LOG" 2>&1
echo "=== watcher start $(date -u +%FT%TZ) ==="
LAST=""
while true; do
  # 1) pull the Kaggle DeBERTa kernel output if the kernel has finished
  if [ ! -d "$T/outputs/kaggle_deberta" ]; then
    ST=$(timeout 60 kaggle kernels status krishnakumarbhat/ntl-stage2-deberta 2>&1 | tail -1)
    case "$ST" in
      *COMPLETE*)
        mkdir -p "$T/outputs/kaggle_deberta"
        timeout 900 kaggle kernels output krishnakumarbhat/ntl-stage2-deberta \
          -p "$T/outputs/kaggle_deberta" >> "$LOG" 2>&1 && echo "$(date -u +%FT%TZ) pulled deberta" ;;
    esac
  fi
  # 2) rebuild the stack if any new artefact appeared
  SIG=$(ls -1 "$T/outputs/colab_pro_artifacts" "$T/outputs/collab_artifacts" \
        "$T/outputs/kaggle_deberta" 2>/dev/null | md5sum | cut -c1-16)
  if [ "$SIG" != "$LAST" ] && [ -n "$SIG" ]; then
    LAST="$SIG"
    echo "$(date -u +%FT%TZ) artefact set changed ($SIG) -> rebuilding stack"
    # canonical pipeline: cross-fitted blend (temps + weights + per-class bias) then the recipe-
    # driven submission builder. The old final_stack/finalize_real/real_row_blend chain is
    # deliberately NOT called: real_row_blend overwrote submission.parquet with the superseded
    # 0.40 blend.
    cd "$T" && timeout 3600 python3 src/stack_v2.py >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) stack_v2 rc=$?"
    cd "$T" && timeout 3600 python3 src/make_submission.py --test data/competition/test.csv \
        --out outputs/submission.csv >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) make_submission rc=$?"
    cd "/home/pope/Desktop/gpu /hackathons" && python3 gen_master_status.py >> "$LOG" 2>&1
  fi
  sleep 600
done
