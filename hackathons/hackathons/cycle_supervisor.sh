#!/usr/bin/env bash
# 20-minute cycle supervisor for the marathon.
#   EVENT A: Colab Pro A100 (pro7b-1792644801) LoRA job finished -> pipeline the NEXT job
#            into the SAME session (NEVER stop it - the account cannot respawn A100s).
#   EVENT B: Kaggle slot 1 (ntl-stage2-deberta) COMPLETE -> pull OOF, push arc-agi2-llm-ttc v5.
#   EVENT C: Kaggle slot free -> push the next prepared kernel.
#   Every cycle: refresh MASTER_STATUS.md.
set -uo pipefail
ROOT="/home/pope/Desktop/gpu /hackathons"
CG="/home/pope/Desktop/gpu /_gpu_probe"
T05="$ROOT/05-name-that-llm"
T04="$ROOT/04-arc-prize-2026-arc-agi-2"
LOG="$ROOT/logs/cycle_supervisor.log"
mkdir -p "$ROOT/logs" "$T04/logs" "$T05/logs"
PRO="pro7b-1792644801"
PRO_CFG="$HOME/.config/colab-cli/sessions.pro.json"
NEXT_ARC="$CG/colab_jobs/arc_llm_ttc_selfcontained.py"
exec >> "$LOG" 2>&1
echo "=== cycle supervisor started $(date -u +%FT%TZ) (protected A100 session: $PRO) ==="

while true; do
  CYCLE=$(date -u +%H%M%S)
  echo "----- cycle $CYCLE -----"

  # ---------- EVENT A: A100 job finished -> pipeline next job into the SAME session ----------
  ZIP=$(ls -1t "$T05/outputs/colab_pro_artifacts"/out_*.zip 2>/dev/null | head -1)
  if [ -n "$ZIP" ]; then
    if [ ! -f "$T05/outputs/colab_pro_artifacts/.piped_$(basename "$ZIP")" ]; then
      echo "[A] LoRA artefact present: $ZIP -> unzipping + stacking"
      mkdir -p "$T05/outputs/colab_artifacts"
      if unzip -t "$ZIP" >/dev/null 2>&1; then
        unzip -o -q "$ZIP" -d "$T05/outputs/colab_artifacts" 2>/dev/null
      else
        echo "[A] skipping corrupt artefact $ZIP"
      fi
      touch "$T05/outputs/colab_pro_artifacts/.piped_$(basename "$ZIP")"
      cd "$T05" && timeout 3600 python3 src/stack_v2.py 2>&1 | tail -12
      cd "$T05" && timeout 1800 python3 src/finalize_real.py 2>&1 | tail -6
    fi
  fi
  # is the A100 idle (no training exec running) but the session still alive? -> send the next job
  if [ -n "${A100_PIPED:-}" ]; then
    :
  else
    RECENT=$(find "$T05/outputs/colab_pro_artifacts" -name 'out_*.zip' -mmin -600 2>/dev/null | head -1)
    if [ -n "$RECENT" ]; then
      echo "[A] A100 job looks finished (recent artefact) -> pipelining ARC-AGI-2 TTC into $PRO"
      colab --config "$PRO_CFG" exec -s "$PRO" -f "$NEXT_ARC" --timeout 12000 2>&1 | tail -20
      echo "[A] ARC exec finished $(date -u +%FT%TZ)"
      mkdir -p "$T04/outputs/pro_artifacts"
      colab --config "$PRO_CFG" download -s "$PRO" /content/out.zip "$T04/outputs/pro_artifacts/" || true
      A100_PIPED=1
    fi
  fi

  # ---------- EVENT B: Kaggle slot 1 complete ----------
  ST=$(timeout 90 kaggle kernels status krishnakumarbhat/ntl-stage2-deberta 2>&1 | tail -1)
  echo "[B] ntl-stage2-deberta: $ST"
  case "$ST" in
    *CANCEL*|*ERROR*|*FAIL*)
      echo "[B] deberta kernel $ST -> slot freed, ARC v5 can use it" ;;
    *COMPLETE*)
      if [ ! -d "$T05/outputs/kaggle_deberta" ]; then
        echo "[B] pulling DeBERTa OOF"
        mkdir -p "$T05/outputs/kaggle_deberta"
        timeout 900 kaggle kernels output krishnakumarbhat/ntl-stage2-deberta \
          -p "$T05/outputs/kaggle_deberta" 2>&1 | tail -3
        cd "$T05" && timeout 3600 python3 src/stack_v2.py 2>&1 | tail -12
        cd "$T05" && timeout 1800 python3 src/finalize_real.py 2>&1 | tail -6
      fi
      ;;
  esac

  # ---------- EVENT C: push arc-agi2-llm-ttc v5 into a free Kaggle GPU slot ----------
  ARCST=$(timeout 90 kaggle kernels status krishnakumarbhat/arc-agi2-llm-ttc 2>&1 | tail -1)
  echo "[C] arc-agi2-llm-ttc: $ARCST"
  case "$ARCST" in
    *COMPLETE*|*ERROR*|*CANCEL*|*FAIL*)
      if [ ! -f "$T04/logs/.arc_v5_pushed" ]; then
        echo "[C] pushing arc-agi2-llm-ttc v5 (parser fix + 4x batching)"
        cd "$T04/kaggle_ttc" && timeout 300 kaggle kernels push -p . 2>&1 | tail -2 \
          && touch "$T04/logs/.arc_v5_pushed"
      fi
      ;;
  esac

  # ---------- harvest the Track 05 Qwen-LoRA kernel (7B) when it completes ----------
  QST=$(timeout 90 kaggle kernels status krishnakumarbhat/ntl-qwen-lora 2>&1 | tail -1)
  echo "[C3] ntl-qwen-lora: $QST"
  case "$QST" in
    *COMPLETE*)
      if [ ! -f "$T05/outputs/kaggle_qwen_lora/.pulled" ]; then
        echo "[C3] pulling 7B LoRA OOF"
        mkdir -p "$T05/outputs/kaggle_qwen_lora"
        timeout 1200 kaggle kernels output krishnakumarbhat/ntl-qwen-lora -p "$T05/outputs/kaggle_qwen_lora" 2>&1 | tail -2
        [ -f "$T05/outputs/kaggle_qwen_lora/cv_report.json" ] && touch "$T05/outputs/kaggle_qwen_lora/.pulled"
        cd "$T05" && timeout 3600 python3 src/stack_v2.py 2>&1 | tail -12
        cd "$T05" && timeout 1800 python3 src/make_submission.py --test data/competition/test.csv --out outputs/submission.csv 2>&1 | tail -5
      fi
      ;;
  esac

  # ---------- push the Track 05 Qwen-LoRA recovery kernel when a GPU slot frees ----------
  NTLST=$(timeout 90 kaggle kernels status krishnakumarbhat/ntl-real-deberta 2>&1 | tail -1)
  echo "[C2] ntl-real-deberta: $NTLST"
  case "$NTLST" in
    *COMPLETE*|*CANCEL*|*ERROR*|*FAIL*)
      if [ ! -f "$T05/logs/.qwen_lora_pushed" ]; then
        echo "[C2] pushing ntl-qwen-lora (A100-recipe replication on Kaggle T4)"
        cd "$T05/kaggle_qwen_lora" 2>/dev/null && timeout 300 kaggle kernels push -p . 2>&1 | tail -2 \
          && touch "$T05/logs/.qwen_lora_pushed"
      fi
      ;;
  esac

  # ---------- local ARC evaluation-split benchmark (honest non-zero target) ----------
  if [ ! -f "$T04/logs/.eval_bench_done" ]; then
    if cd "$T04" && PYTHONPATH="$T04" timeout 3000 python3 src/run_v2.py --split evaluation > "$T04/logs/eval_bench.log" 2>&1; then
      touch "$T04/logs/.eval_bench_done"
      echo "[D] ARC evaluation-split benchmark done"
    fi
  fi

  # ---------- status ----------
  cd "$ROOT" && python3 gen_master_status.py 2>&1 | tail -1
  cp "$ROOT/MASTER_STATUS.md" /home/pope/Desktop/gpu /hackathons/MASTER_STATUS.md 2>/dev/null
  sleep 1200
done
