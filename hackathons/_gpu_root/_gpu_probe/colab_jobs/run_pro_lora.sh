#!/usr/bin/env bash
set -uo pipefail
CG="/home/pope/Desktop/gpu /_gpu_probe"
H="/home/pope/Desktop/gpu /hackathons"
S="arcpro_t4-1792670357"
exec > >(tee -a "$H/05-name-that-llm/logs/colab_pro_lora.log") 2>&1
echo "=== $(date -u +%FT%TZ) Track 05 LoRA on Pro T4 session $S ==="
source "$CG/use-colab.sh" pro
colab --config "$COLAB_CONFIG" exec -s "$S" -f "$CG/colab_jobs/ntl_lora_colab.py" --timeout 16000
echo "=== exec done $(date -u +%FT%TZ) ==="
mkdir -p "$H/05-name-that-llm/outputs/colab_pro_artifacts"
for f in oof_real_fold0.npy oof_real_idx_fold0.npy oof_real_y_fold0.npy fold_status.json cv_report.json; do
  colab --config "$COLAB_CONFIG" download -s "$S" "/content/out/$f" "$H/05-name-that-llm/outputs/colab_pro_artifacts/" || true
done
for i in 1 2 3 4; do
  for pat in oof_real_fold$i.npy oof_real_idx_fold$i.npy oof_real_y_fold$i.npy; do
    colab --config "$COLAB_CONFIG" download -s "$S" "/content/out/$pat" "$H/05-name-that-llm/outputs/colab_pro_artifacts/" || true
  done
done
echo "=== PRO SESSION KEPT ALIVE: $S ==="
