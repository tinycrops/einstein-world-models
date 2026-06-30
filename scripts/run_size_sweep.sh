#!/usr/bin/env bash
# Reasoner-size faithfulness curve: run the two-condition MI experiment at a fixed
# budget across model sizes, to see whether a stronger reasoner extracts more of
# the rollout channel (util = IG / capacity). One GPU job at a time (prod box).
set -euo pipefail
cd /var/local/shared/einstein-world-models

# wait for any in-flight labeling job to release the GPU
while docker ps --format '{{.Names}}' | grep -q ath-ewm-label; do sleep 20; done

MODELS=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct")
for m in "${MODELS[@]}"; do
  tag=$(echo "$m" | sed 's#.*/##; s/-Instruct//')
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
  echo "[$(date +%H:%M:%S)] $tag  (gpu ${t}C)"
  # temp guard: if hot, cool off before adding load
  while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -d ' ')" -ge 74 ]; do
    echo "  gpu hot, cooling..."; sleep 30
  done
  docker run --rm --name "ath-ewm-mi-$tag" --gpus all -v "$PWD":/w -w /w \
    -e HF_HOME=/w/.hfcache ath-ewm:latest \
    python3 scripts/faithfulness_mi.py --model "$m" \
      --n-per-kind 12 --K 6 --temp 0.8 --out "runs/mi_${tag}.jsonl" \
    > "runs/mi_${tag}.log" 2>&1
  echo "  -> runs/mi_${tag}.log"
done
echo "SWEEP DONE"
