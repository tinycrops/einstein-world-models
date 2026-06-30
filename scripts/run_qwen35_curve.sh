#!/usr/bin/env bash
# Reasoner-size faithfulness curve over the Qwen3.5 SMALL DENSE ladder
# (0.8B/2B/4B/9B), served by llama.cpp on the GB10. Replaces the Qwen2.5
# 0.5B/1.5B/3B local-HF curve with a wider, newer ladder. Qwen3.5 is a
# THINKING model (<think>...</think> by default) -> reasoning-native handling in
# faithfulness_mi.py applies (cap the think block, parse the move from content).
#
# House rules: serve on 18080 (NEVER 8080 = prod Caddy), ath-* container names,
# off Gary's network, memory-bounded + --restart=no (we are the sacrificial
# tenant), one GPU job at a time, temp-guarded under the 75C cap. Resume-safe:
# a completed rung (runs/mi_qwen35-<key>.jsonl) is never recomputed.
set -uo pipefail
QWEN=/var/local/shared/qwen3.5-dgx-spark
EWM=/var/local/shared/einstein-world-models
cd "$EWM"

# rung order: smallest first so we get curve points fast.
KEYS=(0.8B 2B 4B 9B)

# don't stack two GPU jobs: wait for any other ath-ewm GPU job (the Qwen2.5
# local-HF runs, gemma serve, labeling) to release the card first.
while docker ps --format '{{.Names}}' | grep -qE 'ath-ewm-(h1|gemma|mi|label)'; do
  echo "[wait] another ath-ewm GPU job running; sleeping 20s"; sleep 20; done

for key in "${KEYS[@]}"; do
  gguf=$(ls "$QWEN/models/qwen3.5-$key"/*.gguf 2>/dev/null | head -1)
  [ -n "$gguf" ] || { echo "[skip $key] no gguf in models/qwen3.5-$key"; continue; }
  [ -s "$EWM/runs/mi_qwen35-$key.jsonl" ] && { echo "[skip $key] already done"; continue; }

  while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -d ' ')" -ge 74 ]; do
    echo "[$key] gpu hot, cooling..."; sleep 30; done

  echo "[$(date +%H:%M:%S)] serving qwen3.5 $key on :18080  ($(basename "$gguf"))"
  docker rm -f ath-ewm-qwen35 >/dev/null 2>&1 || true
  # cap 16g sits well above a 9B-Q4 serve peak (~10g) so normal runs never trip
  # it, but bounds a runaway so a prod spike OOM-kills US, never gary-backend-*.
  docker run -d --name ath-ewm-qwen35 --gpus all --restart=no \
    --memory=16g --memory-swap=16g -p 18080:8080 \
    -v "$QWEN/models":/models:ro llama_host:latest \
    --model "/models/qwen3.5-$key/$(basename "$gguf")" \
    --reasoning-budget 512 -ngl 999 -fa on -c 8192 --no-warmup >/dev/null

  # wait for the OpenAI endpoint (small models load fast, allow generous margin)
  up=0
  for _ in $(seq 1 60); do
    curl -sf --max-time 3 http://localhost:18080/v1/models >/dev/null 2>&1 && { up=1; break; }
    sleep 5; done
  [ "$up" = 1 ] || { echo "[skip $key] endpoint never came up"; docker logs --tail 20 ath-ewm-qwen35; docker rm -f ath-ewm-qwen35 >/dev/null 2>&1; continue; }

  # lean budget so the full ladder finishes fast (the point is the size trend).
  docker run --rm --network host --memory=8g --memory-swap=8g \
    -v "$EWM":/w -w /w ath-ewm:latest \
    python3 scripts/faithfulness_mi.py --endpoint http://localhost:18080/v1 \
      --served-model "qwen3.5-$key" --n-per-kind 8 --K 4 --temp 0.7 \
      --max-new 1024 --req-cooldown 1.0 \
      --out "runs/mi_qwen35-$key.jsonl" > "runs/mi_qwen35-$key.log" 2>&1
  echo "  -> runs/mi_qwen35-$key.log"
  docker rm -f ath-ewm-qwen35 >/dev/null 2>&1 || true
done
echo "QWEN35_CURVE DONE"
