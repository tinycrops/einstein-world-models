#!/usr/bin/env bash
# Extend the reasoner-size faithfulness curve to the Gemma-4 family on the GB10,
# served by llama.cpp (../gemma4-llama-dgx-spark). The 26b-A4B (MoE) and 31b
# (dense) have BUILT-IN chain-of-thought — the sharpest test of F1 (does a
# reasoner anchored to its own CoT still read the spliced rollout?).
#
# House rules: serve on 18080 (NEVER 8080 = prod Caddy), ath-* container names,
# off Gary's network, one GPU job at a time, temp-guarded. Runs AFTER the GRPO
# A/B releases the GPU.
set -uo pipefail
GEMMA=/var/local/shared/gemma4-llama-dgx-spark
EWM=/var/local/shared/einstein-world-models
cd "$EWM"

# (key, gguf filename). Add e2b/e4b/26b after: $GEMMA/scripts/download_model.sh <key>
# NOTE: filenames are case-sensitive on HF — E2B/E4B are UPPERCASE (lowercase 404s).
# CORRECTION: the ENTIRE Gemma-4 E-series is REASONING-NATIVE (verified by probe —
# e2b dumps CoT to reasoning_content and only emits "Answer:" in content AFTER the
# think block closes). With a small max_tokens the think block truncates, content is
# empty, and we get a silent all-zeros artifact (the first e2b/e4b run hit exactly
# this). FIX: serve with `--reasoning-budget 512` (caps thinking, forces the answer
# -> clean content, ~7s/req vs ~31s uncapped) and run the client with --max-new 1024.
# A reasoning-native model anchored to its OWN CoT is the sharpest test of F1.
declare -A GGUF=(
  [e2b]="gemma-4-E2B-it-Q8_0.gguf"
  [e4b]="gemma-4-E4B-it-Q4_K_M.gguf"
)

# MTP speculative-decoding drafter (31b only) -> ~2.9x t/s on the GB10.
MTP_DRAFT="mtp-gemma-4-31B-it.gguf"
# Don't stack two heavy GPU jobs (another gemma serve, the local-HF curve, or
# labeling). We DO run alongside the light Qwen-0.5B GRPO (~3.6g, negligible GPU).
while docker ps --format '{{.Names}}' | grep -qE 'ath-ewm-(gemma|mi|label)'; do sleep 20; done

for key in e2b e4b; do
  gguf="${GGUF[$key]}"
  [ -f "$GEMMA/models/$gguf" ] || { echo "[skip $key] $gguf not downloaded "; continue; }
  # resume-safe: a completed point is never recomputed (idempotent on relaunch).
  [ -s "$EWM/runs/mi_gemma-$key.jsonl" ] && { echo "[skip $key] already done"; continue; }
  while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -d ' ')" -ge 74 ]; do
    echo "gpu hot, cooling..."; sleep 30; done
  # MTP only for 31b (the drafter is paired to it); other keys serve plain.
  spec=()
  if [ "$key" = "31b" ] && [ -f "$GEMMA/models/$MTP_DRAFT" ]; then
    spec=(--model-draft "/models/$MTP_DRAFT" --spec-type draft-mtp --spec-draft-n-max 2)
  fi
  echo "[$(date +%H:%M:%S)] serving gemma $key on :18080 ${spec:+(+MTP)}"
  docker rm -f ath-ewm-gemma >/dev/null 2>&1 || true
  # MEMORY-BOUNDED + no auto-restart: we are the polite tenant. The cap (48g) sits
  # well above the 31B-Q4 peak (~20g) so normal runs never trip it, but bounds a
  # runaway so a prod 16g spike OOM-kills US, never gary-backend-spark-*.
  docker run -d --name ath-ewm-gemma --gpus all --restart=no \
    --memory=48g --memory-swap=48g -p 18080:8080 \
    -v "$GEMMA/models":/models:ro llama_host:latest \
    --model "/models/$gguf" "${spec[@]}" --reasoning-budget 512 \
    -ngl 999 -fa on -c 8192 --no-warmup >/dev/null
  # wait for the OpenAI endpoint to come up (31B load can take a minute)
  for _ in $(seq 1 90); do
    curl -sf --max-time 3 http://localhost:18080/v1/models >/dev/null 2>&1 && break; sleep 5; done
  # served big models are slow -> a leaner budget than the local Qwen curve.
  # --req-cooldown duty-cycles the heavy 31B serve below the 77C thermal cap on
  # the shared prod box (a continuous run pins the GPU at ~77C; pacing keeps margin).
  docker run --rm --network host --memory=8g --memory-swap=8g \
    -v "$EWM":/w -w /w ath-ewm:latest \
    python3 scripts/faithfulness_mi.py --endpoint http://localhost:18080/v1 \
      --served-model "gemma-4-$key" --n-per-kind 8 --K 4 --temp 0.7 \
      --max-new 1024 --req-cooldown 2.0 \
      --out "runs/mi_gemma-$key.jsonl" > "runs/mi_gemma-$key.log" 2>&1
  echo "  -> runs/mi_gemma-$key.log"
  docker rm -f ath-ewm-gemma >/dev/null 2>&1 || true
done
echo "GEMMA_CURVE DONE"
