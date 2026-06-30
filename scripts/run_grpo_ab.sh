#!/usr/bin/env bash
# GRPO A/B: flat budget-penalty r_W vs info-shaped r_W (paid by measured bits).
# Identical model, corpus, split, and hyperparameters -- only the reward differs.
# Tests whether grounding r_W in delivered mutual information fixes the selectivity
# collapse the flat penalty hit at small scale. Runs after the size sweep (one GPU
# job at a time, prod box). Both arms read the LABELED corpus so the seed-0 split
# is byte-identical; flat mode simply ignores the info_gain field.
set -uo pipefail
cd /var/local/shared/einstein-world-models

while docker ps --format '{{.Names}}' | grep -q ath-ewm-mi; do sleep 20; done   # sweep done?
while docker ps --format '{{.Names}}' | grep -q ath-ewm-label; do sleep 20; done

CORPUS=data/chess_corpus_large_labeled.jsonl
COMMON="--corpus $CORPUS --G 6 --rounds 5 --warm 2 --beta 0.1 --temp 1.0 --inner 2 \
        --lr 1e-5 --ntest 30 --train-cap 60"

for mode in flat info; do
  while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -d ' ')" -ge 74 ]; do
    echo "[$(date +%H:%M:%S)] gpu hot, cooling..."; sleep 30
  done
  echo "[$(date +%H:%M:%S)] GRPO reward=$mode"
  docker run --rm --name "ath-ewm-grpo-$mode" --gpus all -v "$PWD":/w -w /w \
    -e HF_HOME=/w/.hfcache ath-ewm:latest \
    python3 scripts/chess_grpo_train.py --reward "$mode" $COMMON \
    > "runs/grpo_ab_${mode}.log" 2>&1
  echo "  -> runs/grpo_ab_${mode}.log"
done
echo "GRPO_AB DONE"
