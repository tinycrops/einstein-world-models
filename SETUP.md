# EWM setup on `spark` (NVIDIA GB10 / DGX Spark, aarch64)

> **House rules (from `~/CLAUDE.md` on spark — read it first):**
> **Docker for everything. NO host-level installs** (no host CUDA/torch/pip/apt).
> **Never bind host port 8080** (prod Caddy/Gary) — use high ports like `18080`.
> **Never touch `gary-backend-spark-*` containers or their network.** Run `nvidia-smi`
> before any GPU job and keep GPU temp <75 °C (manage batch size). Coordinate with Kev
> on OOMs. GB10 = aarch64, SM_121, CUDA 13, ~122 GB **unified** memory → prefer **bf16**,
> no QLoRA/bitsandbytes needed.

The code was rsynced to `/var/local/shared/einstein-world-models/`; **dependencies were
not** shipped (x86_64 `.venv-sd` can't run on ARM, and host installs are banned anyway).
Everything below runs in containers.

## The base image already exists
`spark-ngpt-llama-tribe-lab:latest` (22.7 GB) is the GB10-native training image:
torch 2.6 (nv25.01), **transformers 5.12**, peft 0.19, accelerate 1.14, datasets 4.0,
CUDA cap (12,1). Use it as-is for the torch/SFT/GRPO track — no host setup.

### ⚠️ Two gaps for EWM in the base image
1. **`transformers` mismatch.** Our code was written against the `transformers==4.57.3`
   pin (the SD/diffusers stack breaks on 5.x). The base image has **5.12**. The *chess /
   SFT / GRPO* track uses transformers only for the Qwen policy (`ewm/hf_policy.py`), no
   diffusers — it *may* run fine on 5.12. **Test first; only pin down if it breaks.**
2. **`stockfish` missing** (the §3.1 simulator W) and **`python-chess`** not in the image.

## Option A — quick check, no rebuild (chess track)
Confirm the base image runs our code on transformers 5.12, and add the two missing pieces
ephemerally inside the container:
```bash
cd /var/local/shared/einstein-world-models
nvidia-smi   # check temp/free memory before anything GPU
docker run --rm --gpus all -v $PWD:/w -w /w \
  spark-ngpt-llama-tribe-lab:latest \
  bash -lc "apt-get update -q && apt-get install -y -q stockfish && \
            pip install -q chess && \
            python3 -m pytest -q tests/"
```
20 green tests ⇒ the code + python-chess + torch path work under 5.12. If transformers
5.12 breaks something, fall back to Option B.

## Option B — derived image (the clean, repeatable path)
Bake stockfish + chess (and, only if needed, the 4.57.3 pin) into an `ath-ewm` image.
Create `/var/local/shared/einstein-world-models/Dockerfile.ewm`:
```dockerfile
FROM spark-ngpt-llama-tribe-lab:latest
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir chess
# Uncomment ONLY if transformers 5.12 breaks the policy code:
# RUN pip install --no-cache-dir "transformers==4.57.3"
```
```bash
cd /var/local/shared/einstein-world-models
docker build -f Dockerfile.ewm -t ath-ewm:latest .
docker run --rm --gpus all -v $PWD:/w -w /w ath-ewm:latest python3 -m pytest -q tests/
docker run --rm --gpus all -v $PWD:/w -w /w ath-ewm:latest python3 scripts/smoke.py
```
Point the code at the in-container stockfish (`/usr/games/stockfish`, the default in
`ewm/chess_world.py`).

## Real training runs
```bash
docker run --rm --gpus all -v $PWD:/w -w /w ath-ewm:latest \
  python3 scripts/chess_sft.py <args>      # SFT — the workhorse
# GRPO likewise via scripts/chess_grpo_train.py
```
bf16 on 122 GB unified memory — no 4-bit. Watch `nvidia-smi` temp; drop batch size if it
climbs toward 75 °C. **Do not** kill Gary services to free GPU memory.

## Optional — SD world-module (full visual rollout only)
The chess/SFT/GRPO track does NOT need this. SD needs `diffusers==0.36.0` **+**
`transformers==4.57.3` together (5.x breaks diffusers), so build a *separate* image off
`python:3.12-slim` or the base image with that exact pin, plus copy the ~2 GB checkpoint:
`dreamshaper_8.safetensors` (not in repo). Keep it isolated from the chess image to avoid
the transformers tug-of-war.

## If anything is served on a port
Use a high host port (e.g. `-p 18080:...`). **Never `-p 8080`.** Check `docker ps`/`ss`
for conflicts first. Name containers `ath-ewm-*`, keep them off `gary-backend-spark_default`.

## What was deliberately NOT shipped
- `.venv-sd/` (x86_64 — useless on ARM, and host installs are banned).
- dreamshaper checkpoint (~2 GB) + HF model caches — pull on demand into a mounted volume.
