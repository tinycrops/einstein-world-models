# EWM experiments — what's running, memory posture, relaunch strategy

For anyone sharing this DGX Spark (esp. Kev) — our jobs and how they behave under
memory pressure. **Our containers are all named `ath-ewm-*`.** They are safe to let
the OOM killer sacrifice; nothing here is production.

## Memory posture (why prod is safe)

- Box: 122 GB unified. Typical use ~48 GB, **~73 GB available**.
- Our footprint is small: SFT/GRPO (Qwen-0.5B) container ≈ **3.6 GB**; the heaviest
  job is the Gemma-4 31B serve, peak ≈ **20 GB**.
- Kev's expected **16 GB nightly spikes** + our peak still leave comfortable margin.
  No action needed for the hardware to absorb them.
- **We are the bounded tenant.** All *new* `ath-ewm-*` launches carry explicit
  `--memory` caps (`run_gemma_curve.sh`: serve 48 g, client 8 g) and `--restart=no`.
  The caps sit well above our real peak, so normal runs never trip them, but they
  guarantee a runaway/contention OOM kills **us**, never `gary-backend-spark-*`.
  (The one currently-running container, `ath-ewm-grpo-*`, predates this note and is
  uncapped, but it is ~3.6 GB and finishing within minutes — not worth a restart.)

## Relaunch strategy (if a prod spike kills one of our jobs)

Everything is **idempotent and resume-safe** — a kill costs at most the in-flight
data point, never earlier results:

- **Inputs are cached files**, not regenerated: `data/chess_corpus_large.jsonl`
  (300 puzzles) and `data/chess_corpus_large_labeled.jsonl` (with measured IG).
- **Outputs are per-unit files** (`runs/mi_<size>.jsonl`, `runs/grpo_ab_<mode>.log`,
  `runs/mi_gemma-<key>.jsonl`). A completed unit is never recomputed —
  `run_gemma_curve.sh` skips any `key` whose `runs/mi_gemma-<key>.jsonl` already
  exists. So **just re-run the orchestrator**; it resumes where it died.
- **No auto-restart** (`--restart=no`) on purpose: a one-shot experiment that
  auto-restarted would double-count. Relaunch is deliberate and orchestrated.

### Relaunch commands
```bash
cd /var/local/shared/einstein-world-models
# size curve (Gemma, served on :18080 — never 8080): resumes, skips done points
nohup bash scripts/run_gemma_curve.sh   > runs/gemma_curve.log 2>&1 &
# GRPO flat-vs-info A/B (small, fast): re-runs both arms from the labeled corpus
nohup bash scripts/run_grpo_ab.sh       > runs/grpo_ab.log     2>&1 &
```

All GPU orchestrators self-gate: one job at a time, and they wait while GPU temp
≥ 74 °C (house cap 75 °C). They also wait for any other `ath-ewm-*` GPU job to
exit before starting, so a relaunch can't stack two GPU jobs.
