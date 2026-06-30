"""Aggregate the reasoner-size sweep into the capacity-vs-delivered curve.

Reads runs/mi_*.jsonl produced by run_size_sweep.sh / run_gemma_curve.sh and
prints, per model, the rollout's delivered info gain and channel utilization on
the tactical (needs_visual) puzzles -- the question being whether a stronger
reasoner extracts a larger fraction of the move-line channel's bits.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = {
    "Qwen2.5-0.5B": 0.5,
    "Qwen2.5-1.5B": 1.5,
    "gemma-e2b": 2.0,
    "Qwen2.5-3B": 3.0,
    "gemma-e4b": 4.0,
    "gemma-26b": 26.0,
    "gemma-31b": 31.0,
}


def tag_of(path: str | Path) -> str:
    return Path(path).stem.removeprefix("mi_")


def sort_key(path: str | Path) -> tuple[float, str]:
    tag = tag_of(path)
    return (MODEL_ORDER.get(tag, 1_000.0), tag)


def main() -> int:
    # Only completed model-sweep files, not pilot/old-schema runs.
    files = sorted((f for f in glob.glob(str(ROOT / "runs" / "mi_*.jsonl"))
                    if tag_of(f) in MODEL_ORDER), key=sort_key)
    if not files:
        print("no runs/mi_*.jsonl found -- run scripts/run_size_sweep.sh first")
        return 1
    print(f"{'reasoner':<14}{'tactical_IG':>12}{'capacity':>10}{'util':>8}"
          f"{'acc0':>7}{'acc1':>7}{'lift':>7}")
    rows = []
    for f in files:
        recs = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        nv = [r for r in recs if r["needs_visual"]]
        if not nv:
            continue
        tag = tag_of(f)
        ig = mean(r["info_gain"] for r in nv)
        cap = mean(r["capacity_bits"] for r in nv)
        util = mean(r["utilization"] for r in nv)
        a0 = mean(r["acc_textonly"] for r in nv)
        a1 = mean(r["acc_rollout"] for r in nv)
        rows.append((tag, ig, cap, util, a0, a1))
        print(f"{tag:<14}{ig:>+12.2f}{cap:>10.2f}{util:>8.2f}"
              f"{a0:>7.2f}{a1:>7.2f}{a1-a0:>+7.2f}")
    if len(rows) >= 2:
        best = max(rows, key=lambda r: r[3])
        worst = min(rows, key=lambda r: r[3])
        d_util = rows[-1][3] - rows[0][3]
        print(f"\nbest util: {best[0]} {best[3]:.2f}; "
              f"worst util: {worst[0]} {worst[3]:.2f}")
        print(f"endpoint util change {rows[0][0]} -> {rows[-1][0]}: {d_util:+.2f} "
              "(curve is non-monotonic; delivered faithfulness is model/format-specific)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
