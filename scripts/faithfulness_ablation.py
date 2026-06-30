"""Control experiment: is it the rollout's INFORMATION, or just its presence?

`faithfulness_mi.py` shows a faithful rollout delivers ~2 bits about the true
answer on tactical puzzles and ~0 on factual. But a sceptic's question remains
(the §3.2 faithfulness worry): does the rollout help because it *exposes the
forcing line*, or would any rollout-shaped block of text nudge the weak reasoner?

We hold the reasoner and the puzzle fixed and vary only the rollout's CONTENT,
across three conditions per tactical puzzle:

  none       : text-only (baseline P(Y*|X)).
  faithful   : the real Stockfish forcing line for THIS position.
  mismatched : a real forcing line from a DIFFERENT random mate position --
               identical format and capacity, wrong information.

EWM faithfulness predicts: IG(faithful) >> IG(mismatched) ~ 0. If instead
mismatched also lifts accuracy, the effect is format/placebo, not information.

Run:
  docker run --rm --gpus all -v $PWD:/w -w /w -e HF_HOME=/w/.hfcache ath-ewm:latest \
    python3 scripts/faithfulness_ablation.py --n 30 --K 8 --temp 0.8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.chess_data import puzzle_prompt, verify
from ewm.infometric import task_information_gain
from scripts.faithfulness_mi import rollout_observation, sample_answers, verify_move


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/chess_corpus_large.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=30, help="tactical puzzles to test")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", default="runs/faithfulness_ablation.jsonl")
    args = ap.parse_args()
    random.seed(1); torch.manual_seed(1)

    recs = [json.loads(l) for l in (ROOT / args.corpus).read_text().splitlines() if l.strip()]
    tactical = [r for r in recs if r["needs_visual"]]
    random.shuffle(tactical)
    sel = tactical[:args.n]
    print(f"corpus {len(recs)}: testing {len(sel)} tactical puzzles, 3 rollout conditions")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(dev).eval()

    tmp = ROOT / "runs" / "ablation_rollouts"
    # pre-compute each puzzle's faithful rollout; the mismatched rollout for puzzle
    # i is the faithful rollout of puzzle (i+1) -- a real line from a wrong board.
    obs_cache = [rollout_observation(r, tmp / r["id"])[0] for r in sel]

    recs_out = []
    for i, rec in enumerate(sel):
        problem = puzzle_prompt(rec)
        mismatch_obs = obs_cache[(i + 1) % len(sel)]
        a_none = sample_answers(model, tok, dev, problem, None, args.K, args.temp)
        a_faith = sample_answers(model, tok, dev, problem, obs_cache[i], args.K, args.temp)
        a_mis = sample_answers(model, tok, dev, problem, mismatch_obs, args.K, args.temp)
        n_none = round(mean(verify_move(rec, u) for u in a_none) * args.K)
        n_faith = round(mean(verify_move(rec, u) for u in a_faith) * args.K)
        n_mis = round(mean(verify_move(rec, u) for u in a_mis) * args.K)
        recs_out.append({
            "id": rec["id"], "kind": rec["kind"],
            "acc_none": n_none / args.K, "acc_faithful": n_faith / args.K,
            "acc_mismatched": n_mis / args.K,
            "ig_faithful": task_information_gain(n_none, n_faith, args.K),
            "ig_mismatched": task_information_gain(n_none, n_mis, args.K),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sel)} done")

    (ROOT / args.out).write_text("\n".join(json.dumps(r) for r in recs_out) + "\n")

    print("\n=== Faithful vs mismatched rollout (tactical puzzles) ===")
    print(f"{'condition':<14}{'acc':>7}{'IG_bits':>9}")
    print(f"{'none':<14}{mean(r['acc_none'] for r in recs_out):>7.2f}{0.0:>9.2f}")
    print(f"{'faithful':<14}{mean(r['acc_faithful'] for r in recs_out):>7.2f}"
          f"{mean(r['ig_faithful'] for r in recs_out):>+9.2f}")
    print(f"{'mismatched':<14}{mean(r['acc_mismatched'] for r in recs_out):>7.2f}"
          f"{mean(r['ig_mismatched'] for r in recs_out):>+9.2f}")
    gap = mean(r['ig_faithful'] for r in recs_out) - mean(r['ig_mismatched'] for r in recs_out)
    print(f"\n  faithfulness premium (faithful - mismatched IG): {gap:+.2f} bits")
    print(f"  -> {'INFORMATION, not presence' if gap > 0.3 else 'inconclusive/placebo'}")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
