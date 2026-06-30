"""Attach a measured per-puzzle information-gain label to the corpus.

The bridge from measurement (faithfulness_mi) to training (info-shaped GRPO):
for each puzzle, sample the held-fixed reasoner in two conditions (text-only vs
faithful-rollout), and write the delivered info gain IG(x) onto the record. The
shaped reward `ewm.reward.info_shaped_world_reward` then pays a world-call exactly
IG(x) bits -- so the policy is rewarded for calling W when (and only when) the
rollout is worth its bits *for this reasoner on this puzzle*.

Labelling with the SAME model that will be RL-trained is deliberate: IG is the
information THIS reasoner gains, which is exactly the call-worthiness the policy
should learn. (A tactical puzzle the reasoner can't use even with the rollout
gets IG~0 -- honest: calling it wouldn't help, so don't reward the call.)

Run:
  docker run --rm --gpus all -v $PWD:/w -w /w -e HF_HOME=/w/.hfcache ath-ewm:latest \
    python3 scripts/label_infogain.py --K 6 --temp 0.8 \
      --in data/chess_corpus_large.jsonl --out data/chess_corpus_large_labeled.jsonl
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

from ewm.chess_data import puzzle_prompt
from ewm.infometric import task_information_gain
from scripts.faithfulness_mi import rollout_observation, sample_answers, verify_move


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/chess_corpus_large.jsonl")
    ap.add_argument("--out", default="data/chess_corpus_large_labeled.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)

    recs = [json.loads(l) for l in (ROOT / args.inp).read_text().splitlines() if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"labelling {len(recs)} puzzles with {args.model} (K={args.K})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(dev).eval()

    tmp = ROOT / "runs" / "label_rollouts"
    for i, rec in enumerate(recs):
        problem = puzzle_prompt(rec)
        obs, _ = rollout_observation(rec, tmp / rec["id"])
        a_text = sample_answers(model, tok, dev, problem, None, args.K, args.temp)
        a_roll = sample_answers(model, tok, dev, problem, obs, args.K, args.temp)
        n_t = round(mean(verify_move(rec, u) for u in a_text) * args.K)
        n_r = round(mean(verify_move(rec, u) for u in a_roll) * args.K)
        rec["acc_textonly"] = n_t / args.K
        rec["acc_rollout"] = n_r / args.K
        rec["info_gain"] = task_information_gain(n_t, n_r, args.K)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)} labelled")

    (ROOT / args.out).write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    nv = [r for r in recs if r["needs_visual"]]
    fac = [r for r in recs if not r["needs_visual"]]
    print(f"\nwrote {ROOT / args.out}")
    print(f"  needs_visual mean IG = {mean(r['info_gain'] for r in nv):+.2f} bits "
          f"(n={len(nv)})")
    print(f"  factual      mean IG = {mean(r['info_gain'] for r in fac):+.2f} bits "
          f"(n={len(fac)})")
    pos = sum(1 for r in nv if r["info_gain"] > 0.3)
    print(f"  tactical puzzles with IG>0.3 (call-worthy): {pos}/{len(nv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
