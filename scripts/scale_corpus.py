"""Scale the verifiable EWM chess corpus ~10x for GRPO.

The GRPO degradation documented in CLAUDE.md was diagnosed as a *scale* failure:
a 30-puzzle corpus gives group-relative advantages too noisy to improve on the
SFT mode. This produces a larger, balanced, fully-verifiable corpus.

Two kinds of needs_visual puzzles (both verified deterministically by the same
mating-first-move set, so RLVR reward stays exact):
  * mate_in_1 -- a strong reasoner can often spot these without "seeing" the line.
  * mate_in_2 -- genuinely needs the rollout: you must visualise the opponent's
    forced replies before the mate lands. This sharpens the *needs-the-rollout*
    discriminator the paper's selectivity objective (§2.4.1) depends on.

Balanced against factual no-call puzzles (turn / in-check / material) so D learns
*when not to* call W (Appendix A guardrail).

Run in the EWM container:
  docker run --rm -v $PWD:/w -w /w ath-ewm:latest \
    python3 scripts/scale_corpus.py --n-mate1 110 --n-mate2 40 --n-factual 150 \
      --out data/chess_corpus_large.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.chess_data import (_q_incheck, _q_material, _q_turn,  # noqa: E402
                            material_balance)
from ewm.chess_world import mating_first_moves  # noqa: E402


def _random_positions(rng: random.Random, n_games: int, max_plies: int) -> list[str]:
    seen = []
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(rng.randint(6, max_plies)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
            seen.append(b.fen())
    return seen


def _mate_record(fen: str, dist: int, sol) -> dict:
    return {
        "kind": f"mate_in_{dist}",
        "needs_visual": True,
        "fen": fen,
        "question": (f"White or Black to move as given by the FEN. Find a move that "
                     f"forces checkmate (mate in {dist}). Answer with the move in SAN."),
        "solution_uci": sorted(m.uci() for m in sol),
    }


def generate_large(n_mate1: int, n_mate2: int, n_factual: int,
                   seed: int, n_games: int) -> list[dict]:
    rng = random.Random(seed)
    positions = _random_positions(rng, n_games=n_games, max_plies=50)
    rng.shuffle(positions)

    recs: list[dict] = []
    seen_fen: set[str] = set()
    got1 = got2 = 0
    for fen in positions:
        if got1 >= n_mate1 and got2 >= n_mate2:
            break
        if fen in seen_fen:
            continue
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        m1 = mating_first_moves(board, 1)
        if m1:
            if got1 < n_mate1:
                seen_fen.add(fen)
                recs.append(_mate_record(fen, 1, m1))
                got1 += 1
            continue
        if got2 < n_mate2:
            m2 = mating_first_moves(board, 3)  # forced mate in <=2 (no mate-in-1)
            if m2:
                seen_fen.add(fen)
                recs.append(_mate_record(fen, 2, m2))
                got2 += 1

    # ---- factual no-call puzzles (balanced across the three question types) ----
    fact_pool = [p for p in positions if p not in seen_fen]
    builders = [_q_turn, _q_incheck, _q_material]
    for i in range(n_factual):
        fen = fact_pool[i % len(fact_pool)]
        q, ans, kind = builders[i % len(builders)](chess.Board(fen))
        recs.append({
            "id": f"fact-{kind}-{i}",
            "kind": f"factual_{kind}",
            "needs_visual": False,
            "fen": fen,
            "question": q,
            "answer": ans,
        })

    # stable ids for the mate puzzles
    mi = 0
    for r in recs:
        if r["needs_visual"]:
            r["id"] = f"{r['kind']}-{mi}"
            mi += 1
    rng.shuffle(recs)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mate1", type=int, default=110)
    ap.add_argument("--n-mate2", type=int, default=40)
    ap.add_argument("--n-factual", type=int, default=150)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--n-games", type=int, default=12000)
    ap.add_argument("--out", default="data/chess_corpus_large.jsonl")
    args = ap.parse_args()

    t = time.time()
    recs = generate_large(args.n_mate1, args.n_mate2, args.n_factual,
                          args.seed, args.n_games)
    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    kinds = Counter(r["kind"] for r in recs)
    nv = sum(r["needs_visual"] for r in recs)
    print(f"wrote {len(recs)} puzzles in {time.time()-t:.1f}s -> {out}")
    print(f"  needs_visual {nv} / factual {len(recs)-nv}")
    print(f"  kinds: {dict(kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
