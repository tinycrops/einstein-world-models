"""Verifiable chess puzzles -- the dataset the paper's §5 begs for.

The EWM data bottleneck (§5) is: text-only problems, *verifiable* answers,
balanced between those that NEED a visual-temporal rollout and those that don't,
so the model can learn *when not to* imagine. Chess supplies exactly this for
free:

  needs_visual=True   forced-mate puzzles (FEN in -> mating move out). You must
                      "see" the forcing line unfold -- the rollout's whole job.
  needs_visual=False  factual questions answerable straight from the FEN
                      (whose move, in check?, material balance). Calling the
                      world-module here only wastes the selectivity budget.

Answers are verified deterministically (python-chess), so RLVR reward is exact.
Mate puzzles are mined with Stockfish's mate search over randomly-played
positions; falls back to the pure-python forced-mate search if no engine.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import chess
import chess.engine

from .chess_world import forced_mate_line, mating_first_moves

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material_balance(board: chess.Board) -> int:
    bal = 0
    for sq, pc in board.piece_map().items():
        v = PIECE_VAL[pc.piece_type]
        bal += v if pc.color == chess.WHITE else -v
    return bal


# ---- generation -----------------------------------------------------------
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


def _mate_in_n(board: chess.Board, engine, max_plies: int) -> int | None:
    """Return the forced-mate distance (in moves) for the side to move, or None."""
    if engine is not None:
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].pov(board.turn)
            if score.is_mate() and score.mate() is not None and 0 < score.mate() <= (max_plies + 1) // 2:
                return score.mate()
        except Exception:
            pass
        return None
    line = forced_mate_line(board, max_plies)
    return (len(line) + 1) // 2 if line else None


def generate(n_mate: int = 8, n_factual: int = 8, seed: int = 7,
             max_plies: int = 3) -> list[dict]:
    rng = random.Random(seed)
    engine_path = shutil.which("stockfish")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path) if engine_path else None
    recs: list[dict] = []
    try:
        positions = _random_positions(rng, n_games=2000, max_plies=40)
        rng.shuffle(positions)

        # ---- needs_visual: forced-mate puzzles ----
        # Fast path: harvest mate-in-1 with the pure-python scan (no engine, so
        # no per-position Stockfish call -> generation stays quick). Optionally
        # top up with engine-found mate-in-2 (capped) when fuller depth is asked.
        seen_fen: set[str] = set()
        for fen in positions:
            if len(seen_fen) >= n_mate:
                break
            if fen in seen_fen:
                continue
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            sol = mating_first_moves(board, 1)  # immediate mate, fast
            dist = 1
            if not sol and engine is not None and max_plies >= 3 and board.is_check():
                # cheap pre-filter (already in check) before a deeper engine probe
                d = _mate_in_n(board, engine, max_plies)
                if d:
                    sol = mating_first_moves(board, d * 2)
                    dist = d
            if not sol:
                continue
            seen_fen.add(fen)
            recs.append({
                "id": f"mate{dist}-{len(recs)}",
                "kind": f"mate_in_{dist}",
                "needs_visual": True,
                "fen": fen,
                "question": f"White or Black to move as given by the FEN. Find a move that "
                            f"forces checkmate (mate in {dist}). Answer with the move in SAN.",
                "solution_uci": sorted(m.uci() for m in sol),
            })

        # ---- non-visual: factual questions straight from the FEN ----
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
    finally:
        if engine is not None:
            engine.quit()
    return recs


def _q_turn(b: chess.Board):
    return ("Whose move is it? Answer 'White' or 'Black'.",
            "White" if b.turn == chess.WHITE else "Black", "turn")


def _q_incheck(b: chess.Board):
    return ("Is the side to move in check? Answer 'Yes' or 'No'.",
            "Yes" if b.is_check() else "No", "incheck")


def _q_material(b: chess.Board):
    return ("What is the material balance in points (positive = White ahead, "
            "negative = Black ahead, pawn=1 N=B=3 R=5 Q=9)? Answer with the integer.",
            str(material_balance(b)), "material")


# ---- verification ---------------------------------------------------------
def _parse_move(board: chess.Board, text: str) -> chess.Move | None:
    """Best-effort parse of a model's answer into a legal move (SAN or UCI)."""
    import re
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9=+#-]*|O-O(?:-O)?", text.replace("0", "O")):
        for cand in (tok, tok.rstrip("+#"), tok.replace("0", "O")):
            try:
                return board.parse_san(cand)
            except ValueError:
                pass
    for tok in re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", text.lower()):
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                return mv
        except ValueError:
            pass
    return None


def verify(rec: dict, answer_text: str | None) -> float:
    """Deterministic RLVR reward r(yhat, y*) in {0,1}."""
    if not answer_text:
        return 0.0
    if rec["needs_visual"]:  # mate puzzle: any move in the mating set
        board = chess.Board(rec["fen"])
        mv = _parse_move(board, answer_text)
        return 1.0 if (mv and mv.uci() in set(rec["solution_uci"])) else 0.0
    # factual: normalized substring/exact match
    gold = rec["answer"].strip().lower()
    got = answer_text.strip().lower()
    if gold in ("white", "black", "yes", "no"):
        return 1.0 if gold in got.split() or got.startswith(gold) else 0.0
    return 1.0 if gold == got.strip(" .") or f" {gold} " in f" {got} " else 0.0


def puzzle_prompt(rec: dict) -> str:
    """The text-only problem x shown to the reasoner (same at train + inference)."""
    board = chess.Board(rec["fen"])
    return f"{rec['question']}\nFEN: {rec['fen']}\n{board}\n"


def write_dataset(path: str | Path, **kw) -> list[dict]:
    recs = generate(**kw)
    Path(path).write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return recs


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "chess_puzzles.jsonl"
    recs = write_dataset(out)
    nv = sum(r["needs_visual"] for r in recs)
    print(f"wrote {len(recs)} puzzles ({nv} needs_visual, {len(recs)-nv} factual) -> {out}")
    for r in recs[:4]:
        print(" ", r["id"], r["kind"], "| sol", r.get("solution_uci", r.get("answer")))
