"""Gold EWM traces for SFT warm-start (Appendix A) in the chess domain.

Appendix A: SFT "teaches the syntax and role structure of EWM reasoning traces"
and D_SFT "should include both call and no-call traces, so that the model learns
the world-module interface without learning to invoke W by default."

So we build, deterministically and correctly:
  * needs_visual puzzles  -> CALL traces: think -> tool_call(FEN) ->
    visual_rollout (the real, faithful forcing line from the world-module,
    MASKED in the loss) -> think -> answer(mating move).
  * factual puzzles       -> NO-CALL traces: think -> answer.

These are correct by construction (the answer comes from the verified solution),
so SFT imitates *valid* EWM behavior, including *when not to* call W.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

import chess

from .chess_data import puzzle_prompt
from .chess_world import ChessWorldModule, chess_perceive
from .trace import Trace


def gold_trace(rec: dict) -> Trace:
    prompt = puzzle_prompt(rec)
    if rec["needs_visual"]:
        fen = rec["fen"]
        board = chess.Board(fen)
        first_uci = rec["solution_uci"][0]
        first_san = board.san(chess.Move.from_uci(first_uci))
        W = ChessWorldModule(anchor_fen=fen)
        roll = W.rollout(f"visualize the forcing line from {fen}",
                         Path(mkdtemp(prefix="ewm_gold_")))
        obs = chess_perceive(roll.frame_paths, fen)
        t = Trace(problem=prompt)
        t.think("This position looks tactical; let me visualise how the forcing "
                "line unfolds before committing.")
        t.tool_call(fen)
        t.visual_rollout(obs, query=fen, module=roll.module)
        t.think(f"The rollout makes the forcing sequence explicit. The decisive "
                f"move is {first_san}.")
        t.answer(first_san)
        return t
    # no-call factual trace
    t = Trace(problem=prompt)
    t.think("This is answerable directly from the position; no visualisation is "
            "needed, so I will not call the world-module.")
    t.answer(rec["answer"])
    return t


def build_sft_corpus(recs: list[dict]) -> list[Trace]:
    return [gold_trace(r) for r in recs]
