"""Tests for the chess world-module: forced-mate search correctness, the
deterministic verifier, and text-only perception. No engine required (pure
python-chess search backend)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess

from ewm.chess_world import (ChessWorldModule, chess_perceive,
                             forced_mate_line, mating_first_moves)
from ewm.chess_data import material_balance, verify


# Scholar's mate position: White to move, Qxf7# is mate in 1.
SCHOLAR = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
# add a white queen on h5 to make Qxf7# available
SCHOLAR_Q = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"


def test_forced_mate_in_one_found():
    b = chess.Board(SCHOLAR_Q)
    line = forced_mate_line(b, 1)
    assert line is not None and len(line) == 1
    b.push(line[0])
    assert b.is_checkmate()


def test_mating_first_moves_includes_qxf7():
    b = chess.Board(SCHOLAR_Q)
    sols = {m.uci() for m in mating_first_moves(b, 1)}
    assert "h5f7" in sols  # Qxf7#


def test_no_false_mate_in_start_position():
    assert forced_mate_line(chess.Board(), 3) is None


def test_material_balance_symmetry():
    assert material_balance(chess.Board()) == 0


def test_verifier_accepts_san_and_uci():
    rec = {"needs_visual": True, "fen": SCHOLAR_Q, "solution_uci": ["h5f7"]}
    assert verify(rec, "The move is Qxf7#") == 1.0
    assert verify(rec, "h5f7") == 1.0
    assert verify(rec, "I'll play a3") == 0.0
    assert verify(rec, None) == 0.0


def test_verifier_factual():
    assert verify({"needs_visual": False, "answer": "White"}, "It is White to move") == 1.0
    assert verify({"needs_visual": False, "answer": "White"}, "Black") == 0.0
    assert verify({"needs_visual": False, "answer": "-3"}, "the balance is -3") == 1.0


def test_rollout_is_text_only_and_faithful():
    W = ChessWorldModule(anchor_fen=SCHOLAR_Q)
    roll = W.rollout(f"position {SCHOLAR_Q}", Path("runs/test_chess/call_00"))
    assert all(p.endswith(".txt") for p in roll.frame_paths)  # no images -> no VLM
    obs = chess_perceive(roll.frame_paths, "")
    assert "#" in obs or "mate" in roll.text.lower()  # the mate is exposed
