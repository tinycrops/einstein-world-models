"""Tests for the chess world-module: forced-mate search correctness, the
deterministic verifier, and text-only perception. No engine required (pure
python-chess search backend)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess

from ewm.chess_world import (ChessWorldModule, chess_perceive, critical_defender,
                             forced_mate_line, mating_first_moves)
from ewm.chess_data import material_balance, verify

# Back-rank: White Re1/Kg1; Black Kg8 boxed by f7/g7/h7; a Black rook on d8 is the
# ONLY defender of the back rank -- remove it and Re8 is mate in 1.
BACKRANK = "3r2k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"


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


# ---- counterfactual / intervention rollout (borrowed from VOID) -------------
def test_critical_defender_finds_backrank_guard():
    # no immediate mate while the d8 rook defends...
    assert not mating_first_moves(chess.Board(BACKRANK), 1)
    # ...and the intervention square is exactly that defender.
    assert critical_defender(BACKRANK, ["e1e8"]) == "d8"


def test_counterfactual_removal_enables_the_mate():
    import chess as _c
    b = _c.Board(BACKRANK)
    b.remove_piece_at(_c.parse_square("d8"))
    assert any(m.uci() == "e1e8" for m in mating_first_moves(b, 1))  # Re8# now mate


def test_counterfactual_rollout_text_is_inspectable():
    W = ChessWorldModule(anchor_fen=BACKRANK)
    roll = W.counterfactual_rollout(Path("runs/test_chess/cf_00"), "d8")
    assert "counterfactual" in roll.text.lower() and "d8" in roll.text
    assert all(p.endswith(".txt") for p in roll.frame_paths)


def test_counterfactual_vacuous_and_illegal_interventions():
    W = ChessWorldModule(anchor_fen=BACKRANK)
    assert "vacuous" in W.counterfactual_rollout(Path("runs/test_chess/cf_e"), "a3").text
    assert "not a legal" in W.counterfactual_rollout(Path("runs/test_chess/cf_k"), "g8").text
