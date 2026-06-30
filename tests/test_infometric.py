"""Unit tests for the EWM information-theoretic instrument (no GPU)."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from ewm.infometric import (delivered_mi, line_channel_bits,  # noqa: E402
                            miller_madow_entropy, plugin_entropy,
                            position_branching_bits, task_information_gain)


def test_startpos_branching_is_log2_20():
    # 20 legal first moves from the start position.
    assert abs(position_branching_bits(chess.STARTING_FEN) - math.log2(20)) < 1e-9


def test_channel_bits_accumulate_along_line():
    # First two ply of the main line; capacity = log2(20) + log2(20).
    fen = chess.STARTING_FEN
    bits = line_channel_bits(fen, ["e2e4", "e7e5"])
    assert abs(bits - (math.log2(20) + math.log2(20))) < 1e-9


def test_channel_bits_stop_on_illegal():
    # An illegal continuation halts accumulation (only the first ply counts).
    bits = line_channel_bits(chess.STARTING_FEN, ["e2e4", "zzzz"])
    assert abs(bits - math.log2(20)) < 1e-9


def test_plugin_entropy_uniform_and_degenerate():
    assert plugin_entropy(["a", "b", "c", "d"]) == 2.0   # 4 equiprobable -> 2 bits
    assert plugin_entropy(["a", "a", "a"]) == 0.0        # certain -> 0 bits


def test_miller_madow_geq_plugin():
    s = ["a", "b", "c", "a", "b", "d"]
    assert miller_madow_entropy(s) >= plugin_entropy(s)


def test_delivered_mi_positive_when_rollout_collapses_uncertainty():
    # Text-only: spread over 4 answers (2 bits). With rollout: all agree (0 bits).
    text = ["a", "b", "c", "d"]
    roll = ["a", "a", "a", "a"]
    assert delivered_mi(text, roll, corrected=False) == 2.0


def test_delivered_mi_zero_when_rollout_inert():
    # Rollout doesn't change the answer distribution -> 0 bits delivered.
    same = ["a", "b", "a", "b"]
    assert abs(delivered_mi(same, list(same), corrected=False)) < 1e-9


def test_info_gain_positive_when_rollout_raises_correctness():
    # text-only never right, with-rollout always right -> large positive gain.
    ig = task_information_gain(n_correct_textonly=0, n_correct_rollout=8, K=8)
    assert ig > 0
    # symmetric: a rollout that breaks a correct answer yields negative gain.
    assert task_information_gain(8, 0, 8) < 0


def test_info_gain_zero_when_no_accuracy_change():
    assert abs(task_information_gain(4, 4, 8)) < 1e-9
