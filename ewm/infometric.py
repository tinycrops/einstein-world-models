"""Information-theoretic instrument for grounding the EWM hypothesis.

The Einstein-World-Model claim is that splicing a visual-temporal *rollout* into
the reasoning trace lets the model reason in ways text alone does not support --
**selectively**, where visualisation helps (0 < M << N-1). The paper leaves
"faithfulness" (§3.2: a rollout "exposes information that the reasoner actually
uses and that the final answer depends on") unmeasured. This module turns it into
a number.

Two quantities, both grounded in the chess substrate:

1. CHANNEL CAPACITY of a rollout (the `chessencryption` primitive).
   A rollout is serialised as a move line. Each ply chooses one of
   `|legal_moves|` continuations, so the line carries
       C = sum_plies log2 |legal_moves(ply)|   bits.
   This is the literal "mutual information when you encode intelligence into the
   moves of a chess game": the bandwidth a thought-experiment rollout can carry.

2. DELIVERED (decision-relevant) MUTUAL INFORMATION.
   For answer Y, text problem X, rollout R, estimate from a reasoner's answer
   distribution
       dI = H(Y | X) - H(Y | X, R)   bits,
   a lower bound on I(Y; R | X): the bits the rollout actually removes from the
   answer's uncertainty. EWM predicts dI > 0 exactly where the rollout is needed
   (tactical) and dI ~ 0 where text suffices (factual) -- selectivity, in bits.

Nothing here needs a GPU; the model-based caller supplies sampled answers.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import chess


# --------------------------------------------------------------------------- #
# 1. channel capacity  (chessencryption bandwidth of a rollout)
# --------------------------------------------------------------------------- #
def line_channel_bits(fen: str, line_uci: Sequence[str]) -> float:
    """Bits the move-line rollout carries: sum_plies log2 |legal_moves|.

    This is exactly the encoder's per-ply alphabet in `chessencryption`
    (legal_moves is the symbol set), summed along the realised line."""
    board = chess.Board(fen)
    bits = 0.0
    for uci in line_uci:
        n = board.legal_moves.count()
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)            # only a legally-played ply carries bits
        except (ValueError, AssertionError):
            break
        if n > 1:
            bits += math.log2(n)
    return bits


def position_branching_bits(fen: str) -> float:
    """log2 |legal_moves| at a single position -- the first-ply channel width."""
    n = chess.Board(fen).legal_moves.count()
    return math.log2(n) if n > 1 else 0.0


# --------------------------------------------------------------------------- #
# 2. entropy / mutual-information estimators over sampled answers
# --------------------------------------------------------------------------- #
def plugin_entropy(symbols: Iterable, base: float = 2.0) -> float:
    """Plug-in (MLE) Shannon entropy of an empirical symbol multiset, in bits.

    `None` (unparseable answer) is treated as its own symbol -- a model that
    dissolves into noise has high entropy, which is the honest reading."""
    counts = Counter(symbols)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p, base)
    return h


def miller_madow_entropy(symbols: Iterable, base: float = 2.0) -> float:
    """Miller-Madow bias-corrected entropy: H_MLE + (K-1)/(2N).

    Plug-in entropy under-estimates with few samples; the correction matters at
    the small K we can afford per puzzle."""
    counts = Counter(symbols)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = plugin_entropy(symbols, base)
    k = len(counts)
    return h + (k - 1) / (2 * n) / math.log(base)


def delivered_mi(answers_textonly: Sequence, answers_with_rollout: Sequence,
                 base: float = 2.0, corrected: bool = True) -> float:
    """dI = H(Y|X) - H(Y|X,R): change in the reasoner's *self*-uncertainty.

    A diagnostic of how the rollout reorganises the model's answer distribution.
    NOTE: this is uncertainty about the model's OWN answer, not about the correct
    one -- a confidently-wrong text-only model has low H(Y|X), so a rollout that
    fixes it can *raise* entropy. For the decision-relevant signal use
    `task_information_gain` (information about the TRUE answer)."""
    H = miller_madow_entropy if corrected else plugin_entropy
    return H(answers_textonly, base) - H(answers_with_rollout, base)


def task_information_gain(n_correct_textonly: int, n_correct_rollout: int,
                          K: int, base: float = 2.0, alpha: float = 0.5) -> float:
    """Bits of decision-relevant information the rollout delivers about the TRUE
    answer Y*: the model's surprisal reduction on Y*,

        IG = log [ P(Y* | X, R) ] - log [ P(Y* | X) ],

    with P estimated from the fraction of sampled answers that are correct under
    add-alpha (Laplace) smoothing: P = (n_correct + alpha) / (K + 2*alpha).
    Positive => the rollout makes the correct answer more probable (the EWM
    faithfulness claim, in bits). Tracks accuracy lift but is information-scaled:
    going from 'almost never right' to 'often right' counts for many bits."""
    p_t = (n_correct_textonly + alpha) / (K + 2 * alpha)
    p_r = (n_correct_rollout + alpha) / (K + 2 * alpha)
    return math.log(p_r, base) - math.log(p_t, base)


# --------------------------------------------------------------------------- #
# 3. one-shot summary record for a puzzle
# --------------------------------------------------------------------------- #
def faithfulness_record(rec: dict, line_uci: Sequence[str],
                        answers_textonly: Sequence,
                        answers_with_rollout: Sequence,
                        acc_textonly: float, acc_with_rollout: float) -> dict:
    """Bundle capacity + information gain + accuracy lift for one puzzle.

    Accuracy is passed as a fraction; K is recovered from the sample counts so
    `task_information_gain` sees integer correct-counts."""
    cap = line_channel_bits(rec["fen"], line_uci) if line_uci else \
        position_branching_bits(rec["fen"])
    K = len(answers_textonly)
    n_t = round(acc_textonly * K)
    n_r = round(acc_with_rollout * K)
    ig = task_information_gain(n_t, n_r, K)
    dI = delivered_mi(answers_textonly, answers_with_rollout)
    return {
        "id": rec["id"],
        "kind": rec["kind"],
        "needs_visual": bool(rec["needs_visual"]),
        "capacity_bits": cap,
        "H_textonly": miller_madow_entropy(answers_textonly),
        "H_rollout": miller_madow_entropy(answers_with_rollout),
        "self_mi": dI,                       # diagnostic: self-uncertainty change
        "info_gain": ig,                     # bits about the TRUE answer (primary)
        "utilization": (ig / cap) if cap > 0 else 0.0,
        "acc_textonly": acc_textonly,
        "acc_rollout": acc_with_rollout,
        "acc_lift": acc_with_rollout - acc_textonly,
    }
