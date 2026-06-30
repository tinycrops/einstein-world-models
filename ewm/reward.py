"""EWM reward (Eq. 2) and the selective-use penalty (§2.4.1).

    r_M(T, y*) = r(yhat, y*) + r_W(T)

r is final-answer correctness (verifiable -> RLVR). r_W is the optional,
implementation-dependent term that shapes *when* the world-module is used. The
paper's worked example:

    r_W(T) = -lambda * M(T) / B

where M(T) counts world-module calls, B is a call budget, lambda >= 0. With
lambda = 0 selectivity is learned from the answer reward alone (Search-R1 /
ToolRL style); lambda > 0 explicitly discourages over-calling, so a call must
"pay for itself" in correctness to survive the group-relative advantage.
"""

from __future__ import annotations

import re


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(".!?")


def answer_reward(pred: str | None, gold: str) -> float:
    """r(yhat, y*) = 1[yhat == y*] for exact-answer tasks."""
    if pred is None:
        return 0.0
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def world_use_penalty(n_calls: int, lam: float = 0.25, budget: int = 2) -> float:
    """r_W(T) = -lambda * M(T) / B."""
    if lam <= 0:
        return 0.0
    return -lam * n_calls / max(1, budget)


def ewm_reward(pred: str | None, gold: str, n_calls: int,
               lam: float = 0.25, budget: int = 2) -> float:
    """r_M(T, y*) -- the scalar GRPO optimizes per trajectory."""
    return answer_reward(pred, gold) + world_use_penalty(n_calls, lam, budget)
