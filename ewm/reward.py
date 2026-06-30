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


def info_shaped_world_reward(n_calls: int, info_gain_bits: float,
                             lam_info: float = 0.25, lam_cost: float = 0.10,
                             budget: int = 2) -> float:
    """r_W shaped by MEASURED decision-relevant information (this work).

    The flat penalty `world_use_penalty` discourages *all* calls equally -- it
    only knows the budget, not whether a call was worth it. We replace it with a
    term that pays a call exactly its measured worth:

        r_W(T) = lam_info * IG(x) * 1[W called]  -  lam_cost * M(T) / B

    where IG(x) is the per-puzzle delivered information gain in bits (from
    `infometric.task_information_gain`, precomputed offline and attached to the
    corpus). Consequences, all the right sign:
      * high-IG (tactical) puzzle + call  -> rewarded (the call paid its bits);
      * factual puzzle (IG ~ 0) + call    -> only the cost term bites -> avoid;
      * a call whose rollout *removes* info (IG < 0) is actively penalised;
      * not calling forfeits the IG reward but pays no cost.

    This operationalises §2.4.1 ("selective thought experiments") with a reward
    grounded in the rollout's measured mutual information, rather than a flat
    budget penalty -- the reasoner learns to call *when it pays in bits*."""
    called = 1.0 if n_calls > 0 else 0.0
    return lam_info * info_gain_bits * called - lam_cost * n_calls / max(1, budget)


def ewm_reward_shaped(pred: str | None, gold: str, n_calls: int,
                      info_gain_bits: float, lam_info: float = 0.25,
                      lam_cost: float = 0.10, budget: int = 2) -> float:
    """r_M with the info-shaped r_W: answer correctness + paid-by-the-bit calls."""
    return answer_reward(pred, gold) + info_shaped_world_reward(
        n_calls, info_gain_bits, lam_info, lam_cost, budget)
