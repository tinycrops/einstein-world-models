"""Stage-2: RLVR over complete EWM trajectories, GRPO objective (Eq. 3).

For each problem (x, y*) we sample G complete EWM rollouts with pi_old (which
has access to W), score each with r_M (reward.py), and form group-relative
advantages:

    A_i = (r_i - mean_j r_j) / (std_j r_j + eps)

The clipped surrogate is summed over POLICY-GENERATED tokens only -- the same
1_it mask as SFT, so the returned-rollout observations never receive a policy
gradient (they are observations, not actions):

    J_E = E[ (1/G) sum_i (1 / L_i^g) sum_t 1_it * min(rho_it A_i, clip(rho_it) A_i) ]
          - beta * KL(pi_theta || pi_ref)

    rho_it = pi_theta(z_it | tau_<t) / pi_old(z_it | tau_<t)

This module implements the objective over already-computed per-token ratios so
the GRPO bookkeeping (group advantages + rollout masking + clipping +
per-trajectory generated-token normalization) is unit-testable without a
training stack. Wiring it to a real model = supply rho/KL from forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """A_i for one group of G trajectories sharing a problem."""
    g = len(rewards)
    if g == 0:
        return []
    mean = sum(rewards) / g
    var = sum((r - mean) ** 2 for r in rewards) / g
    std = var ** 0.5
    return [(r - mean) / (std + eps) for r in rewards]


def _clip(x: float, eps: float) -> float:
    return max(1 - eps, min(1 + eps, x))


@dataclass
class Trajectory:
    advantage: float
    ratios: Sequence[float]      # rho_it for every token position t
    gen_mask: Sequence[int]      # 1_it: 1 = policy-generated, 0 = rollout obs
    kl: float = 0.0              # per-trajectory KL(pi_theta || pi_ref)


def trajectory_surrogate(traj: Trajectory, clip_eps: float = 0.2) -> float:
    """Inner term: (1/L_i^g) sum_t 1_it min(rho A, clip(rho) A)."""
    Lg = sum(traj.gen_mask)
    if Lg == 0:
        return 0.0
    A = traj.advantage
    acc = 0.0
    for rho, m in zip(traj.ratios, traj.gen_mask):
        if m == 0:                       # rollout observation -> no gradient
            continue
        acc += min(rho * A, _clip(rho, clip_eps) * A)
    return acc / Lg


def grpo_objective(group: Sequence[Trajectory], beta: float = 0.0,
                   clip_eps: float = 0.2) -> float:
    """J_E for one group: mean surrogate minus beta * mean KL."""
    if not group:
        return 0.0
    surr = sum(trajectory_surrogate(t, clip_eps) for t in group) / len(group)
    kl = sum(t.kl for t in group) / len(group)
    return surr - beta * kl
