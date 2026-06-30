"""Stage-1 warm start: masked supervised fine-tuning on EWM traces (Appendix A).

The SFT loss is ordinary next-token cross-entropy, but summed ONLY over tokens
the reasoner is expected to produce:

    L_SFT = - E[ (1 / sum_t 1_t) * sum_t 1_t * log pi_theta(z*_t | T*_<t) ]

with 1_t = 0 on returned <visual_rollout> observations (and on the given
problem x). Appendix A also stresses: D_SFT must contain BOTH call and no-call
traces, so the model learns the world-module *interface* without learning to
invoke W by default.

This module gives the two things a real SFT run needs:
  1. build_example(): turn a Trace + HF tokenizer into (input_ids, labels) with
     -100 on masked positions -- the exact tensor HF Trainer consumes.
  2. masked_ce(): the loss itself, expressed over per-token log-probs + mask, so
     it is unit-testable without a GPU.
"""

from __future__ import annotations

from typing import Sequence

from .trace import Trace

IGNORE = -100  # HF convention: CrossEntropyLoss ignores these label positions


def build_example(trace: Trace, tokenizer) -> dict:
    """Return {'input_ids', 'labels'} where labels == IGNORE on masked tokens.

    The mask comes straight from Trace.loss_mask, i.e. from Kind.generated, so
    rollout observations and the prompt are excluded from the loss exactly as
    Appendix A specifies.
    """
    ids, mask = trace.loss_mask(tokenizer)
    labels = [tok if m == 1 else IGNORE for tok, m in zip(ids, mask)]
    return {"input_ids": ids, "labels": labels}


def masked_ce(token_logprobs: Sequence[float], mask: Sequence[int]) -> float:
    """L_SFT for one trace given per-token log pi(z*_t) and the 1_t mask.

    Returns the masked mean negative log-likelihood. Tokenizer/GPU-free so the
    masking semantics can be tested directly.
    """
    denom = sum(mask)
    if denom == 0:
        return 0.0
    total = -sum(lp * m for lp, m in zip(token_logprobs, mask))
    return total / denom


def has_balanced_corpus(traces: Sequence[Trace]) -> bool:
    """Appendix A guardrail: D_SFT should hold both call and no-call traces."""
    call = any(t.n_world_calls > 0 for t in traces)
    nocall = any(t.n_world_calls == 0 for t in traces)
    return call and nocall
