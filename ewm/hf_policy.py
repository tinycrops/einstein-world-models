"""HF policy utilities for real EWM training: structured trajectory sampling
(with tool use), masked trajectory encoding, and per-token log-probs.

Shared by SFT (chess_sft) and GRPO (chess_grpo_train). The masking contract is
the paper's: prompt x and returned <visual_rollout> tokens are masked (1_t=0);
think/tool_call/answer are policy tokens (1_t=1). Same predicate (Kind.generated)
everywhere, so SFT and GRPO mask identically.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .chess_data import puzzle_prompt
from .chess_world import ChessWorldModule, chess_perceive
from .reasoner import SYSTEM_PROMPT
from .trace import Kind, Trace, parse_generated

_STOPS = ("</tool_call>", "</answer>")


def _truncate_at_stop(text: str) -> tuple[str, str | None]:
    hits = [(text.find(s) + len(s), s) for s in _STOPS if s in text]
    if not hits:
        return text, None
    end, which = min(hits)
    return text[:end], which


def chat_prefix(tok, problem: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": problem}],
        add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def sample_trajectory(model, tok, rec, dev, temperature=1.0,
                      max_calls=1, max_chunks=4) -> Trace:
    """Sample a complete EWM trajectory (with real chess world-module calls)
    and return it as a structured Trace."""
    problem = puzzle_prompt(rec)
    prefix = chat_prefix(tok, problem)
    world = ChessWorldModule(anchor_fen=rec["fen"])
    trace = Trace(problem=problem)
    assistant, n_calls = "", 0
    for _ in range(max_chunks):
        ids = tok(prefix + assistant, return_tensors="pt").to(dev)
        gen = model.generate(**ids, max_new_tokens=90, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-3), top_p=0.95,
                             pad_token_id=tok.eos_token_id)
        chunk = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        chunk, stop = _truncate_at_stop(chunk)
        for seg in parse_generated(chunk):
            trace.append(seg)
        assistant += chunk
        if stop == "</answer>":
            break
        if stop == "</tool_call>" and n_calls < max_calls:
            q = trace.segments[-1].meta.get("query") or rec["fen"]
            roll = world.rollout(q, Path(f"runs/grpo_sample/{rec['id']}/c{n_calls}"))
            obs = chess_perceive(roll.frame_paths, rec["fen"])
            trace.visual_rollout(obs, query=q)
            assistant += f"<visual_rollout>{obs}</visual_rollout>"
            n_calls += 1
            continue
        # no control tag emitted -> stop (the policy failed to close a segment)
        break
    return trace


def encode_trajectory(trace: Trace, tok) -> tuple[list[int], list[int]]:
    """(input_ids, gen_mask): prompt via chat template (masked), then segments
    with <visual_rollout> masked. gen_mask[t]=1 iff token t is a policy token."""
    prefix_ids = tok(chat_prefix(tok, trace.problem))["input_ids"]
    ids = list(prefix_ids)
    mask = [0] * len(prefix_ids)
    for s in trace.segments:
        seg_ids = tok.encode(s.render(), add_special_tokens=False)
        ids += seg_ids
        mask += [1 if s.kind.generated else 0] * len(seg_ids)
    return ids, mask


def token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position log pi(z_t | z_<t) for t=1..L-1 (aligned to input_ids[1:]).

    Memory-efficient: log p(target) = logit[target] - logsumexp(logits) avoids
    materializing the (L, V) float softmax (V=151936 for Qwen), so the policy
    AND a frozen reference model both fit in 8GB for the KL term.
    """
    logits = model(input_ids).logits[0][:-1]       # (L-1, V), drop last pos
    tgt = input_ids[0, 1:]                          # next-token targets
    gathered = logits.gather(1, tgt.unsqueeze(1)).squeeze(1).float()
    lse = torch.logsumexp(logits, dim=-1).float()   # reduction, no V-wide copy
    return gathered - lse                            # (L-1,)


def final_answer_move(trace: Trace) -> str | None:
    fa = trace.final_answer
    if fa:
        return fa
    for s in reversed(trace.segments):
        if s.kind is Kind.ANSWER:
            return s.text
    return None
