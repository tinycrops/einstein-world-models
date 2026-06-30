"""The EWM inference control loop -- Eq. 1 made executable.

    T_0 = x
    at each step:
        seg ~ pi_theta(. | T_t)
        if seg is a world-module query q_t:
            v_t ~ W(q_t)                      # render frames
            obs = perceive(v_t)               # frames -> visual tokens
            T_{t+1} = T_t  +  [q_t, v_t]       # append query AND observation
        elif seg is final answer:
            return answer
        else:
            T_{t+1} = T_t + s_t                # plain text segment

The returned bundle is the paper's *inspectable hypothesis* artifact (§2.2):
the full serialized trace, every frame on disk, per-call provenance, and the
loss mask -- everything needed to inspect, test, and improve the rollout, and
everything sft.py/grpo.py need to train on it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import perception
from .reasoner import Reasoner
from .trace import Kind, Trace, next_control_segment, parse_generated
from .world_module import WorldModule


def run_episode(problem: str, reasoner: Reasoner, world: WorldModule,
                out_dir: Path, max_steps: int = 8, n_frames: int = 4,
                perceive=perception.describe_rollout) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = Trace(problem=problem)
    call_idx = 0

    for step in range(max_steps):
        raw = reasoner.step(trace.render())
        segs = parse_generated(raw)
        if not segs:  # degenerate output; nudge once then stop
            trace.think("(no parseable output)")
            break

        ctrl = next_control_segment(segs)
        # Commit any leading think segments up to (and including) the control one.
        for s in segs:
            if s.kind is Kind.THINK:
                trace.append(s)
            if s is ctrl:
                break

        if ctrl is None:
            continue  # pure thinking; keep generating

        if ctrl.kind is Kind.ANSWER:
            trace.append(ctrl)
            break

        if ctrl.kind is Kind.TOOL_CALL:
            query = ctrl.meta.get("query") or ctrl.text
            trace.append(ctrl)                                  # append q_t
            frames_dir = out_dir / f"call_{call_idx:02d}"
            rollout = world.rollout(query, frames_dir, n_frames=n_frames)  # v_t ~ W
            obs = perceive(rollout.frame_paths, query)          # -> visual tokens
            trace.visual_rollout(obs, query=query,
                                 frame_paths=rollout.frame_paths,
                                 beats=rollout.beats,
                                 module=rollout.module,
                                 seconds=rollout.seconds)        # append v_t
            call_idx += 1
            continue

    bundle = _bundle(trace, problem)
    (out_dir / "trace.json").write_text(json.dumps(bundle, indent=2))
    (out_dir / "trace.txt").write_text(trace.render())
    return bundle


def _bundle(trace: Trace, problem: str) -> dict:
    return {
        "problem": problem,
        "final_answer": trace.final_answer,
        "n_world_calls": trace.n_world_calls,
        "serialized": trace.render(),
        "segments": [
            {"kind": s.kind.value, "generated": s.kind.generated,
             "text": s.text, "meta": s.meta}
            for s in trace.segments
        ],
        # the exact (span, is_generated) mask that training masks rollouts with
        "loss_mask_spans": [
            {"generated": g, "text": t} for t, g in trace.mask_spans()
        ],
    }
