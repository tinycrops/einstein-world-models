"""Stage-2 GRPO signal, on REAL trajectories.

A full GRPO weight update on a 6GB card is impractical, but the scientific
question the paper never answers is: *is the learning signal correctly oriented?*
i.e. does the group-relative advantage actually push the policy toward
"call the world-module when it helps, and not when it doesn't" (§2.4.1)?

For each puzzle we sample G complete EWM trajectories from the policy (the SFT'd
Qwen + the chess world-module), score each with the verifiable reward
    r_M = r(yhat, y*)  +  r_W(T),     r_W = -lambda * M(T)/B
and compute group-relative advantages A_i. Then we check the sign of the signal:

  * tactical (needs_visual) puzzles -> trajectories that CALLED W and solved it
    should get the highest advantage  => GRPO increases P(call when tactical).
  * factual puzzles -> calling only burns the penalty => no-call correct
    trajectories win => GRPO increases P(not calling when trivial).

That orientation IS learned selective visualisation. This script measures it.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.chess_data import puzzle_prompt, verify
from ewm.chess_world import ChessWorldModule, chess_perceive
from ewm.grpo import group_advantages
from ewm.reasoner import SYSTEM_PROMPT
from ewm.reward import world_use_penalty

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = ROOT / "runs" / "chess_sft" / "adapter"
_STOPS = ("</tool_call>", "</answer>")


def _truncate_at_stop(text: str) -> tuple[str, str | None]:
    hits = [(text.find(s) + len(s), s) for s in _STOPS if s in text]
    if not hits:
        return text, None
    end, which = min(hits)
    return text[:end], which


def sample_trace(model, tok, rec, dev, temperature=0.9, max_calls=2, max_chunks=5):
    base = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": puzzle_prompt(rec)}],
        add_generation_prompt=True, tokenize=False)
    world = ChessWorldModule(anchor_fen=rec["fen"])
    assistant, n_calls, answer = "", 0, None
    for _ in range(max_chunks):
        ids = tok(base + assistant, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=90, do_sample=True,
                                 temperature=temperature, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        chunk = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        chunk, stop = _truncate_at_stop(chunk)
        assistant += chunk
        if stop == "</answer>":
            m = re.search(r"<answer>(.*?)</answer>", assistant, re.DOTALL)
            answer = m.group(1).strip() if m else assistant
            break
        if stop == "</tool_call>" and n_calls < max_calls:
            m = re.search(r"<tool_call>(.*?)</tool_call>", assistant, re.DOTALL)
            q = m.group(1) if m else rec["fen"]
            roll = world.rollout(q, Path(f"runs/chess_grpo/{rec['id']}/c{n_calls}"))
            assistant += f"<visual_rollout>{chess_perceive(roll.frame_paths, rec['fen'])}</visual_rollout>"
            n_calls += 1
            continue
        if stop is None:  # didn't emit a control tag; nudge toward an answer
            assistant += "<answer>"
    return {"answer": answer, "n_calls": n_calls, "text": assistant}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--n_mate", type=int, default=3)
    ap.add_argument("--n_fact", type=int, default=3)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev)
    if ADAPTER.exists():
        model = PeftModel.from_pretrained(model, ADAPTER)
        print(f"loaded SFT adapter from {ADAPTER}")
    model.eval()

    recs = [json.loads(l) for l in (ROOT / "data" / "chess_puzzles.jsonl").read_text().splitlines()]
    mate = [r for r in recs if r["needs_visual"]][:args.n_mate]
    fact = [r for r in recs if not r["needs_visual"]][:args.n_fact]

    rows = []
    for rec in mate + fact:
        traj = [sample_trace(model, tok, rec, dev) for _ in range(args.G)]
        rewards, corrects, calls = [], [], []
        for t in traj:
            r_ans = verify(rec, t["answer"])
            r = r_ans + world_use_penalty(t["n_calls"], lam=args.lam)
            rewards.append(r); corrects.append(r_ans); calls.append(t["n_calls"])
        adv = group_advantages(rewards)
        # advantage of "called W" vs "did not", within this group
        adv_call = [a for a, c in zip(adv, calls) if c > 0]
        adv_nocall = [a for a, c in zip(adv, calls) if c == 0]
        rows.append({
            "id": rec["id"], "needs_visual": rec["needs_visual"],
            "solve_rate": round(mean(corrects), 2),
            "call_rate": round(mean(1 if c > 0 else 0 for c in calls), 2),
            "adv_call": round(mean(adv_call), 3) if adv_call else None,
            "adv_nocall": round(mean(adv_nocall), 3) if adv_nocall else None,
        })
        print(f"[{rec['id']:14} vis={int(rec['needs_visual'])}] solve={rows[-1]['solve_rate']} "
              f"call={rows[-1]['call_rate']} | A(call)={rows[-1]['adv_call']} "
              f"A(nocall)={rows[-1]['adv_nocall']}")

    print("\n=== SELECTIVITY SIGNAL (mean group-relative advantage) ===")
    for vis, name in ((True, "tactical (needs visual)"), (False, "factual (no visual)")):
        grp = [r for r in rows if r["needs_visual"] == vis]
        ac = [r["adv_call"] for r in grp if r["adv_call"] is not None]
        an = [r["adv_nocall"] for r in grp if r["adv_nocall"] is not None]
        print(f"{name:26}: A(call)={mean(ac):+.3f} " if ac else f"{name:26}: A(call)=  n/a ", end="")
        print(f"A(nocall)={mean(an):+.3f}" if an else "A(nocall)=  n/a")
    print("\nGRPO pushes the policy UP the higher-advantage branch in each row.")
    (ROOT / "runs" / "chess_grpo" / "signal.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "chess_grpo" / "signal.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
