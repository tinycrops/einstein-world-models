"""Controlled selectivity probe: is the EWM reward signal correctly ORIENTED?

The GRPO group-advantage collapses to zero once the policy is deterministic
(SFT solved the toy set), so it can't *exhibit* the signal's orientation. This
probe measures the orientation directly, independent of current policy variance.

For every puzzle we run two forced regimes through the SAME model:
  * CALL    : seed think + tool_call(FEN); run the real world-module; inject the
              rollout; let the model answer.   reward = r(yhat,y*) - lambda*1/B
  * NO-CALL : seed a 'answer directly' think;  let the model answer.
              reward = r(yhat,y*)              (no penalty, no rollout)

Then we compare r_M(call) vs r_M(no-call). The §2.4.1 prediction:
  * tactical (needs_visual): the rollout reveals the mate -> r_M(call) > r_M(no-call)
                             => the advantage points toward CALLING.
  * factual:                 both answer correctly, but the call only burns the
                             penalty -> r_M(no-call) > r_M(call)
                             => the advantage points toward NOT calling.
That sign flip across categories IS learned selective visualisation.
"""

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
from ewm.reasoner import SYSTEM_PROMPT
from ewm.reward import world_use_penalty
from ewm.trace import Trace

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = ROOT / "runs" / "chess_sft" / "adapter"
LAM = 0.25


def _answer_after(model, tok, dev, prefix_text: str) -> str | None:
    ids = tok(prefix_text, return_tensors="pt").to(dev)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=40, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    out = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
    return m.group(1).strip() if m else out.strip()


def probe(model, tok, dev, rec) -> tuple[float, float]:
    base = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": puzzle_prompt(rec)}],
        add_generation_prompt=True, tokenize=False)

    # CALL regime: real rollout injected, then model answers
    W = ChessWorldModule(anchor_fen=rec["fen"])
    roll = W.rollout(rec["fen"], Path(f"runs/chess_probe/{rec['id']}"))
    obs = chess_perceive(roll.frame_paths, rec["fen"])
    seed_call = (base
                 + "<think>This looks tactical; let me visualise the line.</think>"
                 + Trace("").tool_call(rec["fen"]).segments[0].render()
                 + f"<visual_rollout>{obs}</visual_rollout><answer>")
    ans_call = _answer_after(model, tok, dev, seed_call)
    r_call = verify(rec, ans_call) + world_use_penalty(1, lam=LAM)

    # NO-CALL regime: answer directly
    seed_nc = base + "<think>I can answer this directly.</think><answer>"
    ans_nc = _answer_after(model, tok, dev, seed_nc)
    r_nc = verify(rec, ans_nc) + world_use_penalty(0, lam=LAM)
    return r_call, r_nc


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev)
    if ADAPTER.exists():
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    recs = [json.loads(l) for l in (ROOT / "data" / "chess_puzzles.jsonl").read_text().splitlines()]
    cats = {True: [], False: []}
    print(f"{'puzzle':16} vis  r_M(call)  r_M(nocall)  favors")
    for rec in recs:
        r_call, r_nc = probe(model, tok, dev, rec)
        fav = "CALL" if r_call > r_nc else ("NO-CALL" if r_nc > r_call else "tie")
        cats[rec["needs_visual"]].append((r_call, r_nc))
        print(f"{rec['id']:16} {int(rec['needs_visual'])}   {r_call:+.3f}     {r_nc:+.3f}     {fav}")

    print("\n=== mean reward by regime (the signal orientation) ===")
    for vis, name in ((True, "tactical (needs visual)"), (False, "factual (no visual)")):
        rc = mean(c for c, _ in cats[vis]); rn = mean(n for _, n in cats[vis])
        verdict = "-> CALL" if rc > rn else "-> NO-CALL"
        print(f"{name:26}: r_M(call)={rc:+.3f}  r_M(nocall)={rn:+.3f}   advantage {verdict}")
    print("\nSign flip across categories = the reward teaches WHEN to visualise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
