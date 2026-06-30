"""Ground the EWM hypothesis: measure the decision-relevant mutual information
a spliced rollout injects into the reasoning trace -- per puzzle kind.

EWM predicts the rollout helps *selectively*: it should remove answer-uncertainty
(and lift accuracy) exactly where text alone underdetermines the answer (tactical
mate puzzles), and do ~nothing where the text already determines it (factual
puzzles). We test this causally with a single reasoner held fixed, toggling only
whether the rollout is present:

    cond A (text-only):  X            -> sample K answers
    cond B (rollout):    X + <rollout> -> sample K answers

For each puzzle:
    capacity_bits = chess-move channel width of the rollout line (chessencryption)
    delivered_mi  = H(Y|X) - H(Y|X,R)     (bits the rollout removes)
    acc_lift      = acc_B - acc_A

The rollout R is the FAITHFUL forcing line from the chess world-module (the same
observation `chess_traces.gold_trace` splices), serialised to text by
`chess_perceive`. So this measures how many of the line's bits the reasoner can
actually use -- the §3.2 faithfulness claim, as a number.

Run (model cached under .hfcache):
  docker run --rm --gpus all -v $PWD:/w -w /w -e HF_HOME=/w/.hfcache ath-ewm:latest \
    python3 scripts/faithfulness_mi.py --corpus data/chess_corpus_large.jsonl \
      --n-per-kind 24 --K 8 --temp 0.8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.chess_data import _parse_move, puzzle_prompt, verify
from ewm.chess_world import ChessWorldModule, chess_perceive
from ewm.infometric import faithfulness_record
from ewm.reasoner import SYSTEM_PROMPT

ANSWER_INSTR = ("Give ONLY your final answer move in SAN on the last line as: "
                "Answer: <move>")


def rollout_observation(rec: dict, tmp: Path, mode: str = "forward") -> tuple[str, list[str]]:
    """The faithful world-module rollout as observation text, plus its UCI move
    line (for the channel-capacity bound).

    mode='forward'        -> the forcing line from the position (default).
    mode='counterfactual' -> VOID-style intervention: on a defender-hinged tactic,
                             remove the critical defender and show the consequence;
                             falls back to forward when no such defender exists."""
    import chess
    from ewm.chess_world import critical_defender
    W = ChessWorldModule(anchor_fen=rec["fen"])
    if mode == "counterfactual" and rec.get("needs_visual"):
        sq = critical_defender(rec["fen"], rec.get("solution_uci", []))
        roll = (W.counterfactual_rollout(tmp, sq) if sq
                else W.rollout(f"visualize the forcing line from {rec['fen']}", tmp))
    else:
        roll = W.rollout(f"visualize the forcing line from {rec['fen']}", tmp)
    obs = chess_perceive(roll.frame_paths, rec["fen"])
    # reconstruct UCI line from the SAN beats so capacity covers the whole line
    line_uci: list[str] = []
    board = chess.Board(rec["fen"])
    san_line = roll.beats[0] if roll.beats else ""
    for san in san_line.split():
        try:
            mv = board.parse_san(san)
        except ValueError:
            break
        line_uci.append(mv.uci())
        board.push(mv)
    return obs, line_uci or rec.get("solution_uci", [])


def _build_user(problem: str, rollout: str | None) -> str:
    """The user turn: the problem, the answer-format instruction, and (optionally)
    the spliced rollout. Shared by the local-HF and served-API sampling paths so
    the only thing that varies across the size curve is the reasoner."""
    user = problem + "\n" + ANSWER_INSTR
    if rollout is not None:
        user += (f"\n\nA visual-temporal rollout of the position has been "
                 f"produced for you:\n<visual_rollout>{rollout}</visual_rollout>\n"
                 f"Use it if helpful.")
    return user


@torch.no_grad()
def sample_answers(model, tok, dev, problem: str, rollout: str | None,
                   K: int, temp: float, max_new=64) -> list:
    """Sample K answers (local HF model); parsed canonical moves (uci) or None."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user(problem, rollout)}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").to(dev)
    out = []
    for _ in range(K):
        gen = model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                             temperature=max(temp, 1e-3), top_p=0.95,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        mv = _parse_move(__import__("chess").Board(problem_fen(problem)), txt)
        out.append(mv.uci() if mv else None)
    return out


def sample_answers_api(endpoint: str, served_model: str, problem: str,
                       rollout: str | None, K: int, temp: float,
                       max_new: int = 512, cooldown: float = 0.0) -> list:
    """Sample K answers from an OpenAI-compatible server (llama.cpp serving the
    Gemma-4 family on the GB10 — see ../gemma4-llama-dgx-spark). Lets the size
    curve extend to served models (e2b/e4b/26b-MoE/31b-dense) over HTTP, no local
    weights. Thinking models (26b/31b) put CoT in reasoning_content and the move
    in content -- we parse the move from content. max_new is generous so a
    <think> block doesn't truncate the final answer. `cooldown` paces requests to
    duty-cycle a heavy served model (e.g. 31B) below the GPU thermal cap on the
    shared box -- the REQUEST_COOLDOWN_S pattern (see gary-host-thermal-policy)."""
    import time

    import requests
    url = endpoint.rstrip("/") + "/chat/completions"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user(problem, rollout)}]
    board_fen = problem_fen(problem)
    out = []
    for _ in range(K):
        try:
            r = requests.post(url, json={"model": served_model, "messages": msgs,
                                         "temperature": max(temp, 1e-3), "top_p": 0.95,
                                         "max_tokens": max_new}, timeout=180)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            # Gemma-4 E-series are REASONING-NATIVE: CoT goes to reasoning_content,
            # the final "Answer:" to content. With too small a budget the think block
            # truncates (finish=length) and content is EMPTY -> a silent all-zeros
            # artifact. max_new must be generous (~3k); if content is still empty we
            # fall back to parsing reasoning_content so we never silently zero out.
            txt = msg.get("content") or ""
            if not txt.strip():
                txt = msg.get("reasoning_content") or ""
        except Exception:
            txt = ""
        mv = _parse_move(__import__("chess").Board(board_fen), txt)
        out.append(mv.uci() if mv else None)
        if cooldown > 0:
            time.sleep(cooldown)
    return out


def problem_fen(problem: str) -> str:
    for ln in problem.splitlines():
        if ln.strip().startswith("FEN:"):
            return ln.split("FEN:", 1)[1].strip()
    raise ValueError("no FEN in problem")


def verify_move(rec: dict, uci) -> float:
    return verify(rec, uci) if uci else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/chess_corpus_large.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-per-kind", type=int, default=24)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--rollout-mode", choices=["forward", "counterfactual"],
                    default="forward", help="forward forcing line vs VOID-style intervention")
    ap.add_argument("--endpoint", default=None,
                    help="OpenAI-compatible base URL (e.g. http://localhost:18080/v1) "
                         "to sample a SERVED reasoner instead of a local HF model "
                         "(scales the curve to the Gemma-4 family on the GB10)")
    ap.add_argument("--served-model", default="gemma-4-31b",
                    help="model name to send to --endpoint")
    ap.add_argument("--max-new", type=int, default=512,
                    help="max_tokens for served models; reasoning-native Gemma-4 "
                         "needs ~3072 so the <think> block doesn't truncate the answer")
    ap.add_argument("--req-cooldown", type=float, default=0.0,
                    help="seconds to sleep between served requests (thermal duty-cycle)")
    ap.add_argument("--out", default="runs/faithfulness_mi.jsonl")
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)

    recs = [json.loads(l) for l in (ROOT / args.corpus).read_text().splitlines() if l.strip()]
    by_kind: dict[str, list] = defaultdict(list)
    for r in recs:
        by_kind[r["kind"]].append(r)
    sel = []
    for kind, group in by_kind.items():
        random.shuffle(group)
        sel.extend(group[:args.n_per_kind])
    print(f"corpus {len(recs)} -> {len(sel)} sampled across {len(by_kind)} kinds "
          f"({sum(r['needs_visual'] for r in sel)} needs_visual)")

    # one sampler interface, two backends: served API (Gemma-4 via llama.cpp) or
    # local HF weights (Qwen). Only the reasoner changes across the size curve.
    if args.endpoint:
        def sampler(problem, rollout):
            return sample_answers_api(args.endpoint, args.served_model, problem,
                                      rollout, args.K, args.temp,
                                      max_new=args.max_new,
                                      cooldown=args.req_cooldown)
        print(f"reasoner SERVED {args.served_model} @ {args.endpoint}; "
              f"K={args.K} temp={args.temp}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(dev).eval()

        def sampler(problem, rollout):
            return sample_answers(model, tok, dev, problem, rollout, args.K, args.temp)
        print(f"reasoner {args.model} on {dev}; K={args.K} temp={args.temp}")

    tmp = ROOT / "runs" / "mi_rollouts"
    records = []
    for i, rec in enumerate(sel):
        problem = puzzle_prompt(rec)
        obs, line_uci = rollout_observation(rec, tmp / rec["id"], args.rollout_mode)
        a_text = sampler(problem, None)
        a_roll = sampler(problem, obs)
        acc_t = mean(verify_move(rec, u) for u in a_text)
        acc_r = mean(verify_move(rec, u) for u in a_roll)
        fr = faithfulness_record(rec, line_uci, a_text, a_roll, acc_t, acc_r)
        records.append(fr)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sel)} done")

    (ROOT / args.out).write_text("\n".join(json.dumps(r) for r in records) + "\n")

    # ---- aggregate by kind: the grounding table ----
    print("\n=== EWM grounding: decision-relevant information delivered by the "
          "rollout, by kind ===")
    print("(cap=channel capacity of the move-line; IG=bits the rollout adds about "
          "the TRUE answer; lift=accuracy gain)")
    print(f"{'kind':<18}{'n':>3}{'cap_bits':>10}{'IG_bits':>9}{'util':>7}"
          f"{'acc0':>7}{'acc1':>7}{'lift':>7}{'self_mi':>9}")
    order = sorted({r["kind"] for r in records},
                   key=lambda k: (not records[[x["kind"] for x in records].index(k)]["needs_visual"], k))
    for kind in order:
        rs = [r for r in records if r["kind"] == kind]
        print(f"{kind:<18}{len(rs):>3}{mean(r['capacity_bits'] for r in rs):>10.2f}"
              f"{mean(r['info_gain'] for r in rs):>+9.2f}{mean(r['utilization'] for r in rs):>7.2f}"
              f"{mean(r['acc_textonly'] for r in rs):>7.2f}{mean(r['acc_rollout'] for r in rs):>7.2f}"
              f"{mean(r['acc_lift'] for r in rs):>+7.2f}{mean(r['self_mi'] for r in rs):>+9.2f}")

    nv = [r for r in records if r["needs_visual"]]
    fac = [r for r in records if not r["needs_visual"]]
    print("\n--- EWM prediction check (rollout helps selectively) ---")
    print(f"  needs_visual : info_gain={mean(r['info_gain'] for r in nv):+.2f} bits "
          f"acc_lift={mean(r['acc_lift'] for r in nv):+.2f}")
    print(f"  factual      : info_gain={mean(r['info_gain'] for r in fac):+.2f} bits "
          f"acc_lift={mean(r['acc_lift'] for r in fac):+.2f}")
    print(f"  -> selectivity gap (visual - factual) in info_gain: "
          f"{mean(r['info_gain'] for r in nv) - mean(r['info_gain'] for r in fac):+.2f} bits")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
