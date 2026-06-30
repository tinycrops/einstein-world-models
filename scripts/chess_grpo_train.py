"""Stage-2 GRPO, FOR REAL: actual weight updates that optimize J_E (Eq. 3).

Pipeline:
  1. scale + split the verifiable puzzle corpus (train / test).
  2. LIGHT SFT warm-start (Appendix A) -- enough to know the trace format, NOT
     enough to memorize, so the policy still explores (nonzero reward variance).
  3. GRPO rounds: sample G trajectories/prompt with the chess world-module,
     score with the verifiable r_M, form group-relative advantages, and take
     real clipped-surrogate steps over POLICY tokens only (rollout tokens masked).
  4. eval on held-out test before vs after -> did RL improve reward/selectivity?

The clipped surrogate uses pi_old log-probs captured at sampling time:
    rho = exp(logp_theta - logp_old);  min(rho A, clip(rho) A); mask; /L_g.
Single sampling per round + a couple inner epochs = standard GRPO.

Run after freeing the GPU (docker stop vibethinker):
    .venv-sd/bin/python scripts/chess_grpo_train.py
"""

import argparse
import random
import sys
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.chess_data import generate, verify
from ewm.chess_traces import gold_trace
from ewm.grpo import group_advantages
from ewm.hf_policy import (encode_trajectory, sample_trajectory,
                           token_logprobs)
from ewm.reward import info_shaped_world_reward, world_use_penalty

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAM = 0.25

# reward shaping config, set in main() from CLI flags
REWARD = {"mode": "flat", "lam_info": 0.25, "lam_cost": 0.10, "budget": 2}


def reward_of(rec, trace):
    """r_M for a trajectory. mode='flat' = paper's budget penalty; mode='info' =
    r_W shaped by the puzzle's MEASURED info gain (rec['info_gain'], in bits), so
    a world-call is paid exactly its worth (see ewm.reward.info_shaped_world_reward)."""
    ans = verify(rec, trace.final_answer)
    if REWARD["mode"] == "info":
        ig = float(rec.get("info_gain", 0.0))
        return ans + info_shaped_world_reward(
            trace.n_world_calls, ig, REWARD["lam_info"], REWARD["lam_cost"],
            REWARD["budget"])
    return ans + world_use_penalty(trace.n_world_calls, lam=LAM, budget=REWARD["budget"])


def light_sft(model, tok, recs, dev, epochs, lr=2e-4):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for rec in recs:
            ids, mask = encode_trajectory(gold_trace(rec), tok)
            ids = ids + [tok.eos_token_id]; mask = mask + [1]
            x = torch.tensor([ids], device=dev)
            labels = torch.tensor([[t if m else -100 for t, m in zip(ids, mask)]], device=dev)
            loss = model(input_ids=x, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item()
        print(f"  warm-sft epoch {ep+1}/{epochs}  loss={tot/len(recs):.4f}")


@torch.no_grad()
def evaluate(model, tok, recs, dev):
    model.eval()
    rew, acc, sel = [], [], []
    for rec in recs:
        tr = sample_trajectory(model, tok, rec, dev, temperature=0.0)
        rew.append(reward_of(rec, tr))
        acc.append(verify(rec, tr.final_answer))
        called = tr.n_world_calls > 0
        sel.append(1.0 if called == bool(rec["needs_visual"]) else 0.0)  # appropriate?
    return mean(rew), mean(acc), mean(sel)


def grpo_round(model, tok, train, dev, opt, G, temp, inner_epochs, ref=None,
               beta=0.0, clip=0.2):
    use_ref = ref is not None and beta > 0
    if use_ref:
        ref.to(dev)  # reference needed only to precompute ref logprobs below
    model.eval()
    samples, round_rewards = [], []
    for rec in train:
        traces = [sample_trajectory(model, tok, rec, dev, temperature=temp) for _ in range(G)]
        rewards = [reward_of(rec, t) for t in traces]
        round_rewards += rewards
        adv = group_advantages(rewards)
        for tr, A in zip(traces, adv):
            if A == 0.0:
                continue  # no gradient from a zero-variance group member
            ids, mask = encode_trajectory(tr, tok)
            x = torch.tensor([ids], device=dev)
            old = token_logprobs(model, x).detach()
            ref_lp = token_logprobs(ref, x).detach() if (ref is not None and beta > 0) else None
            samples.append((x, torch.tensor(mask[1:], device=dev, dtype=torch.float32),
                            A, old, ref_lp))
    if use_ref:  # ref logprobs are now stored; free its ~3GB for the backward pass
        ref.to("cpu")
        torch.cuda.empty_cache()
    if not samples:
        return mean(round_rewards), 0.0, 0
    model.train()
    last = 0.0
    for _ in range(inner_epochs):
        random.shuffle(samples)
        opt.zero_grad()
        for x, m, A, old, ref_lp in samples:
            new = token_logprobs(model, x)
            ratio = torch.exp(new - old)
            surr = torch.minimum(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A)
            Lg = m.sum().clamp(min=1)
            obj = (surr * m).sum() / Lg
            if ref_lp is not None:                       # -beta * KL(pi_theta||pi_ref)
                lr_ref = ref_lp - new                    # unbiased k3 KL estimator
                kl = (torch.exp(lr_ref) - lr_ref - 1.0)
                obj = obj - beta * (kl * m).sum() / Lg
            loss = -obj / len(samples)
            loss.backward()
            last += loss.item()
        opt.step()
    return mean(round_rewards), last, len(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="KL-to-SFT-reference weight")
    ap.add_argument("--model", default=MODEL, help="HF reasoner policy to fine-tune")
    ap.add_argument("--reward", choices=["flat", "info"], default="flat",
                    help="flat = paper budget penalty; info = r_W shaped by measured IG")
    ap.add_argument("--lam-info", type=float, default=0.25)
    ap.add_argument("--lam-cost", type=float, default=0.10)
    ap.add_argument("--corpus", default=None,
                    help="corpus jsonl; defaults to labeled for --reward info")
    ap.add_argument("--ntest", type=int, default=30)
    ap.add_argument("--train-cap", type=int, default=0,
                    help="cap #train puzzles (0 = all); keeps GRPO tractable")
    args = ap.parse_args()
    model_id = args.model
    REWARD["mode"] = args.reward
    REWARD["lam_info"], REWARD["lam_cost"] = args.lam_info, args.lam_cost

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0); torch.manual_seed(0)

    import json
    default_corpus = ("data/chess_corpus_large_labeled.jsonl" if args.reward == "info"
                      else "data/chess_corpus_large.jsonl")
    corpus_path = ROOT / (args.corpus or default_corpus)
    if not corpus_path.exists():
        corpus_path = ROOT / "data" / "chess_corpus.jsonl"
    if corpus_path.exists():
        recs = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
        print(f"loaded corpus from {corpus_path} (reward={args.reward})")
    else:
        print("generating scaled corpus...")
        recs = generate(n_mate=16, n_factual=12, seed=11, max_plies=3)
        corpus_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    if args.reward == "info" and not all("info_gain" in r for r in recs):
        raise SystemExit("--reward info needs an IG-labeled corpus "
                         "(run scripts/label_infogain.py first)")
    random.shuffle(recs)
    ntest = args.ntest
    test, train = recs[:ntest], recs[ntest:]
    if args.train_cap and len(train) > args.train_cap:
        # keep the test/train split fixed across reward arms (seed=0) but cap the
        # train set so the A/B is tractable; balance call/no-call within the cap.
        nv = [r for r in train if r["needs_visual"]][:args.train_cap // 2]
        fac = [r for r in train if not r["needs_visual"]][:args.train_cap - len(nv)]
        train = nv + fac
        random.shuffle(train)
    print(f"corpus {len(recs)}: {len(train)} train / {len(test)} test "
          f"({sum(r['needs_visual'] for r in train)} tactical in train)")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    print("light SFT warm-start...")
    light_sft(model, tok, train, dev, epochs=args.warm)
    if dev == "cuda":
        torch.cuda.empty_cache()  # release warm-SFT optimizer/activation cache

    # Freeze the SFT policy as the KL reference pi_ref (Eq. 3): a second copy of
    # the model with the just-trained adapter, no grad. Anchors GRPO so it can't
    # drift off the selectivity SFT installed -- the fix for the beta=0 collapse.
    ref = None
    if args.beta > 0:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM as _AMC
        ref_dir = ROOT / "runs" / "grpo_ref_adapter"
        model.save_pretrained(ref_dir)
        ref = PeftModel.from_pretrained(
            _AMC.from_pretrained(model_id, torch_dtype=torch.float16).to(dev), ref_dir)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(f"frozen SFT reference loaded (beta={args.beta})")

    r0, a0, s0 = evaluate(model, tok, test, dev)
    print(f"\nTEST before GRPO: reward={r0:+.3f} acc={a0:.2f} selectivity={s0:.2f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print("\nGRPO rounds (mean train reward should rise):")
    for rd in range(args.rounds):
        mr, loss, nused = grpo_round(model, tok, train, dev, opt,
                                     G=args.G, temp=args.temp, inner_epochs=args.inner,
                                     ref=ref, beta=args.beta)
        print(f"  round {rd+1}/{args.rounds}  train_reward={mr:+.3f}  "
              f"updated_on={nused} traj  surrogate_loss={loss:+.4f}")

    r1, a1, s1 = evaluate(model, tok, test, dev)
    print(f"\nTEST after  GRPO: reward={r1:+.3f} acc={a1:.2f} selectivity={s1:.2f}")
    print(f"DELTA: reward {r1-r0:+.3f}  acc {a1-a0:+.2f}  selectivity {s1-s0:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
