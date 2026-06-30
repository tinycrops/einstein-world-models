"""Stage-1 SFT, FOR REAL: LoRA-fine-tune Qwen2.5-0.5B-Instruct on gold chess
EWM traces with the Appendix A masked cross-entropy.

The masking is exact: we set `labels = -100` on (a) the problem/prompt tokens and
(b) every <visual_rollout> observation token, then call the HF model with
`labels=` -- whose internal CrossEntropyLoss ignores -100. That ignore-mask IS
L_SFT's 1_t indicator. So the model is trained to produce think/tool_call/answer
and to *read* (never reproduce) the returned rollout.

This is the first point where the EWM blueprint actually updates weights.

Run (free the GPU first if the vibethinker container is up):
    .venv-sd/bin/python scripts/chess_sft.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.chess_data import puzzle_prompt, write_dataset
from ewm.chess_traces import build_sft_corpus
from ewm.reasoner import SYSTEM_PROMPT
from ewm.trace import Trace

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_OUT = ROOT / "runs" / "chess_sft" / "adapter"


def encode_trace(trace: Trace, tok, system: str):
    """Return (input_ids, labels) with -100 on prompt + rollout tokens."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": trace.problem}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    inp = list(prompt_ids)
    lab = [-100] * len(prompt_ids)          # x is given -> masked
    for s in trace.segments:
        ids = tok.encode(s.render(), add_special_tokens=False)
        inp += ids
        lab += ids if s.kind.generated else [-100] * len(ids)  # rollout masked
    inp += [tok.eos_token_id]
    lab += [tok.eos_token_id]
    return inp, lab


def main() -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    recs = write_dataset(ROOT / "data" / "chess_puzzles.jsonl")
    corpus = build_sft_corpus(recs)
    n_call = sum(t.n_world_calls > 0 for t in corpus)
    print(f"gold corpus: {len(corpus)} traces ({n_call} call, {len(corpus)-n_call} no-call) "
          f"-> balanced={n_call>0 and n_call<len(corpus)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    data = [encode_trace(t, tok, SYSTEM_PROMPT) for t in corpus]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    model.train()
    EPOCHS = 6
    for ep in range(EPOCHS):
        tot = 0.0
        for inp, lab in data:
            input_ids = torch.tensor([inp], device=dev)
            labels = torch.tensor([lab], device=dev)
            out = model(input_ids=input_ids, labels=labels)  # HF ignores -100 -> L_SFT
            out.loss.backward()
            opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"epoch {ep+1}/{EPOCHS}  mean masked-CE loss = {tot/len(data):.4f}")

    ADAPTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT)
    print(f"saved LoRA adapter -> {ADAPTER_OUT}")

    # ---- qualitative before/after: does it learn to CALL on a mate puzzle and
    #      NOT call on a factual one? (selectivity is the whole point) ----
    model.eval()
    for rec in (next(r for r in recs if r["needs_visual"]),
                next(r for r in recs if not r["needs_visual"])):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": puzzle_prompt(rec)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        called = "<tool_call>" in out
        print(f"\n[{rec['id']} needs_visual={rec['needs_visual']}] "
              f"calls_W={called}  -> {out[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
