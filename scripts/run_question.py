"""Full REAL EWM episode: local trainable reasoner + SD world-module +
gpt-5.4-mini perception.

    reasoner   : local Ollama model (CPU, so the 1060 stays free for SD)
    world W    : SD1.5 dreamshaper_8 frame rollout on the GPU
    perceive   : gpt-5.4-mini reads the frames back into the trace

Usage:
    .venv-sd/bin/python scripts/run_question.py [question_id] [--model NAME]

Writes an inspectable bundle to runs/<id>/.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ewm.inference import run_episode
from ewm.reasoner import OllamaReasoner
from ewm.reward import ewm_reward
from ewm.world_module import SDRenderer


def load_problems() -> dict:
    out = {}
    for line in (ROOT / "data" / "simplebench_like.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("qid", nargs="?", default="balls-juggler")
    ap.add_argument("--model", default="mistral:7b-instruct")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    problems = load_problems()
    prob = problems[args.qid]
    print(f"[{args.qid}] needs_visual={prob['needs_visual']}")
    print(f"Q: {prob['problem']}\n")

    bundle = run_episode(
        problem=prob["problem"],
        reasoner=OllamaReasoner(model=args.model, num_gpu=0),  # CPU reasoner
        world=SDRenderer(steps=args.steps),                     # GPU renderer
        out_dir=ROOT / "runs" / args.qid,
        n_frames=args.frames,
        max_steps=6,
    )
    r = ewm_reward(bundle["final_answer"], prob["answer"], bundle["n_world_calls"])
    print(f"\nfinal_answer : {bundle['final_answer']}")
    print(f"gold         : {prob['answer']}")
    print(f"world_calls M: {bundle['n_world_calls']}")
    print(f"ewm_reward   : {r:.3f}")
    print(f"bundle       : {ROOT / 'runs' / args.qid / 'trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
