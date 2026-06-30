# Einstein World Models — a working instantiation

A from-scratch implementation of the EWM blueprint
(Nwadike et al., [*Einstein World Models*](https://arxiv.org/abs/2606.26969)),
wired to this cluster. The paper is a **position/blueprint paper** — no released
code, no experiments (it ends with a *call for datasets*). So "implementing it"
means instantiating the blueprint faithfully and making the load-bearing
mechanisms real and testable.

> **EWM in one sentence:** keep the LLM as the reasoner, but let it call a
> *world-module* (a video/renderer) at a sparse set of steps, splice the
> returned visual-temporal **rollout** back into the reasoning trace as an
> *inspectable hypothesis*, and train the model (SFT → RLVR) on *when* and *how*
> to do this — with the returned pixels masked from the loss.

This is the visual-thought-experiment analogue of tool-use (web search / code
exec): `<think>` → `<tool_call>` → `<visual_rollout>` → `<answer>`.

---

## The three mechanisms that actually carry the idea

Everything else is plumbing. Get these wrong and you haven't implemented the paper.

### 1. The trace + the loss mask  (`ewm/trace.py`)
A trace is `T_0 = x` plus appended segments (Eq. 1). Four tag kinds; **only
`<visual_rollout>` is a returned observation and is masked from the loss**
(`Kind.generated`). Both training objectives sum over `1_it` = "is this a
policy-generated token?":
- `<think>`, `<tool_call>`, `<answer>` → generated → **in** the loss
- `<visual_rollout>` and the given problem `x` → **masked** (`1_it = 0`)

The reasoner is never trained to *produce* the pixels W hands it — only to
decide when/how to query and how to use what comes back.

### 2. The inference loop  (`ewm/inference.py`)
Eq. 1 made executable:
```
seg ~ pi(.|T_t)
  tool_call q_t → v_t ~ W(q_t); obs = perceive(v_t); T += [q_t, v_t]
  answer        → return
  else (think)  → T += s_t
```
We halt generation at `</tool_call>` (Ollama `stop` tokens), run W, splice the
rollout, and continue — exactly how Search-R1-style tool loops work.

### 3. The training math  (`ewm/reward.py`, `ewm/sft.py`, `ewm/grpo.py`)
- **Reward (Eq. 2):** `r_M = r(ŷ,y*) + r_W(T)`. Answer reward is verifiable
  (RLVR). `r_W = −λ·M(T)/B` (§2.4.1) is the **selectivity** term — a world-call
  must "pay for itself" in correctness or the group-relative advantage punishes it.
- **SFT (Appendix A):** masked next-token CE; `build_example()` emits
  `labels = -100` on masked tokens (the exact tensor an HF Trainer eats).
  Guardrail: `D_SFT` must contain **both call and no-call traces** so the model
  doesn't learn to invoke W by default (`has_balanced_corpus`).
- **GRPO (Eq. 3):** group-relative advantages + clipped surrogate summed over
  **generated tokens only** (same mask), normalized by `L_i^g`, minus `β·KL`.

These are implemented over already-computed per-token ratios/logprobs so the
masking + clipping + group-advantage bookkeeping is **unit-tested without a GPU**
(`tests/test_ewm.py`, 13 tests). Wiring to a real model = supply ρ/KL/logprobs
from forward passes.

---

## How this maps to the cluster

| Paper piece | This repo | Hardware |
|---|---|---|
| Reasoner `π_θ` (trainable) | `OllamaReasoner` | local Ollama (VibeThinker-3B / mistral), CPU so the GPU stays free for W |
| World-module `W` (renderer → frames, §3.1) | `SDRenderer` | SD1.5 `dreamshaper_8` img2img on the GTX 1060 |
| "encode rollout into visual tokens" (§2.3) | `perception.py` | **gpt-5.4-mini** vision reads frames → literal text |
| offline everything | `MockReasoner` / `MockRenderer` | none (deterministic smoke) |

**World-module as a renderer.** The paper says renderers are the *default*
world-module and "repeated renderer calls may play the role of a simulator."
I have no video-diffusion model, so `SDRenderer` approximates a rollout as a
**prompt-stepped, img2img-chained** frame sequence (same scene at ordered
temporal beats). For trajectory questions this captures exactly the
reasoning-relevant content — *how the scene evolves in time*.

---

## Run it

```bash
# 1. Offline, deterministic — proves the Eq. 1 loop + masking. No GPU/API.
python3 scripts/smoke.py
python3 -m pytest -q tests/         # 13 tests: trace/mask/reward/grpo

# 2. Real episode: local reasoner + SD on GPU + gpt-5.4-mini vision.
#    Needs the SD venv (system-site torch/diffusers + pinned transformers):
.venv-sd/bin/python scripts/run_question.py balls-juggler --model mistral:7b-instruct
```
Each run writes an **inspectable bundle** to `runs/<id>/`: the serialized trace,
every frame on disk, per-call provenance, and the loss-mask spans.

---

## Honest fidelity gaps (and the upgrade path)

This is a faithful *instantiation*, not a reproduction (there's nothing to
reproduce — the paper trains nothing). Where it diverges from the ideal:

1. **Reasoner modality.** The paper's reasoner is *multimodal and trainable* and
   consumes the rollout as visual tokens. I have either trainable-but-text
   (VibeThinker) **or** multimodal-but-closed (gpt-5.4-mini). I chose
   trainable-text + gpt-5.4-mini transduction, which preserves the **training**
   story (`sft.py`/`grpo.py` apply to the local model) at the cost of routing
   perception through an external model. Drop-in upgrade: a local trainable VLM
   (Qwen2-VL / Llama-3.2-Vision) consuming frames directly — then the
   `perception` step disappears into `π_θ`.
2. **World-module.** SD1.5 frame-stepping ≠ a learned video world-model. §3.2 is
   explicit that rollout usefulness is bounded by the video model's intuitive
   physics. Upgrade: swap `SDRenderer` for a `VideoRenderer` over Wan / LTX /
   HunyuanVideo — the rest of the system only touches `Rollout.frame_paths`.
3. **No training run yet.** SFT/GRPO are implemented and unit-tested as
   objectives, not yet run on a corpus — because, per the paper's own §5, **the
   data is the bottleneck.** The missing dataset is the real frontier:
   text-only problems with verifiable answers, balanced between *needs-a-visual*
   and *doesn't* (so the model learns *when not to* imagine). `data/` holds a
   handful of SimpleBench-style seeds as a placeholder.

The standing next step is #3: this is exactly the kind of synthetic-trace corpus
an autodata-style challenger→solver→judge loop could generate, and an
AutoTrainer-style config-driven run could SFT on.

---

## The chess track — the EWM blueprint, actually trained

The SD path proves the *loop*; the chess path proves the *training*. A chess
engine is the world-module §3.1 explicitly calls a **Simulator**: deterministic,
**verifiable**, faithful by construction — which sidesteps the §3.2 worry that a
video model's rollout is only as good as its physics. And it hands you the
dataset §5 begs for, for free.

| EWM piece | chess instantiation |
|---|---|
| world-module W (Simulator) | `ChessWorldModule` — Stockfish PV / pure-python forced-mate search; rollout = forcing line as **ASCII board frames** (text → no VLM needed) |
| dataset (§5) | `chess_data.generate` — forced-mate puzzles (**needs_visual**) + factual FEN questions (**no visual**), answers verified deterministically |
| reward `r` | `verify()` — exact, move-legal RLVR signal |
| reasoner π_θ | Qwen2.5-0.5B-Instruct (LoRA, trainable) |

**Stage-1 SFT actually updates weights** (`scripts/chess_sft.py`): gold traces
are balanced 8 call / 8 no-call (Appendix A guardrail), masked-CE drops
**1.13 → 0.0015** over 6 epochs, and the model learns **selectivity** — it emits
`<tool_call>` on a mate puzzle and explicitly declines ("no visualisation is
needed") on a factual one. That selective-visualisation behaviour, learned from
the format alone, is the paper's whole point.

**Stage-2 GRPO signal** (`scripts/chess_grpo.py`): a full GRPO weight update is
impractical on a 6 GB card, but the unanswered question is whether the learning
signal is *correctly oriented*. We sample G trajectories/puzzle from the SFT'd
policy + chess world-module, score each with `r_M = r + (−λM/B)`, and measure the
group-relative advantage of *calling W* vs *not*. The signal pushes toward
calling on tactical puzzles and abstaining on factual ones — learned selective
visualisation, on real trajectories. (`runs/chess_grpo/signal.json`.)

**Stage-2 GRPO, real weight updates** (`scripts/chess_grpo_train.py`): light
SFT warm-start (so the policy still explores) → GRPO rounds that sample G
trajectories/prompt, form group-relative advantages, and take **clipped-surrogate
steps over policy tokens only** (`ρ=exp(logp_θ−logp_old)`, rollout tokens masked),
with held-out train/test eval before vs after. This is Eq. 3 actually optimized,
not just specified.

Run:
```bash
.venv-sd/bin/python -m ewm.chess_data         # generate verifiable puzzles
.venv-sd/bin/python scripts/chess_sft.py      # real LoRA SFT (free the GPU first)
.venv-sd/bin/python scripts/chess_selectivity_probe.py  # reward-orientation probe
.venv-sd/bin/python scripts/chess_grpo_train.py         # real GRPO weight updates
```

## Research: who else is building near this?

I did not find a public repository that explicitly implements **Einstein World
Models** by name yet. The paper itself is extremely fresh
([arXiv:2606.26969](https://arxiv.org/abs/2606.26969), June 25, 2026) and reads
as a blueprint/call-for-datasets rather than a released system. The useful
watchlist is therefore adjacent work that already implements one or more of the
load-bearing EWM pieces: selective imagination, external visual scratchpads,
world-model rollouts, visual-verbal traces, or verifiable simulator-backed
training.

| Project / group | Why it matters for EWM | Links |
|---|---|---|
| **EWM authors** — Munachiso Samuel Nwadike, Zangir Iklassov, Ali Mekky, Zayd M. Kawakibi Zuhri, Kentaro Inui | Direct source of the blueprint: sparse calls to a world-module, rollouts spliced into the reasoning trace, and training with observation tokens masked from loss. I found no official code at research time. | [paper](https://arxiv.org/abs/2606.26969), [HTML](https://arxiv.org/html/2606.26969v1) |
| **MindJourney** — Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, Chuang Gan | Closest implementation cousin: a VLM uses a controllable world model at test time to imagine views for spatial reasoning. It is not EWM by name, but it is very close to "reasoner + world-model rollout + answer." | [project](https://umass-embodied-agi.github.io/MindJourney/), [GitHub](https://github.com/UMass-Embodied-AGI/MindJourney), [arXiv](https://arxiv.org/abs/2507.12508) |
| **Adaptive Visual Imagination Control / AVIC** — Shoubin Yu, Yue Zhang, Zun Wang, Jaehong Yoon, Huaxiu Yao, Mingyu Ding, Mohit Bansal | Attacks the central control problem EWM has to solve: deciding *when* and *how much* to invoke visual imagination so imagined observations help instead of becoming default noise. | [project](https://adaptive-visual-tts.github.io/), [GitHub](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control), [arXiv](https://arxiv.org/abs/2602.08236) |
| **Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models** — Tsinghua / ByteDance Seed authors | Builds an interleaved visual-verbal reasoning path around multimodal world-model generations. Philosophically close, though it leans toward an integrated multimodal model rather than EWM's external tool-call loop. | [project](https://thuml.github.io/Reasoning-Visual-World/), [GitHub](https://github.com/thuml/reasoning-visual-world), [arXiv](https://arxiv.org/abs/2601.19834) |
| **Visual Sketchpad** — Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, Ranjay Krishna | Implements a concrete "draw/inspect/reason" loop for static visual artifacts. It is a static sketchpad rather than temporal rollout, but it demonstrates the same inspectable-intermediate-artifact pattern. | [project](https://visualsketchpad.github.io/), [GitHub](https://github.com/Yushi-Hu/VisualSketchpad) |
| **Visualization-of-Thought / MVoT** — Microsoft and collaborators | Visualizes reasoning traces and trains multimodal traces. This is adjacent to EWM's trace format and "visual thought" supervision, though not specifically external simulator calls. | [VoT GitHub](https://github.com/microsoft/visualization-of-thought), [MVoT GitHub](https://github.com/chengzu-li/MVoT), [MVoT arXiv](https://arxiv.org/html/2501.07542v1) |
| **ViperGPT / Whiteboard-of-Thought lineage** — Dídac Surís, Sachit Menon, Carl Vondrick, Richard Zemel and related collaborators | Earlier tool-mediated visual reasoning: LLMs call visual/code modules or draw on a whiteboard-like workspace. EWM explicitly sits in this lineage, but swaps static visual tools for world-module rollouts. | [ViperGPT GitHub](https://github.com/cvlab-columbia/viper), [EWM related-work context](https://arxiv.org/html/2606.26969v1) |
| **World-module substrate builders** — DeepMind Genie, Meta V-JEPA, WBench / WorldBench teams | These are not EWM controllers, but they build or evaluate the world-model substrate EWM would call: interactive generative worlds, video prediction, and diagnostic physics/interaction benchmarks. | [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/), [V-JEPA](https://ai.meta.com/research/vjepa/), [WBench](https://github.com/meituan-longcat/WBench), [WorldBench](https://arxiv.org/abs/2601.21282) |

**Readout.** Public work appears to be converging on the components before the
named EWM recipe: visual scratchpads, interleaved visual-verbal traces,
selective imagination controllers, and learned/simulated rollout providers. The
most natural next implementation would combine AVIC-style call gating,
MindJourney-style world-model rollouts, and this repo's EWM trace/loss masking
so the reasoner learns both to imagine and to abstain.

## Layout
```
ewm/trace.py        # tags, segments, Eq.1 append, the loss mask, parsing
ewm/inference.py    # the control loop (Eq. 1)
ewm/reasoner.py     # Ollama (local/trainable) + Mock backends
ewm/world_module.py # SDRenderer (real frames) + MockRenderer
ewm/perception.py   # gpt-5.4-mini frames -> text (the "visual tokens" step)
ewm/reward.py       # r_M (Eq. 2) + selectivity penalty (§2.4.1)
ewm/sft.py          # masked CE warm-start (Appendix A)
ewm/grpo.py         # group advantages + clipped masked surrogate (Eq. 3)
scripts/smoke.py    # offline end-to-end
scripts/run_question.py  # full real episode
tests/test_ewm.py   # 13 unit tests on the load-bearing math
```
