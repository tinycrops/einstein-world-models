# VOID (Netflix) as an EWM world-module — evaluation

**Model:** `netflix/void-model` — *Video Object and Interaction Deletion*.
CogVideoX-Fun-V1.5-5B 3D transformer, fine-tuned with a **quadmask** condition
(remove / overlap / affected / keep) on paired counterfactual videos (HUMOTO +
Kubric). Two-pass inference: pass 1 = broad counterfactual hypothesis, pass 2 =
deformation fix. BF16(+FP8), 384×672, ≤197 frames, 40 GB+ VRAM. Apache-2.0.
(Netflix + INSAIT; HF model card + paper coverage, Apr 2026.)

## The one-line thesis (theirs) is an EWM thesis

VOID's framing: *"stop thinking about inpainting; start thinking about
**counterfactuals**. If this object had never existed, what would the physics of
this scene actually look like?"* That is **exactly** the EWM motivating move —
Einstein riding beside a beam of light: a counterfactual scene rendered precisely
enough to reason about. VOID is a physics-aware counterfactual *renderer*.

## Where it sits in the EWM world-module taxonomy (§3.1)

| §3.1 class | our existing W | VOID |
|---|---|---|
| Renderer (returns frames) | SD (`world_module.py`) | ✔ video, physics-aware |
| **Simulator** (intervene → observe consequence) | Stockfish (`chess_world.py`, discrete/verifiable) | ✔ **continuous, pixel-space, physical** |
| Planner | the LLM reasoner | (n/a) |

VOID is the **Simulator** interface the paper describes — "allow the reasoner to
intervene in a visualised world and observe what follows" — but for *continuous
physical* scenes, where our chess simulator is discrete and verifiable. They are
complementary: chess proved the mechanism on a channel where the answer is
exactly checkable; VOID is the real visual-temporal payoff.

It is also a **high-§3.2-quality** module by construction: the paper warns a
rollout is only as useful as the world-module's physical intuition, and VOID was
trained and selected specifically for *interaction physics* (objects fall when
their support is removed) — the exact failure mode generic video generators have.

## The catch: VOID is an editor, not a text→video generator

VOID needs **(a) an input video** and **(b) a quadmask** marking the object to
delete. It cannot visualise "from text alone," which is the purest EWM setting
(text-in problem → the model decides to visualise). So VOID slots in as a
*conditional* world-module along one of two paths:

1. **Renderer → Simulator pipeline** (the paper's own suggestion that "repeated
   renderer calls may play the role of a simulator"): a text→image/I2V front-end
   synthesises the base scene from the reasoner's `<tool_call>` query; VOID then
   performs the **counterfactual intervention** (remove object X) and returns the
   physics consequence as the `<visual_rollout>`. The intervention *is* the
   thought experiment.
2. **Scene-given settings** (the VQA/spatial setups §4 contrasts): when the
   problem already supplies a clip, VOID answers "what does object X *do* to this
   scene?" by deleting it and showing what changes — a clean causal probe.

The query the reasoner must learn to emit is richer than chess (an object
reference + the post-removal description that builds the quadmask), but that is
the same "learn *how* to query W" problem the paper already frames.

## Why this is worth doing now: our instrument transfers unchanged

The chess work built `ewm/infometric.py`. Its core metric is **modality-agnostic**:

    IG(x) = log₂ P(Y* | X, R) − log₂ P(Y* | X)      # bits the rollout delivers

Nothing in `task_information_gain` is chess-specific — it only needs a verifiable
answer and a two-condition (with/without rollout) answer distribution. So we can
measure a **VOID counterfactual rollout's faithfulness with the same code** we
used on chess. The chess channel-capacity bound (`line_channel_bits`, the
`chessencryption` term) is the one chess-specific piece; for VOID the capacity
analogue would be a pixel/latent-entropy estimate, but the *delivered MI* — the
number that actually grounds EWM — carries straight over.

**This is the bridge the project has been building toward:** chess validated the
faithfulness instrument on a lossless, exactly-verifiable channel; VOID lets the
same instrument grade a real visual-temporal physics rollout — the setting the
paper actually cares about.

## Concrete application plan

1. **`ewm/void_world.py`** — wrap VOID behind our `Rollout` interface (same shape
   as `ChessWorldModule.rollout`): `query → (frames, text-readback)`. Quadmask
   built from the object reference in the query; perception via a VLM frame
   read-back (reuse the `perception.py` seam, swap gpt-5.4-mini for a local VLM).
2. **Counterfactual-physics eval set** — small, verifiable: questions whose answer
   depends on an object's causal/physical role ("if the supporting hand is
   removed, does the cup stay or fall?"). Balanced with no-counterfactual
   controls (the selectivity test), mirroring our chess needs_visual / factual
   split. SimpleBench-style items (§5) are the north star.
3. **Run the existing info-gain experiment** (`faithfulness_mi.py` generalised):
   text-only vs VOID-rollout, measure IG and the selectivity gap. EWM predicts
   IG>0 where the counterfactual is decision-relevant, ≈0 where it is not.
4. **Feed IG back as `r_W`** — the shaped reward we just built
   (`info_shaped_world_reward`) is module-agnostic: it pays a VOID call exactly
   its measured bits. Same training loop, new world-module.

## Feasibility on spark (GB10 / house rules)

- **Memory:** 5B bf16 ≈ 10 GB weights; 197×384×672 video latents are heavy but
  122 GB unified memory clears the 40 GB requirement comfortably. Prefer **bf16**;
  the card's FP8 path may not be clean on SM_121 — don't rely on it.
- **Stack:** CogVideoX/diffusers on aarch64/CUDA-13 needs its own Docker image
  (diffusers + the FP8/CogVideoX deps), **separate** from `ath-ewm` to avoid the
  transformers/diffusers version tug-of-war (same lesson as the SD track). Build
  off `python:3.12-slim`-ARM or the base image with a pinned diffusers.
- **Cost:** a 5B two-pass video diffusion per rollout is expensive — exactly why
  the **selectivity** objective matters. This makes VOID the *ideal* stress test
  for our info-shaped `r_W`: a call must really pay its bits.
- **Prod safety:** high ports only, `ath-*` names, off Gary's network, watch
  GPU temp <75 °C, serial with our other GPU jobs.

## The other half: perception + harness (NVIDIA VSS blueprint)

`NVIDIA-AI-Blueprints/video-search-and-summarization` (VSS 3.2) is a production
**agentic video-analytics** stack. It is NOT a world-module — it *understands
existing* video (cameras, incidents, search, summarization), it does not generate
counterfactual rollouts. So it sits on the **perception + harness** side of EWM,
not the generation side. Wholesale it is the wrong altitude for us (surveillance/
warehouse ops). But three extracted pieces are exactly the EWM video plumbing:

- **`video_understanding` tool = Cosmos-Reason2-8B**, a *physical-reasoning* VLM,
  served over REST (`:8000`) and MCP. This is the ideal **perception read-back**
  (`ewm/perception.py`) for a VOID **physics** rollout — a physics-aware VLM
  reading a physics rollout. Swap it in for gpt-5.4-mini / `chess_perceive`.
- **RT-Embed (Cosmos-Embed1-448p)** = video embeddings → the literal "visual
  tokens" §2.3 wants spliced into the trace, instead of a text read-back.
- **MCP tool interface** = a ready Search-R1-style tool-calling scaffold; an EWM
  world-module registers as just another tool.

**The complete video EWM, from the pieces surfaced this week:**

    reasoner (πθ, EWM)  --<tool_call>-->  VOID  (W: counterfactual physics rollout)
            ^                               |
            |                               v
    <answer>  <--  Cosmos-Reason2-8B  (perception read-back)  <--  <visual_rollout>
            \____________  ewm/infometric.py grades delivered bits  ___________/

VOID renders the thought experiment, Cosmos-Reason2 perceives it, the EWM reasoner
decides when to call and how to use it, and our info-gain instrument measures
whether the loop delivers decision-relevant bits — the same metric, end to end.

**Feasibility caveat:** these are NIM microservices validated on x86/datacenter
GPUs; aarch64/GB10 (SM_121) NIM builds may not exist. Don't deploy the whole VSS
blueprint — cherry-pick the VLM, and if there's no ARM NIM, run Cosmos-Reason2-8B
directly via vLLM/transformers as a perception endpoint. The `skills/` are a clean
ops manual (agentskills.io spec) for standing these up — operational, not research.

## Recommendation

**Adopt VOID as the EWM intervention/Simulator world-module for the
physical-counterfactual class** — the continuous-physics complement to the chess
search-simulator. It is the cleanest available realisation of the paper's
counterfactual thought-experiment motivation, it is physics-quality by
construction (§3.2), and — decisively — **our faithfulness instrument and shaped
reward already apply to it without change.** Sequence it after the current chess
GRPO A/B closes: `void_world.py` wrapper → tiny counterfactual eval set → the
same info-gain run → IG-shaped `r_W`. Defer only on integration cost (the
diffusers/CogVideoX ARM image), not on conceptual fit — the fit is unusually good.
