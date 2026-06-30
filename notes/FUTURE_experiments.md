# Borrowed ideas + hypotheses (small-model-first)

Decision (2026-06-29, Mason): **don't commit to the big models** (VOID 5B, VSS /
Cosmos-Reason2-8B / Nemotron-9B). Keep velocity on the small models we already run
(Qwen2.5-0.5B/1.5B/3B, Stockfish, the chess channel). But *borrow the ideas* and
bank hypotheses. The big-model pipeline is recorded in `VOID_evaluation.md`; this
file is what we can do **now**, cheaply, plus what to test later.

The throughline is unchanged: **measure the rollout's decision-relevant mutual
information** (`ewm/infometric.py`, `IG = log₂P(Y*|X,R) − log₂P(Y*|X)`) and feed it
back as the selectivity reward (`info_shaped_world_reward`). Every idea below is an
experiment expressed in that instrument.

---

## Borrowed → buildable now (small models)

### B1. Counterfactual-intervention rollouts  (from VOID)  — IMPLEMENTED
VOID's thesis: stop predicting forward, render the **counterfactual** ("if this
object never existed, what physics follows?"). Chess analogue, now in
`ewm/chess_world.py`: `ChessWorldModule.counterfactual_rollout(out_dir, square)`
removes a piece and re-runs the engine line; `critical_defender(fen, sol)` finds
the enemy piece whose removal turns the position into mate-in-1. This is the §3.1
*Simulator* in its sharpest form (intervene → observe), and it localises the
*causal* structure a forward forcing line leaves implicit.

> **H1.** A counterfactual rollout ("with the defender on X gone, it's mate")
> delivers **more** decision-relevant bits (higher IG) than a forward forcing-line
> rollout on defender-hinged tactics, because it exposes *why* the move works.
>
> **Test (ready, GPU):** add `--rollout-mode {forward,counterfactual}` to
> `faithfulness_mi.py`; on puzzles where `critical_defender` is not None, compare
> IG(forward) vs IG(counterfactual). Predict IG_cf > IG_fwd, larger acc lift.

### B2. Two-pass "hypothesis → refine"  (from VOID's 2-pass inference)
VOID makes a coarse counterfactual pass then a deformation-fix pass. EWM's §3.1
note that "repeated renderer calls play the role of a simulator" is the same shape:
inspect rollout 1, revise the query, call again.

> **H2.** Allowing a **second, query-revised** world-call raises cumulative IG over
> a single call — and a reasoner trained with the shaped reward learns to take the
> 2nd call only when the 1st left residual uncertainty.
>
> **Test:** `sample_trajectory(max_calls=2)`; measure marginal IG of call 2;
> check the info-shaped `r_W` keeps the 2nd call only when it pays.

### B3. Causal-targeted perception  (from Cosmos-Reason2, a *physics* VLM)
The big idea isn't the 8B model; it's that **perception should surface the
decision-relevant (causal/physical) variable**, not a generic caption. Our
`chess_perceive` currently dumps frame headers.

> **H3.** A perception read-back that names the *causal* variable ("the f7 square
> is now undefended → Qxf7#") delivers more IG than a raw frame dump, for the same
> rollout. Faithfulness is partly a *perception* problem, not only a world-module
> problem.
>
> **Test (cheap, text-only):** add a `causal` perceiver variant; A/B its IG vs the
> frame-dump perceiver on the same rollouts. No new model.

### B4. Visual tokens via the move-channel  (from RT-Embed §2.3 + chessencryption)
§2.3 wants frames "encoded into visual tokens" spliced into the trace. Our channel
*is* the discrete-token version: the `chessencryption` move-line.

> **H4.** Serialising the rollout as the move-encoded channel (constrained
> legal-move alphabet) vs natural-language read-back changes how many of the
> channel's capacity bits the reasoner extracts (util = IG/capacity). Tests whether
> the bottleneck is representation, not information.

---

## Open finding to chase (not borrowed — ours)

### F1. Non-monotonic rollout extraction across model size  — CURVE COMPLETE
Full sweep (`runs/RESULTS_size_curve.md`): util 0.23 (0.5B) → −0.01 (1.5B) →
**0.44 (3B)**; acc lift +0.26 → −0.01 → **+0.72**. So the *trend* confirms
"stronger reasoner extracts more of the channel," but it is **non-monotonic**: the
1.5B uniquely ignores the rollout. acc0 ≈ 0 for all three rules out a generic parse
failure, so the leading hypothesis is **trust, not capacity** — a mid-size instruct
model anchored to its own wrong CoT discounts the spliced observation. **Open
diagnostic (cheap, deferred behind GRPO):** dump raw 1.5B answers in both
conditions to settle trust-vs-parsing. Reframes faithfulness as *delivered MI of
the (reasoner, rollout) pair*, not a property of the rollout alone.

---

## Scaling rung: the Gemma-4 family (served, GB10)  — WIRED, ready to run

Mason's direction: continue scaling the size curve using the models in
`../gemma4-llama-dgx-spark` — e2b (4.65B dense) · e4b (7.52B dense) · 26b-A4B
(25B/~4B-active MoE, **thinking**) · 31b (30.7B dense, **thinking**), served by
llama.cpp with an OpenAI API. Wired: `faithfulness_mi.py --endpoint
http://localhost:18080/v1 --served-model gemma-4-<key>` (HTTP backend, no local
weights), orchestrated by `scripts/run_gemma_curve.sh` (serves each on **18080**,
not 8080; ath-* names; temp-guarded; runs after the GPU frees).

> **H5 (the sharp one).** The 26b/31b have built-in CoT. F1 suggests a reasoner
> anchored to its own reasoning *discounts* the spliced rollout. Prediction: the
> thinking Gemmas show a **lower** util (IG/capacity) than their parameter count
> would predict from the Qwen trend — i.e. CoT and rollout-trust trade off. If so,
> faithfulness training must teach the model to *defer to the observation* over
> its own `<think>`. Run forward AND `--rollout-mode counterfactual` to see if the
> sharper causal rollout punches through the CoT anchor.

Only `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` is on disk; pull the rest with
`../gemma4-llama-dgx-spark/scripts/download_model.sh {e2b,e4b,26b}`. The per-run
table prints directly; `aggregate_curve.py` needs a small tweak for the gemma
filenames/active-param x-axis.

## Deferred to big models (recorded, not committed) — see VOID_evaluation.md
- VOID 5B as the continuous-physics counterfactual W (the real visual-temporal payoff).
- Cosmos-Reason2-8B as physics-aware video perception read-back.
- VSS MCP tool interface as the agentic harness.
Gate: ARM/GB10 (SM_121) availability of the NIMs / HF weights; cost vs. our small-model velocity.
