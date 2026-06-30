"""Unit tests for the EWM mechanics that must be exactly right: trace masking,
the reward, and the GRPO group/clip/mask bookkeeping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ewm import grpo, reward, sft
from ewm.trace import Kind, Trace, parse_generated


# ---- trace + masking ------------------------------------------------------
def test_visual_rollout_is_only_masked_kind():
    assert Kind.THINK.generated and Kind.TOOL_CALL.generated and Kind.ANSWER.generated
    assert not Kind.VISUAL_ROLLOUT.generated


def test_mask_spans_mask_problem_and_rollout_only():
    t = (Trace("Q?")
         .think("hmm").tool_call("show the scene")
         .visual_rollout("frames show the ball landing").answer("No"))
    spans = t.mask_spans()
    assert spans[0] == ("Q?", False)                       # problem given
    rollout = [s for s in spans if "visual_rollout" in s[0]][0]
    assert rollout[1] is False                              # rollout masked
    for txt, gen in spans:
        if any(tag in txt for tag in ("<think>", "<tool_call>", "<answer>")):
            assert gen is True                              # generated, in loss


class _CharTok:  # toy tokenizer: 1 token per char, for masking math
    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


def test_loss_mask_token_counts_match_spans():
    t = Trace("ab").visual_rollout("xyz").answer("k")
    ids, mask = t.loss_mask(_CharTok())
    assert len(ids) == len(mask)
    assert sum(mask) == len("<answer>k</answer>")           # only answer counts
    assert mask.count(0) == len("ab") + len("<visual_rollout>xyz</visual_rollout>")


def test_sft_build_example_ignores_masked():
    t = Trace("ab").visual_rollout("xyz").answer("k")
    ex = sft.build_example(t, _CharTok())
    assert len(ex["input_ids"]) == len(ex["labels"])
    assert ex["labels"].count(sft.IGNORE) == sum(1 for m in t.loss_mask(_CharTok())[1] if m == 0)


def test_sft_masked_ce_ignores_masked_positions():
    # huge negative logprob on a masked position must not affect the loss
    lp = [-100.0, -1.0, -1.0]
    mask = [0, 1, 1]
    assert sft.masked_ce(lp, mask) == 1.0


def test_balanced_corpus_guard():
    call = Trace("q").tool_call("x").answer("a")
    nocall = Trace("q").answer("a")
    assert sft.has_balanced_corpus([call, nocall])
    assert not sft.has_balanced_corpus([call, call])


# ---- parsing --------------------------------------------------------------
def test_parse_extracts_tool_query():
    segs = parse_generated('<think>z</think><tool_call>{"name":"world_module",'
                           '"query":"a ball falling"}</tool_call>')
    tc = [s for s in segs if s.kind is Kind.TOOL_CALL][0]
    assert tc.meta["query"] == "a ball falling"


# ---- reward ---------------------------------------------------------------
def test_answer_reward_exact_match():
    assert reward.answer_reward("Paris", "paris.") == 1.0
    assert reward.answer_reward("Lyon", "Paris") == 0.0
    assert reward.answer_reward(None, "Paris") == 0.0


def test_world_penalty_and_total():
    assert reward.world_use_penalty(0, lam=0.25, budget=2) == 0.0
    assert reward.world_use_penalty(2, lam=0.25, budget=2) == -0.25
    # a correct answer that used 2 calls still beats a wrong one with 0 calls
    assert reward.ewm_reward("Paris", "Paris", 2) > reward.ewm_reward("x", "Paris", 0)


def test_info_shaped_world_reward_rewards_paying_calls():
    # high-IG (tactical) puzzle: calling pays its bits -> positive r_W.
    hi = reward.info_shaped_world_reward(1, info_gain_bits=2.0, lam_info=0.25, lam_cost=0.10)
    assert hi > 0
    # factual (IG ~ 0): the call only costs -> net negative -> discourage calling.
    lo = reward.info_shaped_world_reward(1, info_gain_bits=0.0, lam_info=0.25, lam_cost=0.10)
    assert lo < 0
    # not calling forfeits the bonus but pays no cost.
    assert reward.info_shaped_world_reward(0, info_gain_bits=2.0) == 0.0
    # a call on a high-IG puzzle should beat a call on a factual one.
    assert hi > lo


def test_info_shaped_penalises_uninformative_rollout():
    # a rollout that REMOVES information (IG<0) makes the call doubly bad.
    bad = reward.info_shaped_world_reward(1, info_gain_bits=-1.0)
    assert bad < reward.info_shaped_world_reward(1, info_gain_bits=0.0)


def test_ewm_reward_shaped_combines_answer_and_shaped_call():
    # correct + a call that paid its bits > wrong + an unpaid call.
    good = reward.ewm_reward_shaped("Qg3", "Qg3", 1, info_gain_bits=2.0)
    bad = reward.ewm_reward_shaped("x", "Qg3", 1, info_gain_bits=0.0)
    assert good > bad


# ---- grpo -----------------------------------------------------------------
def test_group_advantages_zero_mean():
    adv = grpo.group_advantages([1.0, 0.0, 0.0, 1.0])
    assert abs(sum(adv)) < 1e-6


def test_surrogate_masks_rollout_tokens():
    # token 1 is a rollout observation (mask 0) with an absurd ratio; it must
    # be ignored, so the surrogate equals the single generated token's term.
    traj = grpo.Trajectory(advantage=1.0, ratios=[1.0, 999.0], gen_mask=[1, 0])
    assert grpo.trajectory_surrogate(traj) == 1.0


def test_surrogate_clips_positive_advantage():
    # rho=2 with A=1, clip eps 0.2 -> min(2, 1.2) = 1.2
    traj = grpo.Trajectory(advantage=1.0, ratios=[2.0], gen_mask=[1])
    assert abs(grpo.trajectory_surrogate(traj, clip_eps=0.2) - 1.2) < 1e-9


def test_grpo_objective_kl_penalty():
    g = [grpo.Trajectory(1.0, [1.0], [1], kl=2.0)]
    assert abs(grpo.grpo_objective(g, beta=0.5) - (1.0 - 0.5 * 2.0)) < 1e-9
