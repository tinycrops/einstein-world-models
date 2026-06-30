"""All-mock EWM smoke test: proves the Eq. 1 control loop end-to-end with no
GPU and no API. Mirrors autodata's offline smoke discipline.

Runs the mock reasoner (one world-call then answer) against the mock renderer,
writes an inspectable trace bundle, and checks the loss mask is non-trivial
(rollout observation tokens are masked, generated tokens are not).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ewm.inference import run_episode
from ewm.reasoner import MockReasoner
from ewm.world_module import MockRenderer

OUT = Path(__file__).resolve().parents[1] / "runs" / "smoke"


def main() -> int:
    bundle = run_episode(
        problem="A juggler throws a blue ball 1m up, then a purple ball 2m up, "
                "then slowly climbs a tall ladder. Where is the purple ball "
                "relative to the blue ball now?",
        reasoner=MockReasoner(answer="At roughly the same place (both landed)"),
        world=MockRenderer(),
        out_dir=OUT,
        n_frames=4,
    )
    print(f"final_answer : {bundle['final_answer']}")
    print(f"world_calls  : {bundle['n_world_calls']}  (expect 0 < M << N)")
    gen = [s for s in bundle["loss_mask_spans"] if s["generated"]]
    masked = [s for s in bundle["loss_mask_spans"] if not s["generated"]]
    print(f"mask spans   : {len(gen)} generated, {len(masked)} masked")
    roll_masked = any("visual_rollout" in s["text"] and not s["generated"]
                      for s in bundle["loss_mask_spans"])
    assert bundle["n_world_calls"] == 1, "expected exactly one world call"
    assert bundle["final_answer"], "no final answer produced"
    assert roll_masked, "rollout observation was NOT masked from the loss"
    assert any(s["generated"] for s in bundle["loss_mask_spans"]), "nothing generated"
    print(f"bundle       : {OUT/'trace.json'}")
    print("OK: control loop ran, rollout masked, answer produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
