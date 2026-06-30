"""The reasoner pi_theta: a trainable LLM policy that generates the trace and
decides when to query W.

The paper's reasoner is multimodal (it consumes the rollout as visual tokens).
On this cluster I split that into two real, swappable backends:

  * OllamaReasoner  -- a LOCAL, TRAINABLE text model (VibeThinker-3B / mistral).
    Faithful to the *training* story (you can actually SFT/GRPO it, sft.py /
    grpo.py), at the cost of consuming the rollout as text (perception.py
    transduces frames -> text with gpt-5.4-mini). This is the configuration the
    paper's training sections target.

  * MockReasoner   -- deterministic scripted policy for the smoke test. Proves
    the Eq. 1 control loop with zero GPU/API.

Each backend implements .step(trace_text, system) -> raw string, which the loop
parses for <tool_call>/<answer>. The reasoner emits ONE step at a time and is
told to stop after a tool_call so the loop can run W and splice the rollout
back in (matching how Search-R1-style tool loops halt at </tool_call>).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol

SYSTEM_PROMPT = """You are an Einstein World Model reasoner. You solve text-only \
problems that often require imagining how a physical scene unfolds over time.

You may think in <think>...</think>. When visualizing how a scene evolves would \
help, emit exactly one world-module call:
<tool_call>{"name": "world_module", "query": "<a vivid description of the scene to visualize>"}</tool_call>
then STOP and wait. A <visual_rollout>...</visual_rollout> describing the rendered \
frames will be appended; read it, then continue reasoning. Only call the \
world-module when a visual-temporal thought experiment genuinely helps -- it is \
expensive. Many problems need no call at all.

When ready, give the final answer in <answer>...</answer>."""


class Reasoner(Protocol):
    def step(self, trace_text: str, system: str = SYSTEM_PROMPT) -> str: ...


class OllamaReasoner:
    def __init__(self, model: str = "vibethinker-q4km:latest",
                 host: str = "http://localhost:11434", temperature: float = 0.4,
                 num_gpu: int | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        # num_gpu=0 forces CPU inference -- used to keep the GPU dedicated to the
        # SD world-module when reasoner and renderer share one card.
        self.num_gpu = num_gpu

    def step(self, trace_text: str, system: str = SYSTEM_PROMPT) -> str:
        options = {"temperature": self.temperature,
                   # halt the model right after it closes a tool call so the
                   # loop can run W before more tokens are generated.
                   "stop": ["</tool_call>", "</answer>"]}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": trace_text},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
        text = out["message"]["content"]
        # re-attach the stop tag we halted on so the parser sees closed tags
        if "<tool_call>" in text and "</tool_call>" not in text:
            text += "</tool_call>"
        elif "<answer>" in text and "</answer>" not in text:
            text += "</answer>"
        return text


class MockReasoner:
    """Scripted two-call policy for the smoke test.

    Turn 1: think + call W on the scene.
    Turn 2 (after rollout injected): think about what the frames showed + answer.
    Demonstrates 0 < M << N-1 (exactly one call) and the read-back step.
    """

    def __init__(self, answer: str = "No"):
        self.answer = answer

    def step(self, trace_text: str, system: str = SYSTEM_PROMPT) -> str:
        if "<visual_rollout>" not in trace_text:
            return (
                "<think>This depends on how the scene unfolds in time. Let me "
                "visualize it.</think>"
                '<tool_call>{"name": "world_module", "query": '
                '"the described objects moving through their full trajectory"}'
                "</tool_call>"
            )
        return (
            "<think>The rollout shows the motion completing well before the "
            "other events could finish, so the timing resolves the question."
            "</think>"
            f"<answer>{self.answer}</answer>"
        )
