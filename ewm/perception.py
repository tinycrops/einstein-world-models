"""Perception: transduce returned rollout frames into tokens the reasoner can
read -- the paper's "encode into visual tokens" step (§2.3).

The paper's multimodal reasoner does this internally. My trainable reasoner is
text-only (VibeThinker/mistral), so a vision model reads the frames and emits a
compact, *literal* description that goes inside <visual_rollout>. Per the user's
direction this vision model is gpt-5.4-mini.

Design rule, straight from §2.2 / §3.2: perception reports what is VISIBLE, not
what the answer should be. The rollout is a *hypothesis to inspect*, and its
usefulness depends on faithfulness -- the description must expose what the
frames actually show (including implausibilities) so the reasoner can judge it,
rather than smuggling in a conclusion.

For mock .txt frames this just concatenates them (keeps the smoke test offline).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

VISION_MODEL = os.environ.get("EWM_VISION_MODEL", "gpt-5.4-mini")

_PERCEPTION_PROMPT = (
    "These ordered frames are a short visual-temporal rollout of an imagined "
    "scene (a thought experiment). Describe ONLY what is literally visible and "
    "how it changes from frame to frame -- positions, motion, contact, what has "
    "happened by the last frame. Be concrete about timing and spatial relations. "
    "Do NOT answer any question or state conclusions; just report the rollout so "
    "a reasoner can inspect it. 3-5 sentences."
)


def describe_rollout(frame_paths: list[str], query: str) -> str:
    paths = [Path(p) for p in frame_paths]
    if paths and paths[0].suffix == ".txt":  # mock frames -> offline passthrough
        body = " ".join(p.read_text().strip() for p in paths)
        return f"(mock perception) rendered for '{query}': {body}"
    return _gpt_vision(paths, query)


def _gpt_vision(paths: list[Path], query: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    content = [{"type": "text",
                "text": f"{_PERCEPTION_PROMPT}\n\nThe scene queried was: {query}"}]
    for p in paths:
        b64 = base64.b64encode(p.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content.strip()
