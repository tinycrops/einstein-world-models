"""World-module W (§3): produces short visual-temporal rollouts.

Paper taxonomy (§3.1) distinguishes Renderers / Simulators / Planners and
states that **renderers are the default world-module** for EWMs, and that
"repeated renderer calls may play the role of a simulator." So we implement a
Renderer.

A faithful renderer would be a text-to-video / image-to-video diffusion model
(Wan, HunyuanVideo, LTX -- cited in §3.1). My GPU node has SD1.5 (dreamshaper_8)
via diffusers but no video model. So the v1 renderer approximates a rollout as
a *prompt-stepped* sequence of frames: the reasoner's query is expanded into a
few temporal beats ("...at t=0", "...mid-motion", "...settled") and each beat is
rendered. For trajectory questions (the SimpleBench juggler/ball case) this
captures exactly the reasoning-relevant content -- how the scene evolves in
time -- which is the whole point of a thought experiment.

W returns frames (the inspectable artifact, §2.2). It does NOT describe them;
turning frames into tokens the reasoner can consume is perception.py's job
(the paper's "encode into visual tokens"). This keeps W a pure renderer.

UPGRADE PATH: swap SDRenderer for a VideoRenderer wrapping Wan/LTX and the rest
of the system is unchanged -- it only consumes `Rollout.frame_paths`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Rollout:
    """What W(q_t) returns: a short frame sequence + provenance for inspection."""

    query: str
    frame_paths: list[str]
    beats: list[str] = field(default_factory=list)  # the per-frame temporal prompts
    seconds: float = 0.0
    module: str = "unknown"
    text: str = ""  # optional module-supplied textual summary (e.g. chess line)


class WorldModule(Protocol):
    def rollout(self, query: str, out_dir: Path, n_frames: int = 4) -> Rollout: ...


# --------------------------------------------------------------------------
# Mock renderer: deterministic, no GPU. Lets the whole EWM loop run as a smoke
# test (the autodata discipline -- prove the control flow before the hardware).
# --------------------------------------------------------------------------
class MockRenderer:
    module = "mock"

    def rollout(self, query: str, out_dir: Path, n_frames: int = 4) -> Rollout:
        out_dir.mkdir(parents=True, exist_ok=True)
        beats = _temporal_beats(query, n_frames)
        paths = []
        for i, beat in enumerate(beats):
            p = out_dir / f"frame_{i:02d}.txt"
            p.write_text(f"[mock frame {i}] {beat}\n")
            paths.append(str(p))
        return Rollout(query=query, frame_paths=paths, beats=beats,
                       seconds=0.0, module=self.module)


# --------------------------------------------------------------------------
# SD1.5 renderer: real frames via diffusers on the GTX 1060.
# Pinned combo per cluster CLAUDE.md: diffusers 0.36.0 / transformers <5 /
# torch 2.4.x cu1xx, fp16, safety_checker off, attention slicing.
# --------------------------------------------------------------------------
class SDRenderer:
    module = "sd15-dreamshaper8"

    def __init__(self, checkpoint: str | None = None, steps: int = 30,
                 size: int = 512, img2img_strength: float = 0.55):
        self.checkpoint = checkpoint or str(
            Path.home() / "ComfyUI/models/checkpoints/dreamshaper_8.safetensors"
        )
        self.steps = steps
        self.size = size
        self.img2img_strength = img2img_strength
        self._txt2img = None
        self._img2img = None

    def _load(self):
        if self._txt2img is not None:
            return
        import torch
        from diffusers import (StableDiffusionImg2ImgPipeline,
                               StableDiffusionPipeline)

        common = dict(torch_dtype=torch.float16, safety_checker=None)
        self._txt2img = StableDiffusionPipeline.from_single_file(
            self.checkpoint, **common).to("cuda")
        self._txt2img.enable_attention_slicing()
        # Share the loaded components for img2img (chained temporal frames).
        self._img2img = StableDiffusionImg2ImgPipeline(**self._txt2img.components)
        self._img2img.enable_attention_slicing()

    def rollout(self, query: str, out_dir: Path, n_frames: int = 4) -> Rollout:
        self._load()
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        beats = _temporal_beats(query, n_frames)
        t0 = time.time()
        paths: list[str] = []
        prev = None
        gen = torch.Generator("cuda").manual_seed(0)
        for i, beat in enumerate(beats):
            if prev is None:
                img = self._txt2img(
                    beat, num_inference_steps=self.steps,
                    height=self.size, width=self.size, generator=gen
                ).images[0]
            else:
                # img2img-chain off the previous frame -> temporal continuity,
                # i.e. "repeated renderer calls play the role of a simulator".
                img = self._img2img(
                    prompt=beat, image=prev,
                    strength=self.img2img_strength,
                    num_inference_steps=self.steps, generator=gen
                ).images[0]
            p = out_dir / f"frame_{i:02d}.png"
            img.save(p)
            paths.append(str(p))
            prev = img
        return Rollout(query=query, frame_paths=paths, beats=beats,
                       seconds=round(time.time() - t0, 1), module=self.module)


def _temporal_beats(query: str, n: int) -> list[str]:
    """Expand a single scene query into n temporal beats.

    This is the renderer's stand-in for genuine video dynamics: instead of a
    learned dynamics model unrolling the scene, we ask for the same scene at n
    ordered moments. Crude but it makes the *temporal* structure explicit,
    which is what the thought experiment needs.
    """
    if n <= 1:
        return [query]
    phases = ["at the initial instant", "shortly after, mid-motion",
              "later, as motion continues", "at the final settled moment"]
    # stretch/truncate phase labels to exactly n beats
    out = []
    for i in range(n):
        phase = phases[min(int(i / max(1, n - 1) * (len(phases) - 1)), len(phases) - 1)]
        out.append(f"{query}, {phase}, photographic, clear")
    return out
