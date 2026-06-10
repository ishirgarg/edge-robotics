"""Attention study: how the pi-0.5 action expert attends to vision vs language vs proprioception.

This is a VALUE-dependent, interpretability artifact — NOT a latency measurement — so it runs the
model EAGER (graphs off) with the REAL trained weights on a REAL LIBERO observation. (The repo's
"never run eager for latency" rule is about timing; this never claims a latency number.)

During flow-matching denoising, the action-expert suffix tokens (action_horizon of them) attend over
the cached prefix KV (image tokens + language tokens) plus themselves. We capture those softmax
attention weights per layer per denoise step by wrapping `eager_attention_forward` (the model uses
`_attn_implementation="eager"`, so the suffix attention probabilities are computed explicitly), then
bucket the attended prefix positions by modality using the layout from libero_obs.

Proprioception handling (it enters the model differently per config):
  * pi05_libero (discrete_state_input=False): NO proprioceptive token — state is neither in the
    prompt nor a suffix token — so attention splits over VISION vs LANGUAGE only (proprioception=0).
  * discrete-state pi05 (pi05_droid/aloha/base): the tokenizer folds a discretized state span INTO
    the prompt; libero_obs recovers that token sub-range (_discrete_state_token_span) so attention to
    PROPRIOCEPTION is bucketed separately from real LANGUAGE (which becomes two intervals around it).
  * pi0: state is a continuous SUFFIX token (separable at suffix index 0) — handled when a pi0
    observation builder sets layout["proprioception"]; the pi0 suffix is action_horizon+1 tokens.
"""

from __future__ import annotations

import json
import os

import numpy as np


def load_real_model(checkpoint: str, config_name: str, *, prompt_len: int, device):
    """Build PI0Pytorch EAGER (no compile) and load the real converted weights."""
    import safetensors.torch as st
    import torch  # noqa: F401

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    from .systems.openpi_torch import _resolve_config

    # Same config derivation as the profiler backend (openpi get_config), so every pi0/pi05 x dataset
    # config works here too. prompt_len overrides max_token_len for this study.
    cfg = _resolve_config(config_name, prompt_len)
    model = PI0Pytorch(config=cfg).to(device).eval()

    path = checkpoint if checkpoint.endswith(".safetensors") else os.path.join(checkpoint, "model.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"real checkpoint not found: {path} (run scripts/get_pi05_libero_torch.py)")
    missing, _ = st.load_model(model, path, strict=False)
    real_missing = [k for k in missing if "embed_tokens" not in k and "rotary" not in k]
    if real_missing:
        raise RuntimeError(f"checkpoint load left weights uninitialized, e.g. {real_missing[:5]}")
    # eager attention so the softmax probabilities are computed explicitly (and capturable).
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
    return model, cfg


def capture_action_attention(model, obs, *, num_steps: int, suffix_len: int, device) -> np.ndarray:
    """Run sample_actions and capture the action expert's attention probs.

    Returns attn[num_steps, n_layers, heads, suffix_len, kv_len] (kv_len = prefix_len + suffix_len).
    `suffix_len` is the action-expert query length: action_horizon for pi05, action_horizon+1 for pi0
    (whose suffix prepends one continuous state token). Captured by wrapping the module-global
    eager_attention_forward; only suffix-query calls (q_len == suffix_len) are kept, excluding prefill.
    """
    import torch
    import transformers.models.gemma.modeling_gemma as mg

    captures: list[tuple[int, np.ndarray]] = []
    orig = mg.eager_attention_forward

    def patched(module, query, key, value, *args, **kwargs):
        out, w = orig(module, query, key, value, *args, **kwargs)
        if query.shape[-2] == suffix_len:  # action-expert suffix queries (not the prefill)
            li = getattr(module, "layer_idx", None)
            assert li is not None and li >= 0, f"expert attn module missing layer_idx ({li})"
            captures.append((int(li), w.detach().to(torch.float32).cpu().numpy()[0]))  # [heads,q,kv]
        return out, w

    mg.eager_attention_forward = patched
    try:
        with torch.inference_mode():
            model.sample_actions(device, obs, num_steps=num_steps)
    finally:
        mg.eager_attention_forward = orig

    if not captures:
        raise RuntimeError("captured no action-expert attention — eager path not hit?")
    n_layers = max(li for li, _ in captures) + 1
    assert len(captures) % n_layers == 0, f"{len(captures)} captures not a multiple of {n_layers} layers"
    per_step = len(captures) // n_layers
    heads, q, kv = captures[0][1].shape
    attn = np.zeros((per_step, n_layers, heads, q, kv), dtype=np.float32)
    for i, (li, w) in enumerate(captures):
        attn[i // n_layers, li] = w
    return attn


# Modalities, in display order. Each maps to a kv-column range (a (start,end) tuple OR a list of
# such tuples) via the layout. "language" excludes the discrete-state span; "proprioception" (when
# present) is that span — for discrete_state pi05 it sits INSIDE the prompt, so language is two
# intervals around it (see libero_obs.build_observation).
def _modality_ranges(layout: dict, prefix_len: int, suffix_len: int, suffix_state: bool) -> dict:
    cams = layout["cameras"]
    ranges = {
        "vision_base": cams["base"],
        "vision_left_wrist": cams["left_wrist"],
        "vision_right_wrist(masked)": cams["right_wrist"],
        "language": layout["language"],                      # tuple OR list of intervals (sans state)
        "language_pad(masked)": tuple(layout["language_pad"]),
    }
    if suffix_state:
        # pi0: the suffix prepends one continuous state token at kv index prefix_len; the action
        # tokens (and their self-attention) follow it.
        ranges["proprioception"] = (prefix_len, prefix_len + 1)
        ranges["action_self"] = (prefix_len + 1, prefix_len + suffix_len)
    else:
        ranges["action_self"] = (prefix_len, prefix_len + suffix_len)
        if layout.get("proprioception"):  # pi05 discrete: state span lives INSIDE the prompt
            ranges["proprioception"] = tuple(layout["proprioception"])
    return ranges


def bucket_attention(attn: np.ndarray, layout: dict, *, action_horizon: int, suffix_state: bool = False) -> dict:
    """Fraction of attention mass on each modality — overall, per-layer, and per-denoise-step.

    attn rows are softmax probabilities (sum to 1 over kv per query), so the modality fractions sum
    to ~1.0 — a built-in cross-check that the kv ranges tile the whole key axis. `suffix_state` (pi0)
    means the suffix's first token is the continuous proprioceptive state (bucketed as proprioception).
    """
    n_steps, n_layers, heads, q, kv = attn.shape  # q = suffix_len (action_horizon [+1 for pi0])
    prefix_len = layout["prefix_len"]
    ranges = _modality_ranges(layout, prefix_len, q, suffix_state)

    def mass(a, rng):  # sum prob mass over a kv range (tuple) or list of ranges, on a [...,kv] array
        intervals = rng if isinstance(rng, list) else [rng]
        out = None
        for lo, hi in intervals:
            m = a[..., lo:hi].sum(axis=-1)
            out = m if out is None else out + m
        return out

    overall, per_layer, per_step = {}, {}, {}
    for name, rng in ranges.items():
        overall[name] = float(mass(attn, rng).mean())
        per_layer[name] = mass(attn, rng).mean(axis=(0, 2, 3)).tolist()        # over steps,heads,q
        per_step[name] = mass(attn, rng).mean(axis=(1, 2, 3)).tolist()         # over layers,heads,q
    # Modality groups: vision (all cams) vs language vs proprioception vs the suffix self-attention.
    grouped = {
        "vision": sum(overall[k] for k in overall if k.startswith("vision")),
        "language": overall["language"] + overall["language_pad(masked)"],
        "action_self": overall["action_self"],
        # 0.0 when there is no state token (pi05_libero); the real fraction when state is in-prompt.
        "proprioception": overall.get("proprioception", 0.0),
    }
    return {
        "overall_fraction": overall,
        "grouped_fraction": grouped,
        "per_layer_fraction": per_layer,
        "per_step_fraction": per_step,
        "shape": {"n_steps": n_steps, "n_layers": n_layers, "heads": heads,
                  "action_horizon": q, "kv_len": kv},
        "checksum_sum_over_modalities": float(sum(overall.values())),  # ~1.0 expected
    }


def _spatial_base_attention(attn: np.ndarray, layout: dict) -> np.ndarray:
    """Per-patch base-camera attention as a 16x16 grid (avg over steps,layers,heads,queries)."""
    lo, hi = layout["cameras"]["base"]
    side = int(round((hi - lo) ** 0.5))  # 256 -> 16
    grid = attn[..., lo:hi].mean(axis=(0, 1, 2, 3))  # [256]
    return grid.reshape(side, side)


def make_plots(attn: np.ndarray, buckets: dict, layout: dict, frame: dict, outdir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    paths = []
    prompt = layout.get("prompt", "")

    # 1. Overall attention mass per modality (bar).
    ov = buckets["overall_fraction"]
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(ov)
    ax.bar(range(len(names)), [ov[k] for k in names], color="#4C72B0")
    ax.set_ylabel("attention fraction"); ax.set_title(f"Action→prefix attention by modality\n{prompt[:70]}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); p = os.path.join(outdir, "attn_by_modality.png"); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 2. Layer x modality heatmap.
    pl = buckets["per_layer_fraction"]
    mods = list(pl)
    mat = np.array([pl[m] for m in mods])  # [modalities, layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods, fontsize=8)
    ax.set_xlabel("action-expert layer"); ax.set_title("Attention fraction by layer x modality")
    fig.colorbar(im, ax=ax, fraction=0.025); fig.tight_layout()
    p = os.path.join(outdir, "attn_layer_x_modality.png"); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 3. Per-denoise-step trend: vision vs language vs self.
    ps = buckets["per_step_fraction"]
    vis = np.sum([ps[k] for k in ps if k.startswith("vision")], axis=0)
    lang = np.sum([ps[k] for k in ("language", "language_pad(masked)")], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = range(len(vis))
    ax.plot(steps, vis, "-o", label="vision"); ax.plot(steps, lang, "-o", label="language")
    ax.plot(steps, ps["action_self"], "-o", label="action self")
    if "proprioception" in ps:
        ax.plot(steps, ps["proprioception"], "-o", label="proprioception")
    ax.set_xlabel("denoise step"); ax.set_ylabel("attention fraction"); ax.legend()
    ax.set_title("Attention vs denoise step"); fig.tight_layout()
    p = os.path.join(outdir, "attn_vs_denoise_step.png"); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 4. Spatial: where on the base camera do the actions look? (16x16 patch attention over the image)
    grid = _spatial_base_attention(attn, layout)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(frame["base_image"]); axes[0].set_title("base camera"); axes[0].axis("off")
    axes[1].imshow(frame["base_image"], extent=(0, grid.shape[1], grid.shape[0], 0))
    axes[1].imshow(grid, cmap="hot", alpha=0.55, extent=(0, grid.shape[1], grid.shape[0], 0))
    axes[1].set_title("action→base-camera attention (16x16 patches)"); axes[1].axis("off")
    fig.suptitle(prompt[:80]); fig.tight_layout()
    p = os.path.join(outdir, "attn_spatial_base.png"); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


def analyze(*, checkpoint: str, config_name: str, gpu: int, num_steps: int, episode: int, frame_idx: int,
            outdir: str, prompt_len: int | None = None) -> dict:
    """End to end: real model + a real/representative frame -> attention buckets + plots + JSON.

    prompt_len=None uses the config's native max_token_len ('as trained'). The dataset (and thus the
    camera layout + whether proprioception is a separate bucket) is derived from config_name."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import torch

    from . import libero_obs

    device = torch.device("cuda")
    model, cfg = load_real_model(checkpoint, config_name, prompt_len=prompt_len, device=device)
    eff_prompt_len = int(cfg.max_token_len)  # the effective length (native unless prompt_len overrode it)
    frame = libero_obs.load_frame(config_name, episode, frame_idx)
    obs, layout = libero_obs.build_observation(
        frame, prompt_len=eff_prompt_len, action_dim=int(cfg.action_dim), device=device,
        discrete_state=bool(cfg.discrete_state_input))
    suffix_state = not bool(cfg.pi05)  # pi0 prepends a continuous proprioceptive state token to the suffix
    suffix_len = int(cfg.action_horizon) + (1 if suffix_state else 0)
    attn = capture_action_attention(model, obs, num_steps=num_steps, suffix_len=suffix_len, device=device)
    buckets = bucket_attention(attn, layout, action_horizon=int(cfg.action_horizon), suffix_state=suffix_state)
    plots = make_plots(attn, buckets, layout, frame, outdir)

    result = {
        "meta": {"checkpoint": checkpoint, "config_name": config_name, "dataset": frame.get("dataset"),
                 "image_source": frame.get("image_source"), "prompt": frame["prompt"],
                 "episode": episode, "frame": frame_idx, "num_steps": num_steps, "prompt_len": eff_prompt_len,
                 "discrete_state_input": bool(cfg.discrete_state_input),
                 "has_proprioception": layout["has_proprioception"], "layout": layout},
        "attention": buckets, "plots": plots,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "attention.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
