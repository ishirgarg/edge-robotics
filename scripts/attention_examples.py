#!/usr/bin/env python
"""Render N example frames per dataset: the model-input base image + its per-pixel action-attention
heatmap (just the spatial overlay — no bar/layer charts). Loads each real checkpoint ONCE and loops
over frames, which is far cheaper than calling analyze() per frame (avoids reloading ~14.5 GB weights).

    source env.sh
    python scripts/attention_examples.py --gpu 1

One PNG per dataset at <out>/<config>.png: a 2 x N grid (top = model input, bottom = attention).
"""

import os
import sys

import tyro

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_CKPT_ROOT = "/scratch/ishirgarg/openpi_cache/openpi-assets/checkpoints"
# config_name -> converted torch checkpoint (see scripts/get_pi0_torch.py).
_DATASETS = ("pi05_libero", "pi05_droid", "pi0_aloha_sim")


def _examples_for(config_name, checkpoint, *, n, num_steps, episodes, device):
    """Return [(processed_base_img, per_pixel_heat, prompt, episode)] for up to n frames."""
    import numpy as np

    from edge_robotics import attention, libero_obs

    model, cfg = attention.load_real_model(checkpoint, config_name, prompt_len=None, device=device)
    eff_prompt_len = int(cfg.max_token_len)
    suffix_state = not bool(cfg.pi05)
    suffix_len = int(cfg.action_horizon) + (1 if suffix_state else 0)

    out = []
    for ep in episodes:
        if len(out) >= n:
            break
        try:
            frame = libero_obs.load_frame(config_name, episode=ep, frame=0, checkpoint=checkpoint)
            if frame.get("image_source") != "real-lerobot":
                print(f"  [{config_name}] ep{ep}: not a real frame ({frame.get('image_source')}); skipping")
                continue
            obs, layout = libero_obs.build_observation(
                frame, prompt_len=eff_prompt_len, action_dim=int(cfg.action_dim), device=device,
                discrete_state=bool(cfg.discrete_state_input))
            attn = attention.capture_action_attention(model, obs, num_steps=num_steps,
                                                      suffix_len=suffix_len, device=device)
            grid = attention._spatial_base_attention(attn, layout)  # noqa: SLF001
            base_img = libero_obs.resize_with_pad_uint8(np.asarray(frame["base_image"]))
            heat = attention._upsample_to_pixels(grid, base_img.shape[0])  # noqa: SLF001
            out.append((base_img, heat, frame["prompt"], ep))
            print(f"  [{config_name}] ep{ep}: {frame['prompt'][:60]}")
        except Exception as exc:  # noqa: BLE001 — skip an unavailable frame, keep collecting
            print(f"  [{config_name}] ep{ep} unavailable ({type(exc).__name__}: {exc}); skipping")

    del model
    import torch
    torch.cuda.empty_cache()
    return out


def _save_grid(examples, config_name, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    n = len(examples)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.8), squeeze=False)
    for j, (img, heat, prompt, ep) in enumerate(examples):
        axes[0][j].imshow(img)
        axes[0][j].set_title("\n".join(textwrap.wrap(f"ep{ep}: {prompt}", 28)), fontsize=8)
        axes[0][j].axis("off")
        axes[1][j].imshow(img)
        axes[1][j].imshow(heat, cmap="jet", alpha=0.5)
        axes[1][j].axis("off")
    axes[0][0].set_ylabel("model input", fontsize=9)
    axes[1][0].set_ylabel("attention", fontsize=9)
    fig.suptitle(f"{config_name}: action→base-camera attention (per-pixel)", fontsize=12)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{config_name}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main(gpu: int, out: str = "out/attention/examples", n: int = 5, num_steps: int = 10) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import torch

    device = torch.device("cuda")
    for config_name in _DATASETS:
        checkpoint = os.path.join(_CKPT_ROOT, f"{config_name}_torch")
        print(f"\n=== {config_name} ({checkpoint}) ===")
        # try extra episodes so we still reach n if some frames are unavailable
        examples = _examples_for(config_name, checkpoint, n=n, num_steps=num_steps,
                                 episodes=range(n + 5), device=device)
        if not examples:
            print(f"  [{config_name}] no real frames available; skipping plot")
            continue
        p = _save_grid(examples[:n], config_name, out)
        print(f"  wrote {p} ({len(examples[:n])} examples)")


if __name__ == "__main__":
    tyro.cli(main)
