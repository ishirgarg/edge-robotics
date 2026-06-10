#!/usr/bin/env python
"""Attention heatmaps: how the pi0/pi05 action expert attends to vision vs language vs proprioception.

Runs the REAL converted weights EAGER on a real (LIBERO) / representative (DROID/ALOHA) frame and
captures the action expert's softmax attention over the prefix (see edge_robotics/attention.py).
The dataset (camera layout, and whether proprioception is a separate in-prompt bucket) is derived
from config_name. This is an interpretability artifact (value-dependent), separate from the profiler.

    source env.sh
    python scripts/attention_heatmaps.py \
        --checkpoint /scratch/ishirgarg/openpi_cache/pi05_libero_torch \
        --config-name pi05_libero --gpu 6 --out out/attention/pi05_libero

Outputs <out>/attention.json + PNGs (by-modality bar, layer x modality, vs-denoise-step, spatial).
prompt_len defaults to the config's native max_token_len.
"""

import os
import sys

import tyro

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(
    checkpoint: str,
    gpu: int,
    out: str = "out/attention/run",
    config_name: str = "pi05_libero",
    num_steps: int = 10,
    episode: int = 0,
    frame: int = 0,
    prompt_len: int | None = None,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from edge_robotics.attention import analyze

    res = analyze(checkpoint=checkpoint, config_name=config_name, gpu=gpu, num_steps=num_steps,
                  episode=episode, frame_idx=frame, outdir=out, prompt_len=prompt_len)
    m = res["meta"]
    g = res["attention"]["grouped_fraction"]
    print(f"\n{config_name} ({m['dataset']}, image_source={m['image_source']}) — Task: {m['prompt']}")
    print(f"Proprioception present: {m['has_proprioception']} "
          f"(in-prompt state for discrete pi05; absent for pi05_libero)")
    print("Action→prefix attention (grouped): " +
          "  ".join(f"{k}={v:.1%}" for k, v in g.items() if v is not None))
    print(f"checksum (Σ modalities ~1.0): {res['attention']['checksum_sum_over_modalities']:.3f}")
    print(f"wrote {out}/attention.json + {len(res['plots'])} plots")


if __name__ == "__main__":
    tyro.cli(main)
