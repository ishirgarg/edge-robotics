#!/usr/bin/env python
"""Attention heatmaps: how the pi-0.5 action expert attends to vision vs language vs proprioception.

Runs the REAL pi05_libero weights EAGER on a REAL LIBERO frame and captures the action expert's
softmax attention over the prefix (see edge_robotics/attention.py). This is an interpretability
artifact (value-dependent), separate from the latency profiler.

    source env.sh
    python scripts/attention_heatmaps.py \
        --checkpoint /scratch/ishirgarg/openpi_cache/pi05_libero_torch \
        --gpu 6 --out out/attention/pi05_libero

Outputs <out>/attention.json + PNGs (by-modality bar, layer x modality, vs-denoise-step, spatial).
"""

import os
import sys

import tyro

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(
    checkpoint: str,
    gpu: int,
    out: str = "out/attention/pi05_libero",
    config_name: str = "pi05_libero",
    num_steps: int = 10,
    episode: int = 0,
    frame: int = 0,
    prompt_len: int = 200,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from edge_robotics.attention import analyze

    res = analyze(checkpoint=checkpoint, config_name=config_name, gpu=gpu, num_steps=num_steps,
                  episode=episode, frame_idx=frame, outdir=out, prompt_len=prompt_len)
    g = res["attention"]["grouped_fraction"]
    print(f"\nTask: {res['meta']['prompt']}")
    print(f"Proprioception present: {res['meta']['has_proprioception']} "
          f"(pi05_libero omits state -> vision/language only)")
    print("Action→prefix attention (grouped): " +
          "  ".join(f"{k}={v:.1%}" for k, v in g.items() if v is not None))
    print(f"checksum (Σ modalities ~1.0): {res['attention']['checksum_sum_over_modalities']:.3f}")
    print(f"wrote {out}/attention.json + {len(res['plots'])} plots")


if __name__ == "__main__":
    tyro.cli(main)
