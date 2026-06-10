#!/usr/bin/env python
"""Offline action-prediction accuracy (simulator-free) for a real pi0/pi05 checkpoint.

Compares the model's predicted action chunk to the dataset's ground-truth, in normalized action
space, over sampled frames. Wired for LIBERO (real LeRobot parquet with GT actions).

    source env.sh
    python scripts/eval_offline.py \
        --checkpoint /scratch/ishirgarg/openpi_cache/pi05_libero_torch \
        --config-name pi05_libero --gpu 0 --out out/eval/pi05_libero
"""

import json
import os
import sys

import tyro

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(checkpoint: str, gpu: int, config_name: str = "pi05_libero",
         out: str = "out/eval/run", n_frames: int = 16, episode: int = 0,
         num_steps: int = 10, stride: int = 10) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from edge_robotics.eval import offline_action_error

    res = offline_action_error(checkpoint=checkpoint, config_name=config_name, gpu=gpu,
                               n_frames=n_frames, episode=episode, num_steps=num_steps, stride=stride)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "offline_eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    if res.get("ok"):
        print(f"\n{config_name} ({res['dataset']}, {res['image_source']}) offline action-prediction error "
              f"(normalized, {res['norm']}, {res['n_frames']} frames × {res['action_horizon']} horizon):")
        print(f"  RMSE={res['normalized_rmse']:.4f}  MAE={res['normalized_mae']:.4f}")
        print(f"  per-dim RMSE={res['per_dim_rmse']}")
    else:
        print(f"\n[eval] not run: {res.get('error')}")
    print(f"wrote {out}/offline_eval.json")


if __name__ == "__main__":
    tyro.cli(main)
