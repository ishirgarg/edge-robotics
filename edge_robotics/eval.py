"""Offline action-prediction accuracy: predicted action chunk vs dataset ground-truth.

A simulator-free accuracy proxy that runs the REAL inference path: sample frames from the model's
dataset, build the real observation, run `sample_actions`, and compare the predicted action chunk to
the dataset's ground-truth chunk — both in the model's NORMALIZED action space (the flow-matching
training target; `sample_actions` returns normalized actions, the policy un-normalizes afterward).
Reports normalized RMSE / MAE over the real action dims, averaged over sampled frames × horizon.

Currently wired for LIBERO (its public LeRobot parquet carries ground-truth `actions`). DROID/ALOHA
need their real episodes on disk to score offline error (the representative attention frames have no
ground-truth). Sim success-rate (LIBERO/ALOHA) is a separate, simulator-dependent path.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np


def _load_norm_stats(checkpoint: str) -> dict:
    fs = glob.glob(os.path.join(checkpoint, "assets", "**", "norm_stats.json"), recursive=True)
    if not fs:
        raise FileNotFoundError(f"no norm_stats.json under {checkpoint}/assets (needed to normalize GT actions)")
    ns = json.load(open(fs[0]))
    return ns.get("norm_stats", ns)


def offline_action_error(*, checkpoint: str, config_name: str, gpu: int, n_frames: int = 16,
                         episode: int = 0, num_steps: int = 10, stride: int = 10) -> dict:
    """Normalized action-prediction error of the real model vs dataset ground-truth (LIBERO)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import torch

    from . import libero_obs
    from .attention import load_real_model

    dataset = libero_obs._dataset_of(config_name)  # noqa: SLF001
    if dataset != "libero":
        return {"ok": False, "config_name": config_name, "dataset": dataset,
                "error": f"offline action-error needs real episodes with GT actions; only LIBERO is wired "
                         f"(got '{dataset}'). Provide that dataset's LeRobot episodes to extend."}

    device = torch.device("cuda")
    model, cfg = load_real_model(checkpoint, config_name, prompt_len=None, device=device)
    eff_len, ah = int(cfg.max_token_len), int(cfg.action_horizon)
    stats = _load_norm_stats(checkpoint)
    use_q = bool(cfg.pi05)  # pi05 -> quantile norm, pi0 -> z-score (openpi convention)
    real_dim = len(stats["actions"]["q01" if use_q else "mean"])

    import pandas as pd
    df = pd.read_parquet(libero_obs.fetch_libero_episode(episode))

    actions_all = np.stack([np.asarray(a, dtype=np.float32) for a in df["actions"].to_list()])  # [T, real_dim]
    prompt = libero_obs._task_for_index(int(df.iloc[0]["task_index"])) or "complete the task"  # noqa: SLF001
    idxs = list(range(0, max(len(df) - ah + 1, 0), max(stride, 1)))[:n_frames]
    if not idxs:
        return {"ok": False, "error": f"episode too short ({len(df)} frames) for horizon {ah}"}

    errs = []
    with torch.inference_mode():
        for t in idxs:
            row = df.iloc[t]
            # Normalize state the same way the real openpi pipeline does before the model (a no-op for
            # pi05, which ignores obs.state; needed so a continuous-state pi0 config isn't fed OOD state).
            state = np.asarray(row["state"], dtype=np.float32)
            if "state" in stats:
                state = libero_obs._normalize_vec(state, stats["state"], use_q)  # noqa: SLF001
            frame = {"images": {"base_0_rgb": libero_obs._decode_lerobot_image(row["image"]),  # noqa: SLF001
                                "left_wrist_0_rgb": libero_obs._decode_lerobot_image(row["wrist_image"])},
                     "state": state, "prompt": prompt,
                     "dataset": "libero", "image_source": "real-lerobot"}
            obs, _ = libero_obs.build_observation(frame, prompt_len=eff_len, action_dim=int(cfg.action_dim),
                                                  device=device, discrete_state=bool(cfg.discrete_state_input))
            pred = model.sample_actions(device, obs, num_steps=num_steps)  # [1, ah, action_dim], normalized
            pred_norm = pred[0, :, :real_dim].detach().to(torch.float32).cpu().numpy()
            gt_norm = libero_obs._normalize_vec(actions_all[t:t + ah, :real_dim], stats["actions"], use_q)  # noqa: SLF001
            errs.append(pred_norm - gt_norm)

    e = np.concatenate(errs, axis=0)  # [n_frames*ah, real_dim]
    return {
        "ok": True, "config_name": config_name, "dataset": dataset, "image_source": "real-lerobot",
        "n_frames": len(idxs), "action_horizon": ah, "num_steps": num_steps, "real_action_dim": real_dim,
        "norm": "quantile" if use_q else "zscore",
        "normalized_rmse": float(np.sqrt((e ** 2).mean())),
        "normalized_mae": float(np.abs(e).mean()),
        "per_dim_rmse": np.sqrt((e ** 2).mean(axis=0)).round(4).tolist(),
    }
