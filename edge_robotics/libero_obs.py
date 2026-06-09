"""Build a REAL pi-0.5 observation from a real LIBERO frame.

The latency profiler is value-independent and feeds zeros. The attention study is NOT — it needs the
real trained weights AND a real observation (real camera image + real task instruction) for the
attention pattern to mean anything. This module fetches one frame from the public LIBERO LeRobot
dataset (`physical-intelligence/libero`, the same data openpi trains pi05_libero on) and assembles
the model `Observation`, plus the exact prefix token layout the attention study buckets over.

Token layout it returns (for pi05_libero, 3 image slots x 256 SigLIP tokens + padded language):
  [   0: 256) base camera        (attended)
  [ 256: 512) left wrist camera  (attended)
  [ 512: 768) right wrist        (zero-filled + MASKED — pi0 family masks the absent 3rd cam)
  [ 768: 768+n_real) language    (the real task tokens; attended)
  [768+n_real : 768+max_token_len) language PAD (masked)
pi05_libero sets discrete_state_input=False, so proprioceptive state is NOT a prefix token and the
pi05 action expert's suffix carries no state token either — there is no "proprioception" modality to
attend to for this checkpoint (a finding, surfaced by the study).
"""

from __future__ import annotations

import io
import os
import pathlib

import numpy as np

_HF_REPO = "physical-intelligence/libero"
_TOKENS_PER_IMAGE = 256
_IMG_PX = 224


def fetch_libero_episode(episode: int = 0, *, dest_dir: str | None = None) -> str:
    """Download one LIBERO episode parquet (LeRobot format) and return the local path."""
    dest_dir = dest_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "libero_episodes")
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest = os.path.join(dest_dir, f"episode_{episode:06d}.parquet")
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        # curl is markedly more reliable than hf_hub_download for these large LFS parquet files.
        import subprocess
        url = (f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/"
               f"data/chunk-000/episode_{episode:06d}.parquet")
        r = subprocess.run(["curl", "-sL", "--max-time", "600", "-o", dest, url],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise RuntimeError(f"failed to download {url}: {r.stderr.strip()}")
    return dest


def _task_for_index(task_index: int) -> str | None:
    """Resolve the natural-language task string for a task_index from the dataset metadata."""
    try:
        import json

        from huggingface_hub import hf_hub_download
        p = hf_hub_download(_HF_REPO, "meta/tasks.jsonl", repo_type="dataset")
        for line in open(p):
            rec = json.loads(line)
            if int(rec.get("task_index", -1)) == int(task_index):
                return rec.get("task")
    except Exception:  # noqa: BLE001  — fall back to a generic prompt if metadata is unreachable
        return None
    return None


def load_libero_frame(episode: int = 0, frame: int = 0, *, parquet_path: str | None = None) -> dict:
    """Return one real frame: {base_image[H,W,3] uint8, wrist_image, state[8], prompt, episode, frame}."""
    import pandas as pd
    from PIL import Image

    path = parquet_path or fetch_libero_episode(episode)
    df = pd.read_parquet(path)
    row = df.iloc[min(frame, len(df) - 1)]

    def _img(v) -> np.ndarray:
        if isinstance(v, dict):  # LeRobot stores images as {"bytes": <png>, "path": ...}
            v = v["bytes"]
        if isinstance(v, (bytes, bytearray)):
            return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"))
        return np.asarray(v)

    prompt = _task_for_index(int(row["task_index"])) or "complete the task"
    return {
        "base_image": _img(row["image"]),
        "wrist_image": _img(row["wrist_image"]),
        "state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt,
        "episode": int(episode),
        "frame": int(frame),
    }


def build_observation(frame: dict, *, prompt_len: int, action_dim: int, device, discrete_state: bool):
    """Assemble the model `Observation` + the prefix token layout the attention study buckets over.

    Images -> float32 [-1,1] NCHW (the model's expected range; uint8/255*2-1). Prompt -> PaliGemma
    tokens (pi05 with discrete_state=True folds a discretized state span into the prompt; pi05_libero
    uses discrete_state=False, so state is absent and the prompt is just the task)."""
    import torch
    from PIL import Image

    from openpi.models import model as _model
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import array_typing as at

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        # Pre-resize to 224 in [-1,1] NCHW. LIBERO frames are square, so a plain resize == the model's
        # resize_with_pad (no padding); we resize here because the model's internal resize path is only
        # exercised for non-224 inputs and doesn't preserve the batch dim — openpi's real pipeline
        # likewise resizes to 224 BEFORE the model, so the model sees a no-op resize either way.
        im = Image.fromarray(img).resize((_IMG_PX, _IMG_PX), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0   # -> [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)  # [1,3,H,W]

    base = _to_tensor(frame["base_image"])
    wrist = _to_tensor(frame["wrist_image"])
    right = torch.zeros_like(base)  # absent 3rd camera, masked below
    images = {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": right}
    image_masks = {"base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
                   "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
                   "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device)}

    tok = PaligemmaTokenizer(max_len=prompt_len)
    state8 = frame["state"]
    tokens, tok_mask = tok.tokenize(frame["prompt"], state8 if discrete_state else None)
    n_real = int(np.asarray(tok_mask).sum())

    state = torch.zeros(1, action_dim, dtype=torch.float32, device=device)
    state[0, : min(action_dim, state8.shape[0])] = torch.from_numpy(state8[:action_dim]).to(device)
    tokenized_prompt = torch.from_numpy(np.asarray(tokens))[None].to(torch.int64).to(device)
    tokenized_prompt_mask = torch.from_numpy(np.asarray(tok_mask))[None].to(torch.bool).to(device)

    with at.disable_typechecking():
        obs = _model.Observation(
            images=images, image_masks=image_masks, state=state,
            tokenized_prompt=tokenized_prompt, tokenized_prompt_mask=tokenized_prompt_mask,
        )

    n_img = len(images)
    img_tokens = n_img * _TOKENS_PER_IMAGE
    layout = {
        "prefix_len": img_tokens + prompt_len,
        "cameras": {"base": (0, 256), "left_wrist": (256, 512), "right_wrist": (512, 768)},
        "masked_cameras": ["right_wrist"],
        "language": (img_tokens, img_tokens + n_real),       # real task tokens
        "language_pad": (img_tokens + n_real, img_tokens + prompt_len),
        "vision_range": (0, img_tokens),
        "n_real_language_tokens": n_real,
        "tokens_per_image": _TOKENS_PER_IMAGE,
        "prompt": frame["prompt"],
        "has_proprioception": bool(discrete_state),  # False for pi05_libero — no state token
    }
    return obs, layout
