"""Build a pi0/pi05 observation (+ prefix token layout) for the attention study, per dataset.

The latency profiler is value-independent and feeds zeros. The attention study is NOT — it needs the
real trained weights AND a real/representative observation for the attention pattern to mean anything.
`load_frame(config_name, ...)` returns a frame for the config's dataset: LIBERO uses a REAL frame from
the public LeRobot dataset (`physical-intelligence/libero`); DROID/ALOHA use a representative frame
(real task prompt + the dataset's true camera layout + state dim; synthetic-random pixels, flagged via
image_source) since their training data isn't a single public parquet. `build_observation` then
assembles the model `Observation` + the prefix token layout the study buckets over.

Token layout (3 image slots x 256 SigLIP tokens + padded language):
  [   0: 256) base camera; [256:512) left wrist; [512:768) right wrist  — a slot is REAL when its
  image is populated, else zero-filled + MASKED (LIBERO/DROID mask the 3rd cam; ALOHA uses all three).
  [768 : 768+n_real) language; [768+n_real : 768+max_token_len) PAD (masked).
Proprioception: pi05 with discrete_state_input folds discretized state tokens INTO the prompt — they
are recovered and bucketed SEPARATELY from language (see _discrete_state_token_span). pi05_libero sets
discrete_state_input=False -> no state token at all (proprioception absent). pi0 instead carries state
as a continuous action-expert SUFFIX token (handled separately).
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
    base, wristimg = _img(row["image"]), _img(row["wrist_image"])
    return {
        # base + left wrist are real; the 3rd slot is absent (zero-filled + masked in build_observation).
        "images": {"base_0_rgb": base, "left_wrist_0_rgb": wristimg},
        "base_image": base,  # kept for the spatial-attention overlay plot
        "state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt, "dataset": "libero", "image_source": "real-lerobot",
        "episode": int(episode), "frame": int(frame),
    }


# Per-dataset observation spec (mirrors openpi's *Inputs transforms). The pi0 family always feeds 3
# fixed SigLIP slots; a slot is REAL when populated, else zero-filled + masked. LIBERO/DROID use
# base + one wrist (3rd slot masked); ALOHA uses all three (high + both wrists, none masked).
_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_SHORT = {"base_0_rgb": "base", "left_wrist_0_rgb": "left_wrist", "right_wrist_0_rgb": "right_wrist"}
_DATASET_SPECS = {
    "libero": {"real_slots": ("base_0_rgb", "left_wrist_0_rgb"), "state_dim": 8,
               "prompt": "pick up the object and place it on the plate"},
    "droid":  {"real_slots": ("base_0_rgb", "left_wrist_0_rgb"), "state_dim": 8,
               "prompt": "pick up the object and put it in the bin"},
    "aloha":  {"real_slots": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), "state_dim": 14,
               "prompt": "fold the towel and place it on the right"},
}


def _dataset_of(config_name: str) -> str:
    for d in ("libero", "droid", "aloha"):
        if d in config_name:
            return d
    return "libero"


def load_frame(config_name: str, episode: int = 0, frame: int = 0) -> dict:
    """Representative observation frame for the config's dataset, for the attention study.

    LIBERO uses a REAL frame from its public LeRobot dataset. DROID/ALOHA use a representative frame
    (real task prompt + the dataset's true camera layout + state dim; synthetic-random pixels —
    flagged via `image_source`) since their training data isn't a single public parquet. The
    attention MODALITY split, proprioception separation, and camera/state layout are faithful
    regardless of dataset; only the spatial pixel content differs for DROID/ALOHA."""
    dataset = _dataset_of(config_name)
    if dataset == "libero":
        return load_libero_frame(episode, frame)
    spec = _DATASET_SPECS[dataset]
    rng = np.random.default_rng(episode)
    images = {s: rng.integers(0, 256, size=(_IMG_PX, _IMG_PX, 3), dtype=np.uint8) for s in spec["real_slots"]}
    state = rng.uniform(-1.0, 1.0, size=spec["state_dim"]).astype(np.float32)
    return {"images": images, "base_image": images["base_0_rgb"], "state": state,
            "prompt": spec["prompt"], "dataset": dataset, "image_source": "synthetic-random",
            "episode": int(episode), "frame": int(frame)}


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

    # 3 fixed slots: a slot present in frame["images"] is REAL (mask True); an absent slot is
    # zero-filled + masked False (LIBERO/DROID mask the 3rd cam; ALOHA populates all three).
    frame_images = frame["images"]
    images, image_masks, masked_cameras = {}, {}, []
    for slot in _SLOTS:
        if slot in frame_images:
            images[slot] = _to_tensor(frame_images[slot])
            image_masks[slot] = torch.ones(1, dtype=torch.bool, device=device)
        else:
            images[slot] = torch.zeros_like(next(iter(images.values())))  # base_0_rgb is always real & first
            image_masks[slot] = torch.zeros(1, dtype=torch.bool, device=device)
            masked_cameras.append(_SHORT[slot])

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
    lang_lo, lang_hi = img_tokens, img_tokens + n_real

    # Proprioception: for discrete_state pi05 the tokenizer folds the discretized state digits INTO
    # the prompt (`Task: {text}, State: {digits};\nAction:`), so the state occupies a token sub-range
    # in the MIDDLE of the language block — recover it and split it OUT of "language" so the attention
    # study reports attention to proprioception vs real language separately. For pi05_libero
    # (discrete_state=False) there is no state token at all -> proprioception absent.
    proprioception = None
    language = [(lang_lo, lang_hi)]
    if discrete_state:
        s0, s1 = _discrete_state_token_span(tok, frame["prompt"], state8, n_real)
        proprioception = (lang_lo + s0, lang_lo + s1)
        language = [(lang_lo, lang_lo + s0), (lang_lo + s1, lang_hi)]  # task text + template, sans state

    layout = {
        "prefix_len": img_tokens + prompt_len,
        "cameras": {"base": (0, 256), "left_wrist": (256, 512), "right_wrist": (512, 768)},
        "masked_cameras": masked_cameras,                    # absent slots (LIBERO/DROID: right_wrist; ALOHA: none)
        "language": language,                                # list of kv-intervals (excludes state)
        "language_pad": (lang_hi, img_tokens + prompt_len),
        "proprioception": proprioception,                    # (lo,hi) kv-range of state digits, or None
        "vision_range": (0, img_tokens),
        "n_real_language_tokens": n_real,
        "tokens_per_image": _TOKENS_PER_IMAGE,
        "prompt": frame["prompt"],
        "dataset": frame.get("dataset"),
        "image_source": frame.get("image_source"),
        "has_proprioception": proprioception is not None,
    }
    return obs, layout


def _discrete_state_token_span(tok, prompt: str, state: np.ndarray, n_real: int) -> tuple[int, int]:
    """Token sub-range [start, end) of the discretized state digits within the language block.

    Replicates PaligemmaTokenizer.tokenize's discrete-state format (openpi tokenizer.py:23-29), then
    recovers the span by CHARACTER OFFSETS (SentencePiece piece .begin/.end are char offsets in the
    normalized text — verified empirically), not length-differencing. Length-differencing is off-by-one
    when the first state digit is "-1" (state[0] < -1): SentencePiece merges the trailing space of
    "State: " with the "-" into one `▁-` piece, so the standalone space piece that `encode(head)`
    counts vanishes in the full encoding. Offset-overlap instead assigns the merged piece to the state
    (its span crosses the state boundary), which is correct. Clamped to n_real for truncation."""
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    disc = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, disc))
    full = f"Task: {cleaned}, State: {state_str};\nAction: "
    head = f"Task: {cleaned}, State: "
    c0 = len(head)                                      # char index where the state digits begin
    c1 = c0 + len(state_str)                            # char index where they end (before ";")
    sp = tok._tokenizer  # noqa: SLF001 — the SentencePiece processor PaligemmaTokenizer wraps
    pieces = sp.EncodeAsImmutableProto(full).pieces     # each piece carries char .begin/.end (no BOS)
    start = next((i for i, p in enumerate(pieces) if p.end > c0), len(pieces))
    end = next((i for i, p in enumerate(pieces) if p.begin >= c1), len(pieces))
    start, end = start + 1, end + 1                     # tokens are encoded with add_bos=True (+1)
    return min(start, n_real), min(end, n_real)         # clamp into the real (non-pad) token range
