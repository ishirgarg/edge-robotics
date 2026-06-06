"""pi-0.5 on the realtime-vla (PyTorch + Triton) backend.

Profiles dexmal/realtime-vla's `Pi05Inference` — a fused-Triton reimplementation of pi-0.5 that
runs the steady-state forward as a captured CUDA graph (`infer_graph.replay()`). We load it exactly
the way the repo's own `benchmark.py` does and never reimplement the forward math; the only
instrumentation mirrors `pi05_jax.apply_namescope_patch`:

  * E2E  -> the real fast path: `Pi05Inference.forward()` (CUDA-graph replay). Timed wall, device-
            synced with `torch.cuda.synchronize`.
  * Vision / VLM / Action -> we wrap realtime-vla's three public eager stage functions
            (`vision_encoder` / `transformer_encoder` / `transformer_decoder`, the globals
            `pi05_model` dispatches to) with `torch.cuda.Event` timing and run the repo's own eager
            `record_run()`. This is graphs-off (the graph replay is opaque), the exact analogue of
            why the JAX backend disables CUDA graphs for attribution. Percentages are sound;
            absolute per-stage ms are the eager numbers (the per-row note surfaces eager-sum vs the
            graph E2E = the graph speedup).

The realtime-vla repo is vendored in-tree at <repo>/realtime-vla (cloned by install.sh) and
imported via sys.path — it ships as copy-in files, not a pip package. Override the location with
REALTIME_VLA_DIR if needed.

CAVEATS (see README): diffusion steps are hard-locked to 10 in realtime-vla's forward (time embeds
are precomputed for 10), so --num-steps != 10 is ignored (warned). The Triton kernels are tuned for
RTX 4090/5090; they run on other archs but won't hit the advertised latencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from .base import LoadedSystem, ProfiledSystem

# Mirrors the pi05_* entries in pi05_jax.PI05_REGISTRY, mapped to realtime-vla's knobs.
# chunk_size == action_horizon; num_views == number of camera images. The Gemma sizes are baked
# into the Triton kernels (gemma_2b prefix + gemma_300m action expert), so they aren't selectable.
RT_REGISTRY: dict[str, dict] = {
    "pi05_base": dict(num_views=3, chunk_size=50, discrete_state_input=True),
    "pi05_aloha": dict(num_views=3, chunk_size=50, discrete_state_input=True),
    "pi05_droid": dict(num_views=3, chunk_size=15, discrete_state_input=True),
    "pi05_libero": dict(num_views=3, chunk_size=10, discrete_state_input=False),
}

_TOKENS_PER_IMAGE = 256
_FIXED_NUM_STEPS = 10  # realtime-vla pi05 forward is hard-locked to 10 flow steps.

# realtime-vla's three eager stage entrypoints -> our phase scopes. These are module globals in
# `pi05_infer` that `pi05_model` (run by `record_run`) dispatches to, so patching the module
# attribute is picked up without touching the forward math (cf. pi05_jax.apply_namescope_patch).
_STAGE_SCOPE = {
    "vision_encoder": "vision",
    "transformer_encoder": "vlm",
    "transformer_decoder": "action",
}

# Shared timing state for the monkeypatched wrappers. `on` gates collection so the graph-capture
# record_run()s in Pi05Inference.__init__ aren't timed; events accumulate across the timed loop.
_timing: dict = {"on": False, "events": {"vision": [], "vlm": [], "action": []}}


def _realtime_vla_dir() -> Path:
    """Vendored realtime-vla checkout (repo_root/realtime-vla), overridable via REALTIME_VLA_DIR."""
    env = os.environ.get("REALTIME_VLA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # this file: <repo>/edge_robotics/systems/pi05_realtimevla.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2] / "realtime-vla"


def _import_pi05_infer():
    """Put the vendored realtime-vla dir on sys.path and import its `pi05_infer` module."""
    d = _realtime_vla_dir()
    if not (d / "pi05_infer.py").exists():
        raise FileNotFoundError(
            f"realtime-vla not found at {d} (need pi05_infer.py). Clone it in-tree:\n"
            f"  git clone https://github.com/dexmal/realtime-vla.git {d}\n"
            f"or set REALTIME_VLA_DIR."
        )
    # append (not insert) to minimize shadowing of stdlib/other modules by the repo's loose files.
    if str(d) not in sys.path:
        sys.path.append(str(d))
    import pi05_infer  # noqa: PLC0415

    return pi05_infer


def _apply_stage_timing_patch(pi05_infer) -> None:
    """Wrap the three eager stage functions with cuda-event timing, once per process (sentinel)."""
    if getattr(pi05_infer, "_edge_stage_timed", False):
        return

    def make_wrapper(orig, scope):
        def wrapped(*args, **kwargs):
            if not _timing["on"]:
                return orig(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = orig(*args, **kwargs)
            end.record()
            _timing["events"][scope].append((start, end))
            return out

        return wrapped

    for fn_name, scope in _STAGE_SCOPE.items():
        setattr(pi05_infer, fn_name, make_wrapper(getattr(pi05_infer, fn_name), scope))
    pi05_infer._edge_stage_timed = True


class Pi05RealtimeVlaSystem(ProfiledSystem):
    def load(
        self,
        *,
        config_name: str,
        checkpoint: str,
        prompt_len: int,
        num_steps: int,
        batch_size: int,
    ) -> LoadedSystem:
        import triton

        if config_name not in RT_REGISTRY:
            raise ValueError(f"unknown config_name '{config_name}'. known: {sorted(RT_REGISTRY)}")
        if not torch.cuda.is_available():
            raise RuntimeError("realtime-vla needs a CUDA GPU (torch.cuda.is_available() is False).")
        if batch_size != 1:
            raise ValueError("realtime-vla runs a single observation per forward (batch_size must be 1).")
        if num_steps != _FIXED_NUM_STEPS:
            print(
                f"[pi05_realtimevla] NOTE: realtime-vla hard-locks flow steps to {_FIXED_NUM_STEPS}; "
                f"--num-steps={num_steps} is ignored for this backend."
            )

        cfg = RT_REGISTRY[config_name]
        num_views = int(cfg["num_views"])
        chunk_size = int(cfg["chunk_size"])

        pi05_infer = _import_pi05_infer()
        Pi05Inference = pi05_infer.Pi05Inference

        # Spec-conformant dummy inputs (latency is value-independent). cuda/bf16 per the repo API.
        images = torch.randn(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        noise = torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda")

        if checkpoint == "random":
            # benchmark.py's tokenizer-free path: provide only random language embeds; the rest of
            # the weight buffers stay uninitialized (fine for latency). discrete_state_input=False
            # so no tokenizer/embedding table is needed. prompt_len = #language tokens.
            ckpt = {"language_embeds": torch.randn(prompt_len, 2048, dtype=torch.bfloat16, device="cuda")}
            infer = Pi05Inference(ckpt, num_views=num_views, chunk_size=chunk_size, discrete_state_input=False)
            fwd_args: tuple = (images, noise)
            ckpt_resolved = "random-init"
            discrete = False
        else:
            # Real converted .pkl (from realtime-vla/convert_from_jax_pi05.py). EXPERIMENTAL /
            # deferred: needs the HF paligemma-3b-pt-224 tokenizer (REALTIME_VLA_TOKENIZER).
            import pickle

            import numpy as np

            discrete = bool(cfg["discrete_state_input"])
            with open(checkpoint, "rb") as f:
                ckpt = pickle.load(f)
            tok = os.environ.get("REALTIME_VLA_TOKENIZER")
            if discrete and not tok:
                raise ValueError(
                    "discrete_state_input checkpoint needs a tokenizer: set REALTIME_VLA_TOKENIZER "
                    "to a local paligemma-3b-pt-224 dir."
                )
            infer = Pi05Inference(
                ckpt,
                num_views=num_views,
                chunk_size=chunk_size,
                tokenizer_path=tok,
                max_tokenize_len=prompt_len,
                discrete_state_input=discrete,
            )
            if discrete:
                fwd_args = (images, noise, "pick up the object", np.zeros(8, dtype=np.int64))
            else:
                fwd_args = (images, noise)
            ckpt_resolved = str(checkpoint)
            print("[pi05_realtimevla] NOTE: real-weights path is experimental/unvalidated.")

        def infer_at(k: int):
            return lambda: infer.forward(*fwd_args)

        def block(_out):
            torch.cuda.synchronize()

        def phase_profiler(*, warmup: int, iters: int, logdir: str | None = None) -> dict:
            """Vision/VLM/Action via cuda-event timing of the eager Triton stages (graphs off).
            `logdir` is unused (no trace files for this backend)."""
            _apply_stage_timing_patch(pi05_infer)
            # Drive a realistic workload into the buffers + warm the eager kernels (timing off).
            for _ in range(max(warmup, 1)):
                infer.forward(*fwd_args)
            for _ in range(2):
                infer.record_run()
            torch.cuda.synchronize()

            ev = _timing["events"]
            for v in ev.values():
                v.clear()
            _timing["on"] = True
            try:
                for _ in range(iters):
                    infer.record_run()
                torch.cuda.synchronize()
            finally:
                _timing["on"] = False

            n = max(int(iters), 1)
            per = {
                s: (sum(a.elapsed_time(b) for a, b in evs) / n if evs else 0.0)
                for s, evs in ev.items()
            }
            total = per["vision"] + per["vlm"] + per["action"]
            return {
                "ok": total > 0,
                "method": "eager-stage-timing",
                "phases_ms_per_infer": {"vision": per["vision"], "vlm": per["vlm"], "action": per["action"]},
                "residual_ms_per_infer": 0.0,  # pi05_model == exactly these three stages
                "total_gpu_ms_per_infer": total,
                "attributed_frac": 1.0,
                "top_kernels": [],
                "error": None if total > 0 else "no stage events captured",
            }

        meta = {
            "backend": "torch-triton",
            "attribution": "cuda-event timing of eager Triton stages (graphs off); E2E = CUDA-graph replay",
            "checkpoint": ckpt_resolved,
            "discrete_state_input": discrete,
            "action_horizon": chunk_size,
            "action_dim": 32,
            "max_token_len": int(prompt_len),
            "n_images": num_views,
            "tokens_per_image_nominal": _TOKENS_PER_IMAGE,
            "prefix_len_nominal": num_views * _TOKENS_PER_IMAGE + int(prompt_len),
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "dtype": "bfloat16",
            "num_steps_fixed": _FIXED_NUM_STEPS,
            "device_kind": torch.cuda.get_device_name(0),
            "n_devices": 1,
            "torch_version": torch.__version__,
            "triton_version": getattr(triton, "__version__", "?"),
        }

        return LoadedSystem(
            name="pi05_realtimevla",
            config_name=config_name,
            prompt_len=prompt_len,
            num_steps=_FIXED_NUM_STEPS,
            batch_size=batch_size,
            infer_at=infer_at,
            block=block,
            meta=meta,
            phase_profiler=phase_profiler,
        )
