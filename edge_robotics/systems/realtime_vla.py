"""pi-0.5 on the realtime-vla (PyTorch + Triton) backend.

Profiles dexmal/realtime-vla's `Pi05Inference` — a fused-Triton reimplementation of pi-0.5. The repo
captures the WHOLE forward (`pi05_model` = vision_encoder -> transformer_encoder ->
transformer_decoder) into ONE `torch.cuda.CUDAGraph` and replays it with a single `cudaGraphLaunch`
(`infer.forward()`), which leaves no eager phase boundaries for NVTX to grab.

We get the SAME vision/vlm/action NVTX split as the openpi-torch backend by re-capturing the model
as THREE separate sub-graphs — one per stage — and replaying them in eager glue with an NVTX range
around each (graphs still ON). This is clean here because the three stages already communicate only
through persistent in-place `buffers` (no transient allocations, no returned tensors), so splitting
the capture is numerically identical to the single graph; only the launch count changes (3 vs 1).
We do this from THIS wrapper without editing the vendored repo. The original single graph
(`infer.forward`) is kept as the `infer_native` cross-check.

  * infer_native    -> the repo's own single captured graph (the deployed path)  = HEADLINE latency
  * infer_segmented -> [vision][vlm][action] sub-graph replays + NVTX = the within-run BREAKDOWN
  * component_profiler -> each sub-graph replayed standalone (secondary cross-check, graphs ON)

nsys `nvtx_gpu_proj_sum` then attributes GPU time per phase; kernel-family buckets are also reported.

The realtime-vla repo is vendored in-tree at <repo>/realtime-vla (cloned by install.sh) and imported
via sys.path. Override with REALTIME_VLA_DIR.

CAVEATS (see README): diffusion steps are hard-locked to 10 in realtime-vla's forward, so
--num-steps != 10 is ignored (warned). The Triton kernels are tuned for RTX 4090/5090.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

from ..profiling.nsys import nvtx_range
from .base import LoadedSystem, ProfiledSystem

# realtime-vla's pi05 knobs per config (chunk_size == action_horizon; num_views == #camera images).
# Gemma sizes / action_dim=32 / 10 flow steps are baked into the Triton kernels (pi05 only for now).
RT_REGISTRY: dict[str, dict] = {
    "pi05_base": dict(num_views=3, chunk_size=50, discrete_state_input=True),
    "pi05_aloha": dict(num_views=3, chunk_size=50, discrete_state_input=True),
    "pi05_droid": dict(num_views=3, chunk_size=15, discrete_state_input=True),
    "pi05_libero": dict(num_views=3, chunk_size=10, discrete_state_input=False),
}

_TOKENS_PER_IMAGE = 256
_FIXED_NUM_STEPS = 10  # realtime-vla pi05 forward is hard-locked to 10 flow steps.


def _realtime_vla_dir() -> Path:
    """Vendored realtime-vla checkout (repo_root/realtime-vla), overridable via REALTIME_VLA_DIR."""
    env = os.environ.get("REALTIME_VLA_DIR")
    if env:
        return Path(env).expanduser().resolve()
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
    if str(d) not in sys.path:
        sys.path.append(str(d))
    import pi05_infer  # noqa: PLC0415

    return pi05_infer


class RealtimeVlaSystem(ProfiledSystem):
    def load(
        self,
        *,
        config_name: str,
        checkpoint: str,
        prompt_len: int | None,
        num_steps: int,
        batch_size: int,
    ) -> LoadedSystem:
        import triton

        # realtime-vla currently wraps pi05 only (RT_REGISTRY; the repo's Pi0Inference could add pi0
        # but isn't wired yet). prompt_len=None -> the pi05 native max_token_len (200).
        prompt_len = 200 if prompt_len is None else int(prompt_len)

        if config_name not in RT_REGISTRY:
            raise ValueError(f"unknown config_name '{config_name}'. known: {sorted(RT_REGISTRY)}")
        if not torch.cuda.is_available():
            raise RuntimeError("realtime-vla needs a CUDA GPU (torch.cuda.is_available() is False).")
        if batch_size != 1:
            raise ValueError("realtime-vla runs a single observation per forward (batch_size must be 1).")
        if num_steps != _FIXED_NUM_STEPS:
            print(
                f"[realtime_vla] NOTE: realtime-vla hard-locks flow steps to {_FIXED_NUM_STEPS}; "
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
            # benchmark.py's tokenizer-free path: only random language embeds; discrete_state_input
            # =False so no tokenizer/embedding table needed. prompt_len = #language tokens.
            ckpt = {"language_embeds": torch.randn(prompt_len, 2048, dtype=torch.bfloat16, device="cuda")}
            infer = Pi05Inference(ckpt, num_views=num_views, chunk_size=chunk_size, discrete_state_input=False)
            fwd_args: tuple = (images, noise)
            ckpt_resolved = "random-init"
            discrete = False
        else:
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
                ckpt, num_views=num_views, chunk_size=chunk_size, tokenizer_path=tok,
                max_tokenize_len=prompt_len, discrete_state_input=discrete,
            )
            fwd_args = (images, noise, "pick up the object", np.zeros(8, dtype=np.int64)) if discrete else (images, noise)
            ckpt_resolved = str(checkpoint)
            print("[realtime_vla] NOTE: real-weights path is experimental/unvalidated.")

        # The eager input-copy from Pi05Inference.forward (lines 811-824), factored out so the
        # captured sub-graphs only contain the *compute*, exactly like the repo's single graph (whose
        # input copies are also eager, before replay). prompt_embeds/decoder_rope depend only on the
        # fixed prompt, so precompute once.
        if discrete:
            prompt_embeds, prompt_len_actual = infer.build_prompt_embeds(
                task_prompt=fwd_args[2], state_tokens=fwd_args[3]
            )
        else:
            prompt_embeds = infer.weights["language_embeds"]
            prompt_len_actual = prompt_embeds.shape[0]
        _start = num_views * _TOKENS_PER_IMAGE
        _decoder_rope = infer.get_decoder_rope_weights(prompt_len_actual)

        def _fill_inputs():
            b = infer.buffers
            b["encoder_x"][_start : _start + prompt_len_actual].copy_(prompt_embeds)
            b["valid_encoder_len"].fill_(_start + prompt_len_actual)
            b["decoder_rope_weights"].copy_(_decoder_rope)
            b["observation_images_normalized"].copy_(images)
            b["diffusion_noise"].copy_(noise)  # decoder overwrites this in place; re-seed each run

        # Re-capture the model as three per-stage sub-graphs sharing one capture pool. Warm the full
        # pipeline first so every Triton kernel is JIT-compiled/autotuned (capture forbids that).
        for _ in range(3):
            pi05_infer.pi05_model(infer.weights, infer.buffers, num_views, infer.encoder_seq_len)
        torch.cuda.synchronize()

        def _capture(fn, pool=None):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                fn()
            return g

        g_vision = _capture(lambda: pi05_infer.vision_encoder(infer.weights, infer.buffers, num_views))
        _pool = g_vision.pool()
        g_vlm = _capture(
            lambda: pi05_infer.transformer_encoder(infer.weights, infer.buffers, infer.encoder_seq_len), _pool
        )
        g_action = _capture(
            lambda: pi05_infer.transformer_decoder(
                infer.weights, infer.buffers, infer.encoder_seq_len, _FIXED_NUM_STEPS
            ),
            _pool,
        )
        _phase_graphs = {"vision": g_vision, "vlm": g_vlm, "action": g_action}

        @torch.inference_mode()
        def infer_segmented():
            # Three sub-graph replays glued in eager python with NVTX around each — graphs ON, and
            # nsys attributes each replay's kernels to its range. Numerically identical to the repo's
            # single graph (same kernels, same persistent buffers); only the launch count differs.
            _fill_inputs()
            with nvtx_range("vision"):
                g_vision.replay()
            with nvtx_range("vlm"):
                g_vlm.replay()
            with nvtx_range("action"):
                g_action.replay()
            return infer.buffers["diffusion_noise"]

        @torch.inference_mode()
        def infer_native():
            # The repo's own single captured graph — cross-check for the segmented wall.
            return infer.forward(*fwd_args)

        def block(_out):
            torch.cuda.synchronize()

        def component_profiler(*, warmup: int, iters: int) -> dict:
            """Time each stage sub-graph standalone (graphs ON). Buffers persist across replays, so a
            stage can be replayed in isolation once the pipeline has been primed."""
            n = max(int(iters), 1)
            with torch.inference_mode():
                _fill_inputs()
                for _ in range(max(warmup, 1)):
                    g_vision.replay()
                    g_vlm.replay()
                    g_action.replay()
                torch.cuda.synchronize()

                def _time(graph) -> float:
                    xs = []
                    for _ in range(n):
                        t0 = time.perf_counter()
                        graph.replay()
                        torch.cuda.synchronize()
                        xs.append((time.perf_counter() - t0) * 1000.0)
                    return statistics.median(xs)

                per = {name: _time(g) for name, g in _phase_graphs.items()}
            total = sum(per.values())
            return {
                "ok": total > 0,
                "method": "standalone-subgraph-replay-wall",
                "phases_ms_per_infer": per,
                "total_gpu_ms_per_infer": total,
                "error": None if total > 0 else "no component timings",
            }

        meta = {
            "backend": "torch-triton",
            "attribution": "headline = the repo's single captured graph (deployed); component split = "
            "nsys NVTX GPU projection over per-stage CUDA sub-graphs (graphs ON)",
            "checkpoint": ckpt_resolved,
            "real_weights": checkpoint != "random",
            "pi05": True,
            "discrete_state_input": discrete,
            "proprioception": "prompt_discrete" if discrete else "none",
            "action_horizon": chunk_size,
            "action_dim": 32,
            "max_token_len": int(prompt_len),
            "n_images": num_views,
            "tokens_per_image_nominal": _TOKENS_PER_IMAGE,
            "prefix_len_nominal": num_views * _TOKENS_PER_IMAGE + int(prompt_len),
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "num_steps_fixed": _FIXED_NUM_STEPS,
            "graphs_on": True,
            "device_kind": torch.cuda.get_device_name(0),
            "n_devices": 1,
            "torch_version": torch.__version__,
            "triton_version": getattr(triton, "__version__", "?"),
        }

        return LoadedSystem(
            name="realtime_vla",
            config_name=config_name,
            prompt_len=prompt_len,
            num_steps=_FIXED_NUM_STEPS,
            batch_size=batch_size,
            infer_segmented=infer_segmented,
            block=block,
            nvtx_phases=("vision", "vlm", "action"),
            infer_native=infer_native,
            component_profiler=component_profiler,
            meta=meta,
        )
