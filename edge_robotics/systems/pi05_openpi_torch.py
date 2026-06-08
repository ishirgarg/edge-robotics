"""pi-0.5 on openpi's NATIVE PyTorch backend (`openpi.models_pytorch.pi0_pytorch.PI0Pytorch`).

Profiled the new way: the model runs ONLY in its fully-optimized, CUDA-graphs-ON form, and the
per-component split is read back from a single Nsight Systems capture — we never run the model
eager just to time submodules.

Why the model is restructured into per-phase graphs
---------------------------------------------------
openpi ships `sample_actions` compiled as ONE `torch.compile(mode="max-autotune")` unit (a CUDA
graph). NVTX ranges pushed at python level can't attribute kernels INSIDE a single fused graph:
the whole thing launches under one `cudaGraphLaunch` that falls outside any NVTX window opened at
replay time, so the split collapses (empirically verified). The fix — the only thing that survives
CUDA graphs — is to run the three phases as SEPARATE compiled/cudagraph callables glued in EAGER
python, with the NVTX range pushed AROUND each call:

  vision -> `embed_prefix`            (SigLIP vision tower x3 images + token embed)
  vlm    -> PaliGemma (gemma_2b) prefill -> KV cache
  action -> the full denoise loop     (num_steps x gemma_300m action expert)

nsys `nvtx_gpu_proj_sum` then attributes GPU device time per phase (validated: sums to ~99% of E2E,
segmented wall within noise of the native single-compile path). Cross-phase tensors (prefix embeds,
the KV cache, masks, state) are `.clone()`d out of each graph's static pool and
`cudagraph_mark_step_begin()` is called before each graph invocation, as CUDA-graph trees require.

Three things this backend exposes (see base.LoadedSystem):
  * infer_segmented — the per-phase-graph E2E with NVTX (headline wall + the nsys split vehicle).
  * infer_native    — openpi's own single `torch.compile(sample_actions)` (cross-check; the wall
                      gap vs segmented is the segmentation overhead).
  * component_profiler — each phase timed standalone in its graphs-on form (ms/infer).

Random-init weights (latency is value-independent); no JAX->torch checkpoint conversion. Requires
openpi's patched transformers (transformers_replace, see install.sh); PI0Pytorch.__init__ checks it.
"""

from __future__ import annotations

import os
import statistics
import time

import torch

from ..profiling.nsys import nvtx_range
from .base import LoadedSystem, ProfiledSystem

# pi-0.5 configs. Mirrors openpi training/config.py; kwargs go straight to Pi0Config. (Moved here
# from the deleted JAX backend — this is now the only consumer.)
PI05_REGISTRY: dict[str, dict] = {
    "pi05_base": dict(pi05=True),  # action_horizon=50 (default)
    "pi05_aloha": dict(pi05=True),
    "pi05_droid": dict(pi05=True, action_horizon=15),
    "pi05_libero": dict(pi05=True, action_horizon=10, discrete_state_input=False),
    # tiny model for fast pipeline smoke tests (no checkpoint needed; use --checkpoint random)
    "debug_pi05": dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
}

PHASE_SCOPES = ("vision", "vlm", "action")
_TOKENS_PER_IMAGE = 256


class Pi05OpenpiTorchSystem(ProfiledSystem):
    def load(
        self,
        *,
        config_name: str,
        checkpoint: str,
        prompt_len: int,
        num_steps: int,
        batch_size: int,
    ) -> LoadedSystem:
        # openpi's pi0_pytorch transitively imports openpi.models.gemma (JAX-side, get_config only,
        # no device ops). Pin JAX to CPU so it can never contend with torch for the GPU.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

        from openpi.models import model as _model
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
        from openpi.shared import array_typing as at

        if config_name not in PI05_REGISTRY:
            raise ValueError(f"unknown config_name '{config_name}'. known: {sorted(PI05_REGISTRY)}")
        if not torch.cuda.is_available():
            raise RuntimeError("openpi-torch needs a CUDA GPU (torch.cuda.is_available() is False).")
        if checkpoint != "random":
            raise ValueError(
                "pi05_openpi_torch currently profiles random-init only (latency is value-independent). "
                "Use --checkpoint random."
            )

        # Two compile modes, both CUDA-graph backed:
        #  * segmented breakdown (OPENPI_TORCH_COMPILE_MODE) — default reduce-overhead: compiles fast
        #    and the component %-split is mode-insensitive, so this keeps the breakdown cheap.
        #  * native cross-check (OPENPI_TORCH_NATIVE_COMPILE_MODE) — default max-autotune, which is
        #    openpi's OWN Pi0Config default (pytorch_compile_mode="max-autotune"). This makes the
        #    native E2E a FAITHFUL "fully-optimized original" baseline, exactly as openpi ships it.
        compile_mode = os.environ.get("OPENPI_TORCH_COMPILE_MODE", "reduce-overhead")
        native_mode = os.environ.get("OPENPI_TORCH_NATIVE_COMPILE_MODE", "max-autotune")
        graphs_on = compile_mode.lower() not in ("", "none", "off", "eager")
        native_graphs = native_mode.lower() not in ("", "none", "off", "eager")

        # Build the model with NO openpi-side compile — we compile per-phase ourselves (segmented)
        # and separately wrap sample_actions for the native cross-check.
        cfg = Pi0Config(**PI05_REGISTRY[config_name], max_token_len=prompt_len, pytorch_compile_mode=None)
        device = torch.device("cuda")
        # Move to device ONLY (no dtype): PI0Pytorch.__init__ already set openpi's deliberate MIXED
        # precision (bf16 weights, fp32 embeddings/norms/projections). A blanket .to(bf16) would
        # break it under torch.compile's pad_mm autotune.
        model = PI0Pytorch(config=cfg).to(device=device).eval()

        # openpi sets `_attn_implementation="eager"` INSIDE sample_actions/denoise_step. Pull it out
        # to here (set once) so the per-phase compile never graph-breaks on the config mutation.
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        pwe = model.paligemma_with_expert

        # ---- Synthetic on-device observation (Pi0Config.inputs_spec; latency is value-independent).
        h, w = _model.IMAGE_RESOLUTION
        img_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        # openpi's torch preprocessing feeds SigLIP NCHW only for channels-FIRST input -> [B,3,H,W].
        images = {k: torch.zeros(batch_size, 3, h, w, dtype=torch.float32, device=device) for k in img_keys}
        image_masks = {k: torch.ones(batch_size, dtype=torch.bool, device=device) for k in img_keys}
        state = torch.zeros(batch_size, cfg.action_dim, dtype=torch.float32, device=device)
        tokenized_prompt = torch.zeros(batch_size, cfg.max_token_len, dtype=torch.int32, device=device)
        tokenized_prompt_mask = torch.ones(batch_size, cfg.max_token_len, dtype=torch.bool, device=device)
        with at.disable_typechecking():
            obs = _model.Observation(
                images=images, image_masks=image_masks, state=state,
                tokenized_prompt=tokenized_prompt, tokenized_prompt_mask=tokenized_prompt_mask,
            )

        # ---- The three phases as plain functions (reused for segmented E2E AND component timing).
        def _vision(obs):
            imgs, img_masks, lang_tokens, lang_masks, st = model._preprocess_observation(obs, train=False)
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                imgs, img_masks, lang_tokens, lang_masks
            )
            return prefix_embs, prefix_pad_masks, prefix_att_masks, st

        def _vlm(prefix_embs, prefix_pad_masks, prefix_att_masks):
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_4d = model._prepare_attention_masks_4d(prefix_att_2d)
            _, pkv = pwe.forward(
                attention_mask=prefix_att_2d_4d, position_ids=pos, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True,
            )
            # Return the KV cache as plain tensors so torch.compile/cudagraph doesn't have to carry a
            # DynamicCache across the phase boundary; rebuilt in _denoise.
            return tuple(pkv.key_cache), tuple(pkv.value_cache)

        def _denoise(state, prefix_pad_masks, key_cache, value_cache, x_t, timestep):
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            cache.key_cache = list(key_cache)
            cache.value_cache = list(value_cache)
            return model.denoise_step(state, prefix_pad_masks, cache, x_t, timestep)

        if graphs_on:
            vision_fn = torch.compile(_vision, mode=compile_mode)
            vlm_fn = torch.compile(_vlm, mode=compile_mode)
            denoise_fn = torch.compile(_denoise, mode=compile_mode)
        else:
            vision_fn, vlm_fn, denoise_fn = _vision, _vlm, _denoise

        def _mark():
            # CUDA-graph trees: tells torch a new model iteration begins so it can reuse the static
            # pool; required before invoking a cudagraph'd callable whose prior output we still hold.
            if graphs_on:
                torch.compiler.cudagraph_mark_step_begin()

        dt_val = -1.0 / num_steps

        @torch.inference_mode()
        def infer_segmented(noise=None):
            """Per-phase-graph E2E with NVTX ranges (headline wall + nsys split vehicle)."""
            if noise is None:
                noise = model.sample_noise((batch_size, cfg.action_horizon, cfg.action_dim), device)
            _mark()
            with nvtx_range("vision"):
                pe, ppm, pam, st = vision_fn(obs)
                pe, ppm, pam, st = pe.clone(), ppm.clone(), pam.clone(), st.clone()
            _mark()
            with nvtx_range("vlm"):
                kc, vc = vlm_fn(pe, ppm, pam)
                kc = tuple(t.clone() for t in kc)
                vc = tuple(t.clone() for t in vc)
            with nvtx_range("action"):
                dt = torch.tensor(dt_val, dtype=torch.float32, device=device)
                x_t = noise
                tt = torch.tensor(1.0, dtype=torch.float32, device=device)
                for _ in range(num_steps):
                    _mark()
                    v_t = denoise_fn(st, ppm, kc, vc, x_t, tt.expand(batch_size)).clone()
                    x_t = x_t + dt * v_t
                    tt = tt + dt
            return x_t

        # Native cross-check: openpi's OWN single torch.compile(sample_actions) fast path, compiled
        # with openpi's default mode (max-autotune) so it faithfully reproduces the repo's
        # fully-optimized inference — the baseline the segmented breakdown is compared against.
        if native_graphs:
            native_sa = torch.compile(model.sample_actions, mode=native_mode)
        else:
            native_sa = model.sample_actions

        @torch.inference_mode()
        def infer_native():
            return native_sa(device, obs, num_steps=num_steps)

        def block(_out):
            torch.cuda.synchronize()

        def component_profiler(*, warmup: int, iters: int, logdir: str | None = None) -> dict:
            """Time each component standalone in its graphs-on form (ms/infer). Upstream inputs are
            computed once and reused so each phase is timed in isolation."""
            n = max(int(iters), 1)
            with torch.inference_mode():
                # Warm + capture stable upstream inputs (cloned out of the graph pools).
                for _ in range(max(warmup, 1)):
                    _mark()
                    pe, ppm, pam, st = vision_fn(obs)
                pe, ppm, pam, st = pe.clone(), ppm.clone(), pam.clone(), st.clone()
                for _ in range(max(warmup, 1)):
                    _mark()
                    kc, vc = vlm_fn(pe, ppm, pam)
                kc = tuple(t.clone() for t in kc)
                vc = tuple(t.clone() for t in vc)
                noise = model.sample_noise((batch_size, cfg.action_horizon, cfg.action_dim), device)
                tt = torch.ones(batch_size, dtype=torch.float32, device=device)
                for _ in range(max(warmup, 1)):
                    _mark()
                    denoise_fn(st, ppm, kc, vc, noise, tt)
                torch.cuda.synchronize()

                def _time(fn) -> float:
                    xs = []
                    for _ in range(n):
                        t0 = time.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        xs.append((time.perf_counter() - t0) * 1000.0)
                    return statistics.median(xs)

                def _vision_call():
                    _mark(); vision_fn(obs)

                def _vlm_call():
                    _mark(); vlm_fn(pe, ppm, pam)

                def _action_call():
                    # Full denoise loop, the way it runs in E2E.
                    dt = torch.tensor(dt_val, dtype=torch.float32, device=device)
                    x_t = noise
                    t_ = torch.tensor(1.0, dtype=torch.float32, device=device)
                    for _ in range(num_steps):
                        _mark()
                        v_t = denoise_fn(st, ppm, kc, vc, x_t, t_.expand(batch_size)).clone()
                        x_t = x_t + dt * v_t
                        t_ = t_ + dt

                per = {"vision": _time(_vision_call), "vlm": _time(_vlm_call), "action": _time(_action_call)}

            total = sum(per.values())
            return {
                "ok": total > 0,
                "method": "standalone-graphs-on-wall",
                "phases_ms_per_infer": per,
                "total_gpu_ms_per_infer": total,
                "error": None if total > 0 else "no component timings",
            }

        meta = {
            "backend": "openpi-torch",
            "attribution": "nsys NVTX GPU projection over per-phase CUDA graphs (graphs ON); "
            "E2E = segmented per-phase graphs (native single-compile reported as cross-check)",
            "checkpoint": "random-init",
            "action_horizon": int(cfg.action_horizon),
            "action_dim": int(cfg.action_dim),
            "max_token_len": int(cfg.max_token_len),
            "n_images": len(img_keys),
            "tokens_per_image_nominal": _TOKENS_PER_IMAGE,
            "prefix_len_nominal": len(img_keys) * _TOKENS_PER_IMAGE + int(cfg.max_token_len),
            "paligemma_variant": cfg.paligemma_variant,
            "action_expert_variant": cfg.action_expert_variant,
            "dtype": cfg.dtype,
            "compile_mode": compile_mode if graphs_on else "eager",
            "native_compile_mode": native_mode if native_graphs else "eager",
            "graphs_on": graphs_on,
            "device_kind": torch.cuda.get_device_name(0),
            "n_devices": 1,
            "torch_version": torch.__version__,
        }

        return LoadedSystem(
            name="pi05_openpi_torch",
            config_name=config_name,
            prompt_len=prompt_len,
            num_steps=num_steps,
            batch_size=batch_size,
            infer_segmented=infer_segmented,
            block=block,
            nvtx_phases=PHASE_SCOPES,
            infer_native=infer_native,
            component_profiler=component_profiler,
            meta=meta,
        )
