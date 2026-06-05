"""pi-0.5 (native JAX/Flax) system.

Loads the model exactly the way openpi's own inference path does
(`restore_params` + `BaseModelConfig.load`, then `nnx_utils.module_jit(sample_actions)` —
the same JIT wrapper `openpi.policies.policy.Policy` uses) and hands back an UNMODIFIED
`sample_actions` callable. We never reimplement or instrument the forward pass; the profiler
attributes phases from the JAX profiler trace + a num_steps regression over this callable.

Config values mirror openpi/src/openpi/training/config.py so we don't need to import the
training config registry (which pulls in lerobot/tensorflow data-loading deps we don't want).
"""

from __future__ import annotations

from .base import LoadedSystem, ProfiledSystem

# Mirrors the pi05_* entries in openpi training/config.py. Kwargs go straight to Pi0Config.
# (Pi0Config.__post_init__ fills max_token_len=200 and discrete_state_input=pi05 when None.)
PI05_REGISTRY: dict[str, dict] = {
    "pi05_base": dict(pi05=True),  # action_horizon=50 (default)
    "pi05_aloha": dict(pi05=True),
    "pi05_droid": dict(pi05=True, action_horizon=15),
    "pi05_libero": dict(pi05=True, action_horizon=10, discrete_state_input=False),
    # tiny model for fast pipeline smoke tests (no checkpoint needed; use --checkpoint random)
    "debug_pi05": dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
}

# SigLIP So400m/14 @224x224 -> (224/14)^2 = 256 tokens per image (nominal; informational only).
_TOKENS_PER_IMAGE = 256


class Pi05JaxSystem(ProfiledSystem):
    def load(
        self,
        *,
        config_name: str,
        checkpoint: str,
        prompt_len: int,
        num_steps: int,
        batch_size: int,
    ) -> LoadedSystem:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0_config import Pi0Config
        from openpi.shared import download, nnx_utils

        if config_name not in PI05_REGISTRY:
            raise ValueError(
                f"unknown config_name '{config_name}'. known: {sorted(PI05_REGISTRY)}"
            )

        # Build the model config directly (prompt length is the pi-0.5 spec knob max_token_len).
        model_cfg = Pi0Config(**PI05_REGISTRY[config_name], max_token_len=prompt_len)

        # Load weights the same way openpi does, OR random-init for a no-download smoke test.
        if checkpoint == "random":
            model = model_cfg.create(jax.random.key(0))
            ckpt_resolved = "random-init"
        else:
            # openpi-assets is a public bucket; gsutil isn't installed here so openpi falls back
            # to gcsfs, which needs token="anon" for anonymous read. maybe_download forwards
            # kwargs to fsspec. (Ignored for local paths, which short-circuit before fsspec.)
            dl_kwargs = {"token": "anon"} if str(checkpoint).startswith("gs://") else {}
            ckpt_dir = download.maybe_download(checkpoint, **dl_kwargs)
            params = _model.restore_params(str(ckpt_dir / "params"), dtype=jnp.bfloat16)
            model = model_cfg.load(params)
            ckpt_resolved = str(ckpt_dir)

        # Dummy inputs conforming to the pi-0.5 spec (no data pipeline; timing is value-independent).
        obs = model_cfg.fake_obs(batch_size)

        # The exact JIT wrapper Policy uses. num_steps is passed as a (dynamic) arg, so a single
        # compiled function serves every k in the regression — no recompilation per step count.
        sample = nnx_utils.module_jit(model.sample_actions)
        key = jax.random.key(0)

        def infer_at(k: int):
            return lambda: sample(key, obs, num_steps=k)

        # Vision-only probe: run JUST the SigLIP encoder over all camera images, exactly as
        # embed_prefix does (one call per image). This is a minimal, direct use of the model's
        # public vision submodule (PaliGemma.img) so we can separate Vision from VLM, which the
        # fused XLA trace cannot. We split the model once (same pattern as module_jit) and jit it.
        _graphdef, _state = nnx.split(model)
        _images = obs.images

        @jax.jit
        def _vision(state):
            m = nnx.merge(_graphdef, state)
            return [m.PaliGemma.img(_images[name], train=False)[0] for name in _images]

        def vision_infer():
            return _vision(_state)

        n_images = len(obs.images)
        meta = {
            "backend": "jax",
            "checkpoint": ckpt_resolved,
            "action_horizon": int(model_cfg.action_horizon),
            "action_dim": int(model_cfg.action_dim),
            "max_token_len": int(model_cfg.max_token_len),
            "n_images": n_images,
            "tokens_per_image_nominal": _TOKENS_PER_IMAGE,
            "prefix_len_nominal": n_images * _TOKENS_PER_IMAGE + int(model_cfg.max_token_len),
            "paligemma_variant": model_cfg.paligemma_variant,
            "action_expert_variant": model_cfg.action_expert_variant,
            "dtype": model_cfg.dtype,
            "device_kind": jax.devices()[0].device_kind,
            "n_devices": jax.device_count(),
            "jax_version": jax.__version__,
        }

        return LoadedSystem(
            name="pi05_jax",
            config_name=config_name,
            prompt_len=prompt_len,
            num_steps=num_steps,
            batch_size=batch_size,
            infer_at=infer_at,
            block=jax.block_until_ready,
            meta=meta,
            vision_infer=vision_infer,
        )
