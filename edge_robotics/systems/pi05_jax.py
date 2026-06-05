"""pi-0.5 (native JAX/Flax) system.

Loads the model exactly the way openpi's own inference path does
(`restore_params` + `BaseModelConfig.load`, then `nnx_utils.module_jit(sample_actions)` —
the same JIT wrapper `openpi.policies.policy.Policy` uses) and hands back the `sample_actions`
callable. The forward math is never reimplemented; the only instrumentation is semantically
inert `jax.named_scope` tags wrapped around the public image/LLM submodule calls so the JAX
profiler trace can attribute device time to the Vision / VLM / Action phases (see
`apply_namescope_patch` below and `edge_robotics/profiling/jax_profiler.py:parse_trace`).

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

# The three phases the profiler trace is bucketed into (see profiling/jax_profiler.py). The
# strings are the jax.named_scope labels and the trace path-segments parse_trace matches on.
PHASE_SCOPES = ("vision", "vlm", "action")


def apply_namescope_patch(model) -> None:
    """Tag the public image/LLM submodule calls with `jax.named_scope` so the profiler trace can
    attribute device time per phase. NOT a forward-pass rewrite: each wrapper just opens a scope
    and forwards verbatim to the original `__call__`.

    pi-0.5 builds `self.PaliGemma = nnx.Dict(llm=ToNNX(Gemma), img=ToNNX(SigLIP))`, so `img` and
    `llm` are instances of the SAME `nnx_bridge.ToNNX` class -- we therefore patch that one
    `__call__` and pick the scope from the call arguments:
      * `method="embed"`        -> "vlm"     (Gemma token embedding)
      * first arg is a token list -> "vlm" if no kv_cache (prefix prefill), else "action" (denoise)
      * otherwise (image tensor)  -> "vision" (SigLIP encode)
    The patch is applied to the class once per process (guarded by a sentinel) so a prompt-length
    sweep -- which calls load() once per length -- never stacks wrappers.
    """
    import jax

    bridge_t = type(model.PaliGemma.img)
    if getattr(bridge_t, "_edge_namescoped", False):
        return  # already patched this process
    orig_call = bridge_t.__call__

    def scoped_call(self, *args, **kwargs):
        method = kwargs.get("method")
        first = args[0] if args else None
        if method == "embed":
            scope = "vlm"
        elif isinstance(first, (list, tuple)):  # Gemma forward: [prefix, None] or [None, suffix]
            scope = "action" if kwargs.get("kv_cache") is not None else "vlm"
        else:  # SigLIP image encode (4-D image tensor + train= kwarg)
            scope = "vision"
        with jax.named_scope(scope):
            return orig_call(self, *args, **kwargs)

    bridge_t.__call__ = scoped_call
    bridge_t._edge_namescoped = True


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

        # Tag the public image/LLM submodule calls with named scopes so the profiler trace is
        # parseable into Vision/VLM/Action. Inert annotations only — no forward-pass rewrite.
        apply_namescope_patch(model)

        # Dummy inputs conforming to the pi-0.5 spec (no data pipeline; timing is value-independent).
        obs = model_cfg.fake_obs(batch_size)

        # The exact JIT wrapper Policy uses. num_steps is passed as a (dynamic) arg, so a single
        # compiled function serves any step count without recompilation.
        sample = nnx_utils.module_jit(model.sample_actions)
        key = jax.random.key(0)

        def infer_at(k: int):
            return lambda: sample(key, obs, num_steps=k)

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
        )
