"""CLI config + orchestration for the edge-robotics profiler.

This module is import-light on purpose: defining `Config` must NOT import jax/openpi, so the
entrypoint (scripts/profile_policy.py) can parse args and set CUDA_VISIBLE_DEVICES *before*
the first `import jax`. All heavy imports live inside `run()`.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # --- required identity flags (no silent defaults) ---
    system: Literal["pi05_jax"]
    """Which system to profile. Only pi05_jax for now."""
    config_name: str
    """pi-0.5 config name (e.g. pi05_droid, pi05_libero, pi05_base, debug_pi05)."""
    checkpoint: str
    """Checkpoint dir: gs:// URI, local path, or the literal 'random' for random-init (no download)."""
    gpu: int
    """CUDA device index to pin (this machine: 4-7 are idle)."""

    # --- operational knobs (have defaults, all overridable) ---
    prompt_lens: list[int] = field(default_factory=lambda: [200])
    """Prompt lengths (max_token_len) to sweep, e.g. --prompt-lens 16 32 64 128 200."""
    num_steps: int = 10
    """Flow-matching denoise steps for the headline run."""
    warmup: int = 3
    """Warmup iterations (absorb JIT compile) before timing."""
    iters: int = 20
    """Timed iterations per measurement."""
    batch_size: int = 1
    """Inference batch size."""
    output: str = "out/profile"
    """Output path prefix for <prefix>.json and <prefix>.csv."""

    # --- phase-attribution methods ---
    regression: bool = True
    """Run the num_steps regression (Action vs Vision+VLM split). Robust, parser-free."""
    regression_steps: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    """num_steps values used for the regression fit."""
    probe_vision: bool = True
    """Directly time the public SigLIP encoder to split Vision from VLM (VLM = intercept - Vision)."""
    jax_trace: bool = False
    """Capture a JAX profiler trace and parse it for the Vision-vs-VLM device-time split."""
    trace_dir: str | None = None
    """Where to write JAX traces (default: <output>_trace)."""
    nsys: bool = False
    """Print the Nsight Systems (CUDA) command to wrap this run for a kernel-level timeline."""


def _make_system(name: str):
    if name == "pi05_jax":
        from .systems.pi05_jax import Pi05JaxSystem

        return Pi05JaxSystem()
    raise ValueError(f"unknown system '{name}'")


def run(config: Config) -> None:
    import jax

    from .metrics import build_row
    from .profiling.jax_profiler import parse_trace, profile_inference
    from .profiling.regression import fit_num_steps_regression
    from .profiling.walltime import time_callable
    from .report import render_table, write_outputs

    n_dev = jax.device_count()
    dev_kind = jax.devices()[0].device_kind
    print(f"[run] jax devices: {n_dev} x {dev_kind}")
    if n_dev != 1:
        print(
            f"[run] WARNING: expected exactly 1 visible GPU (set CUDA_VISIBLE_DEVICES={config.gpu}); "
            f"got {n_dev}. Timings may be off."
        )

    system = _make_system(config.system)
    trace_dir = config.trace_dir or f"{config.output}_trace"

    rows = []
    top_meta: dict = {}
    for L in config.prompt_lens:
        print(f"\n[run] === prompt_len (max_token_len) = {L} ===")
        loaded = system.load(
            config_name=config.config_name,
            checkpoint=config.checkpoint,
            prompt_len=L,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
        )
        if not top_meta:
            top_meta = {
                "system": config.system,
                "config_name": config.config_name,
                "checkpoint": loaded.meta.get("checkpoint"),
                "device_kind": loaded.meta.get("device_kind"),
                "n_devices": loaded.meta.get("n_devices"),
                "num_steps": config.num_steps,
                "batch_size": config.batch_size,
                "warmup": config.warmup,
                "iters": config.iters,
                "regression_steps": config.regression_steps if config.regression else None,
                "jax_version": loaded.meta.get("jax_version"),
                "model": {
                    k: loaded.meta.get(k)
                    for k in (
                        "action_horizon",
                        "action_dim",
                        "paligemma_variant",
                        "action_expert_variant",
                        "dtype",
                        "n_images",
                        "prefix_len_nominal",
                    )
                },
            }

        print("[run] timing E2E (full sample_actions)...")
        e2e_samples = time_callable(loaded.infer(), loaded.block, warmup=config.warmup, iters=config.iters)

        vision_probe_ms = None
        if config.probe_vision and loaded.vision_infer is not None:
            print("[run] probing Vision (public SigLIP encoder over all images)...")
            import numpy as _np

            vsamples = time_callable(loaded.vision_infer, loaded.block, warmup=config.warmup, iters=config.iters)
            vision_probe_ms = float(_np.median(vsamples))
            print(f"[run]   vision={vision_probe_ms:.3f} ms")

        regression = None
        if config.regression:
            print(f"[run] num_steps regression over {config.regression_steps}...")
            regression = fit_num_steps_regression(
                loaded.infer_at,
                loaded.block,
                steps_list=config.regression_steps,
                warmup=config.warmup,
                iters=config.iters,
            )
            print(
                f"[run]   slope={regression['slope_ms_per_step']:.3f} ms/step, "
                f"intercept={regression['intercept_ms']:.3f} ms, r2={regression['r2']:.4f}"
            )

        trace = None
        if config.jax_trace:
            ld = os.path.join(trace_dir, f"L{L}")
            print(f"[run] capturing JAX profiler trace -> {ld}")
            profile_inference(loaded.infer(), loaded.block, ld, warmup=config.warmup, iters=config.iters)
            trace = parse_trace(ld, iters=config.iters)
            if trace.get("ok"):
                print(
                    f"[run]   trace total GPU device time: {trace['total_gpu_ms_per_infer']:.3f} ms/infer "
                    f"(cross-check vs E2E)"
                )
            else:
                print(f"[run]   trace parse failed: {trace.get('error')} (headline numbers unaffected)")

        row = build_row(
            prompt_len=L,
            num_steps=config.num_steps,
            e2e_samples=e2e_samples,
            regression=regression,
            trace=trace,
            meta=loaded.meta,
            vision_probe_ms=vision_probe_ms,
        )
        rows.append(row)
        # free device memory between configs (each L re-jits at a new shape)
        del loaded

    print("\n" + render_table(rows, top_meta))
    json_path, csv_path = write_outputs(config.output, rows, top_meta)
    print(f"\n[run] wrote {json_path} and {csv_path}")

    if config.nsys:
        from .profiling.cuda_profiler import print_nsys_hint

        print_nsys_hint(config)


def main() -> None:
    """Console-script entry (`profile-policy`). Prefer scripts/profile_policy.py for GPU pinning."""
    import tyro

    config = tyro.cli(Config)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    run(config)


if __name__ == "__main__":
    main()
