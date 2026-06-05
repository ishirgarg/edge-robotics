"""CLI config + orchestration for the edge-robotics profiler.

This module is import-light on purpose: defining `Config` must NOT import jax/openpi, so the
entrypoint (scripts/profile_policy.py) can parse args and set CUDA_VISIBLE_DEVICES / XLA_FLAGS
*before* the first `import jax`. All heavy imports live inside `run()`.
"""

from __future__ import annotations

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
    """CUDA device index to pin."""

    # --- operational knobs (have defaults, all overridable) ---
    prompt_lens: list[int] = field(default_factory=lambda: [200])
    """Prompt lengths (max_token_len) to sweep, e.g. --prompt-lens 16 32 64 128 200."""
    num_steps: int = 10
    """Flow-matching denoise steps."""
    warmup: int = 3
    """Warmup iterations (absorb JIT compile) before timing."""
    iters: int = 20
    """Timed iterations per measurement."""
    batch_size: int = 1
    """Inference batch size."""
    output: str = "out/profile"
    """Output path prefix for <prefix>.json and <prefix>.csv (overwritten on rerun)."""
    trace_dir: str | None = None
    """Where to write JAX traces (default: <output>_trace). Wiped per prompt_len on each run."""


def _make_system(name: str):
    if name == "pi05_jax":
        from .systems.pi05_jax import Pi05JaxSystem

        return Pi05JaxSystem()
    raise ValueError(f"unknown system '{name}'")


def run(config: Config) -> None:
    import os
    import shutil

    import jax

    from .metrics import build_row
    from .profiling.jax_profiler import parse_trace, profile_inference
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
    # Phase attribution needs CUDA graphs off so kernels keep their named_scope metadata; the
    # entrypoint sets this before importing jax. Warn loudly if it didn't take.
    if "xla_gpu_enable_command_buffer=" not in os.environ.get("XLA_FLAGS", ""):
        print(
            "[run] WARNING: XLA_FLAGS lacks '--xla_gpu_enable_command_buffer=' — CUDA graphs may be "
            "ON, which strips per-op trace metadata and breaks phase attribution. Run via "
            "scripts/profile_policy.py (it sets this before importing jax)."
        )

    system = _make_system(config.system)
    trace_dir = config.trace_dir or f"{config.output}_trace"

    # A rerun overrides old results: JSON/CSV are truncated by write_outputs, and we wipe the whole
    # trace dir up front so orphaned per-L subdirs from a previous run (e.g. a different
    # --prompt-lens) can't linger or be mis-parsed.
    shutil.rmtree(trace_dir, ignore_errors=True)
    if os.path.exists(f"{config.output}.json"):
        print(f"[run] overwriting existing results at {config.output}.{{json,csv}} and {trace_dir}/")

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
                "jax_version": loaded.meta.get("jax_version"),
                "attribution": "jax.named_scope trace buckets (CUDA graphs disabled)",
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

        ld = os.path.join(trace_dir, f"L{L}")
        print(f"[run] capturing JAX profiler trace -> {ld}")
        profile_inference(loaded.infer(), loaded.block, ld, warmup=config.warmup, iters=config.iters)
        trace = parse_trace(ld, iters=config.iters)
        if trace.get("ok"):
            p = trace["phases_ms_per_infer"]
            print(
                f"[run]   vision={p['vision']:.3f} vlm={p['vlm']:.3f} action={p['action']:.3f} ms/infer "
                f"(attributed {trace['attributed_frac']*100:.1f}% of {trace['total_gpu_ms_per_infer']:.3f}ms GPU)"
            )
        else:
            print(f"[run]   trace parse failed: {trace.get('error')} (E2E/Freq still reported)")

        rows.append(build_row(prompt_len=L, num_steps=config.num_steps, e2e_samples=e2e_samples,
                              trace=trace, meta=loaded.meta))
        # free device memory between configs (each L re-jits at a new shape)
        del loaded

    print("\n" + render_table(rows, top_meta))
    json_path, csv_path = write_outputs(config.output, rows, top_meta)
    print(f"\n[run] wrote {json_path} and {csv_path}")


def main() -> None:
    """Console-script entry (`profile-policy`). Prefer scripts/profile_policy.py for GPU pinning."""
    import os

    import tyro

    config = tyro.cli(Config)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    _ensure_command_buffers_disabled()
    run(config)


def _ensure_command_buffers_disabled() -> None:
    """Append '--xla_gpu_enable_command_buffer=' (disable CUDA graphs) to XLA_FLAGS so the
    profiler can attribute device time per phase. Must run before the first `import jax`."""
    import os

    flag = "--xla_gpu_enable_command_buffer="
    existing = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_enable_command_buffer=" not in existing:
        os.environ["XLA_FLAGS"] = (existing + " " + flag).strip()


if __name__ == "__main__":
    main()
