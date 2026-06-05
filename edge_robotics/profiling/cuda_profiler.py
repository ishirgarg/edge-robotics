"""CUDA profiler (Nsight Systems) helpers.

Two notes:
  1. The JAX profiler trace already captures CUDA kernels via CUPTI, so for most phase
     attribution you do NOT need nsys — see jax_profiler.py.
  2. For a dedicated kernel-level CUDA timeline, wrap the WHOLE entrypoint in `nsys profile`.
     nsys cannot be started mid-process around a Python region cleanly, so this module just
     (a) reports whether nsys is available and (b) prints the exact command to use.
"""

from __future__ import annotations

import shutil


def nsys_available() -> bool:
    return shutil.which("nsys") is not None


def nsys_command(argv: list[str], *, out: str, gpu: int) -> str:
    """Return the nsys command that wraps a `python scripts/profile_policy.py ...` invocation."""
    inner = " ".join(argv)
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} nsys profile -o {out} "
        f"--trace=cuda,nvtx,osrt --cuda-memory-usage=true --force-overwrite=true "
        f"{inner}"
    )


def print_nsys_hint(config) -> None:
    avail = nsys_available()
    status = "FOUND" if avail else "NOT INSTALLED on this machine"
    print("\n[cuda] Nsight Systems (nsys):", status)
    cmd = nsys_command(
        [
            "python",
            "scripts/profile_policy.py",
            f"--system {config.system}",
            f"--config-name {config.config_name}",
            f"--checkpoint {config.checkpoint}",
            f"--gpu {config.gpu}",
            f"--prompt-lens {' '.join(str(x) for x in config.prompt_lens)}",
            f"--num-steps {config.num_steps}",
            f"--warmup {config.warmup}",
            f"--iters {config.iters}",
            f"--batch-size {config.batch_size}",
            f"--output {config.output}",
            "--no-jax-trace",
            "--no-regression",
            "--no-nsys",
        ],
        out=f"{config.output}_nsys",
        gpu=config.gpu,
    )
    print("[cuda] To capture a kernel-level CUDA timeline, run:\n  " + cmd)
    if not avail:
        print("[cuda] (install Nsight Systems to enable; the JAX trace already has CUDA kernels.)")
