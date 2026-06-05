#!/usr/bin/env python
"""Entrypoint for the edge-robotics profiler.

Pins the GPU via CUDA_VISIBLE_DEVICES BEFORE jax is imported (tyro parsing and Config import
are jax-free; run() imports jax lazily). Run from the repo root after `source env.sh`:

    python scripts/profile_policy.py --system pi05_jax --config-name pi05_droid \
        --checkpoint gs://openpi-assets/checkpoints/pi05_droid --gpu 4 \
        --prompt-lens 16 32 64 128 200 --num-steps 10 --warmup 3 --iters 30 \
        --output out/pi05_droid --jax-trace
"""

import os
import sys

import tyro

# make `import edge_robotics` work even without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edge_robotics.cli import Config, _ensure_command_buffers_disabled, run  # noqa: E402


def main() -> None:
    config = tyro.cli(Config)
    # Must happen before the first `import jax` (which run() triggers).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    # Disable CUDA graphs so the profiler trace keeps per-op named_scope metadata (phase split).
    _ensure_command_buffers_disabled()
    run(config)


if __name__ == "__main__":
    main()
