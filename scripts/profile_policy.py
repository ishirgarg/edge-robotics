#!/usr/bin/env python
"""Entrypoint for the edge-robotics profiler (one system, one config, one prompt_len).

Pins the GPU via CUDA_VISIBLE_DEVICES before any CUDA context is created (Config parsing is
torch-free; run() imports torch lazily). Drives one of four modes (time/nsys/parse/report); the
shell wrappers (profile_one.sh, profile_sweep.sh) sequence them and wrap the `nsys` mode under
`nsys profile`. Run from the repo root after `source env.sh`:

    python scripts/profile_policy.py --system pi05_openpi_torch --config-name pi05_libero \
        --checkpoint random --gpu 6 --prompt-len 200 --num-steps 10 --mode time \
        --output out/run/profile
"""

import os
import sys

import tyro

# make `import edge_robotics` work even without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edge_robotics.cli import Config, run  # noqa: E402


def main() -> None:
    config = tyro.cli(Config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    # openpi transitively imports JAX (config helpers only); keep it off the GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    run(config)


if __name__ == "__main__":
    main()
