#!/usr/bin/env python
"""Fetch a real pi0/pi05 checkpoint and convert it to PyTorch safetensors (ANY registered config).

Maps a config name to its public checkpoint at
`gs://openpi-assets/checkpoints/<config_name>` (override with --gs), downloads via anonymous gcsfs,
runs openpi's `convert_jax_model_to_pytorch.py` (config-driven, so it works for every pi0/pi05 ×
dataset config), and copies norm_stats. The converted dir loads via `--checkpoint <dir>` on the
`openpi_torch` backend and the attention study. JAX restores orbax params on CPU only.

Usage:
    python scripts/get_pi0_torch.py --config-name pi05_droid
    python scripts/get_pi0_torch.py --config-name pi0_aloha_sim          # task-tuned ALOHA
    python scripts/get_pi0_torch.py --config-name pi0_droid --out /scratch/.../pi0_droid_torch
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True, help="openpi config name (= checkpoint dir name)")
    ap.add_argument("--gs", default=None, help="override gs:// dir (default .../checkpoints/<config>)")
    ap.add_argument("--out", default=None, help="output torch dir (default <cache>/<config>_torch)")
    ap.add_argument("--precision", default="float32")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # restore orbax params on CPU only, never the GPU
    repo = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "openpi" / "src"))

    import openpi.shared.download as download

    gs = args.gs or f"gs://openpi-assets/checkpoints/{args.config_name}"
    print(f"[get_pi0] downloading {gs} (params + assets) via anonymous gcsfs ...", flush=True)
    local = pathlib.Path(download.maybe_download(gs, gs={"token": "anon"}))
    print(f"[get_pi0] downloaded checkpoint dir -> {local}", flush=True)
    if not (local / "params").exists():
        raise FileNotFoundError(f"expected params/ under {local}; got {sorted(p.name for p in local.iterdir())}")

    out = pathlib.Path(args.out or (local.parent / f"{args.config_name}_torch"))
    if (out / "model.safetensors").exists():
        print(f"[get_pi0] converted model already present at {out}; skipping conversion.", flush=True)
    else:
        convert = repo / "openpi" / "examples" / "convert_jax_model_to_pytorch.py"
        cmd = [sys.executable, str(convert), "--checkpoint_dir", str(local),
               "--config_name", args.config_name, "--output_path", str(out), "--precision", args.precision]
        print(f"[get_pi0] converting JAX -> PyTorch:\n  {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(repo / "openpi"))

    # openpi's converter looks for assets in checkpoint_dir.parent/assets; some configs keep them
    # INSIDE the ckpt dir, so copy norm_stats over explicitly if the converter missed them.
    assets_src, assets_dst = local / "assets", out / "assets"
    if assets_src.exists() and not assets_dst.exists():
        shutil.copytree(assets_src, assets_dst)
        print(f"[get_pi0] copied assets (norm_stats) -> {assets_dst}", flush=True)

    print(f"[get_pi0] DONE. Converted torch checkpoint at: {out}", flush=True)
    print(f"[get_pi0] profile with: --checkpoint {out}", flush=True)


if __name__ == "__main__":
    main()
