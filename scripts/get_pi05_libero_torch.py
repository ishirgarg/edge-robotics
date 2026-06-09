#!/usr/bin/env python
"""Fetch the REAL pi05_libero checkpoint and convert it to a PyTorch safetensors.

openpi ships pi-0.5 weights as a JAX/orbax (OCDBT) checkpoint in the PUBLIC
`gs://openpi-assets` bucket. This:
  1. downloads `gs://openpi-assets/checkpoints/pi05_libero` (params/ + assets/) over
     anonymous gcsfs (no gsutil/gcloud needed) into OPENPI_DATA_HOME, then
  2. runs openpi's own `convert_jax_model_to_pytorch.py` to produce a `model.safetensors`
     that `PI0Pytorch(Pi0Config(...pi05_libero...))` loads with `load_state_dict(strict=False)`.

The converted dir (model.safetensors + config.json + assets/norm_stats) is what the
`pi05_openpi_torch` profiler system loads when `--checkpoint <that_dir>` is given. JAX runs on
CPU only (JAX_PLATFORMS=cpu) — it is used purely to restore orbax params, never the GPU.

Usage:
    python scripts/get_pi05_libero_torch.py            # default out under OPENPI_DATA_HOME
    python scripts/get_pi05_libero_torch.py --out /scratch/.../pi05_libero_torch
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

GS_CKPT = "gs://openpi-assets/checkpoints/pi05_libero"
CONFIG_NAME = "pi05_libero"


def main() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # JAX restores orbax params on CPU only
    out = None
    args = sys.argv[1:]
    if "--out" in args:
        out = args[args.index("--out") + 1]

    repo = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "openpi" / "src"))

    import openpi.shared.download as download

    print(f"[get_pi05] downloading {GS_CKPT} (params + assets) via anonymous gcsfs ...", flush=True)
    # Anonymous read of the public bucket; lands under OPENPI_DATA_HOME/openpi-assets/...
    local = download.maybe_download(GS_CKPT, gs={"token": "anon"})
    local = pathlib.Path(local)
    print(f"[get_pi05] downloaded checkpoint dir -> {local}", flush=True)
    params_dir = local / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"expected params/ under {local}; got {sorted(p.name for p in local.iterdir())}")

    if out is None:
        out = str(local.parent / "pi05_libero_torch")
    out_path = pathlib.Path(out)

    done_marker = out_path / "model.safetensors"
    if done_marker.exists():
        print(f"[get_pi05] converted model already present at {out_path}; skipping conversion.", flush=True)
    else:
        convert = repo / "openpi" / "examples" / "convert_jax_model_to_pytorch.py"
        cmd = [
            sys.executable, str(convert),
            "--checkpoint_dir", str(local),
            "--config_name", CONFIG_NAME,
            "--output_path", str(out_path),
            "--precision", "float32",
        ]
        print(f"[get_pi05] converting JAX -> PyTorch:\n  {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(repo / "openpi"))

    # openpi's converter looks for assets in checkpoint_dir.parent/assets; pi05_libero keeps them
    # INSIDE the ckpt dir, so copy norm_stats over explicitly if the converter missed them.
    assets_src = local / "assets"
    assets_dst = out_path / "assets"
    if assets_src.exists() and not assets_dst.exists():
        shutil.copytree(assets_src, assets_dst)
        print(f"[get_pi05] copied assets (norm_stats) -> {assets_dst}", flush=True)

    print(f"[get_pi05] DONE. Converted torch checkpoint at: {out_path}", flush=True)
    print(f"[get_pi05] profile with: --checkpoint {out_path}", flush=True)


if __name__ == "__main__":
    main()
