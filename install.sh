#!/usr/bin/env bash
# Finish installing the edge-robotics environment AFTER `conda env create -f environment.yml`
# and `conda activate edge-robotics`.
#
# openpi and our package are installed editable + --no-deps because all required
# third-party deps are already pinned in environment.yml. --no-deps is what keeps
# openpi's unused/heavy deps (lerobot, tensorflow, gym-aloha) out of the env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Send pip's build/cache to /scratch (home is only ~20GB here).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/ishirgarg/pip-cache}"

# openpi (JAX inference path) is cloned in-tree and installed editable; it ships as a checkout,
# not via PyPI. Clone if missing (--no-deps at install time keeps its heavy unused deps out).
if [[ ! -d "$HERE/openpi" ]]; then
  echo ">> Cloning openpi (JAX inference path)"
  GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git openpi
fi

# realtime-vla (PyTorch+Triton backend) is vendored in-tree and imported via sys.path; it ships as
# copy-in files, not a pip package. Clone if missing (no extra deps beyond torch/triton/transformers,
# all already in environment.yml).
if [[ ! -d "$HERE/realtime-vla" ]]; then
  echo ">> Cloning realtime-vla (PyTorch+Triton backend)"
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/dexmal/realtime-vla.git realtime-vla
fi

echo ">> Installing openpi (editable, no-deps)"
pip install --no-deps -e ./openpi

echo ">> Installing edge_robotics (editable, no-deps)"
pip install --no-deps -e .

echo ">> Sanity import check (JAX inference path, no lerobot/tensorflow needed)"
python - <<'PY'
import openpi.models.pi0  # noqa: F401
import openpi.models.model  # noqa: F401
import openpi.shared.download  # noqa: F401
import openpi.shared.nnx_utils  # noqa: F401
print("openpi import OK")
PY

echo ">> Sanity import check (realtime-vla PyTorch+Triton backend)"
python - <<'PY'
import torch, triton  # noqa: F401
print(f"torch {torch.__version__}, triton {triton.__version__}, cuda available: {torch.cuda.is_available()}")
PY

echo ">> Done. Activate with: conda activate edge-robotics"
