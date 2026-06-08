#!/usr/bin/env bash
# One-shot setup for the edge-robotics environment on a fresh machine.
# Assumes ONLY that this repo is cloned and `conda` is on PATH. Run from anywhere:
#
#     bash install.sh
#
# It will, idempotently:
#   1. clone openpi in-tree (MUST exist before env create — environment.yml
#      installs `-e ./openpi/packages/openpi-client`).
#   2. clone realtime-vla in-tree (PyTorch+Triton backend, imported via sys.path).
#   3. create the `edge-robotics` conda env from environment.yml (if missing).
#   4. install openpi + edge_robotics editable + --no-deps (deps are pinned in
#      environment.yml; --no-deps keeps openpi's heavy unused deps — lerobot,
#      tensorflow, gym-aloha — out of the env).
#   5. sanity-check both backends import.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Send pip's build/cache to /scratch (home is only ~20GB here). Override as needed on a new box.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/ishirgarg/pip-cache}"

ENV_NAME="edge-robotics"

# --- conda must be available; source its shell hooks so `conda activate` works in this script ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install miniforge/miniconda first." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# conda's profile script trips set -u; relax nounset only while sourcing/activating.
set +u
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

# --- 1. openpi (JAX inference path), cloned in-tree, BEFORE env create ---
if [[ ! -d "$HERE/openpi" ]]; then
  echo ">> Cloning openpi (JAX inference path)"
  GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
    https://github.com/Physical-Intelligence/openpi.git openpi
fi

# --- 2. realtime-vla (PyTorch+Triton backend), cloned in-tree ---
if [[ ! -d "$HERE/realtime-vla" ]]; then
  echo ">> Cloning realtime-vla (PyTorch+Triton backend)"
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/dexmal/realtime-vla.git realtime-vla
fi

# --- 3. create the conda env (idempotent: skip if it already exists) ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> conda env '$ENV_NAME' already exists; skipping create"
else
  echo ">> Creating conda env '$ENV_NAME' from environment.yml"
  GIT_LFS_SKIP_SMUDGE=1 conda env create -f environment.yml
fi

set +u
conda activate "$ENV_NAME"
set -u
echo ">> Activated conda env: $ENV_NAME"

# --- 4. editable installs (deps already pinned in environment.yml) ---
echo ">> Installing openpi (editable, no-deps)"
pip install --no-deps -e ./openpi

echo ">> Installing edge_robotics (editable, no-deps)"
pip install --no-deps -e .

# openpi's native PyTorch backend (PI0Pytorch) needs openpi's PATCHED transformers files copied
# over the installed transformers package; PI0Pytorch.__init__ hard-checks this and errors if absent.
echo ">> Installing openpi's patched transformers (transformers_replace) for the PyTorch backend"
TRANSFORMERS_DIR="$(python -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')"
cp -r ./openpi/src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/

# --- 5. sanity import checks ---
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

echo ">> Sanity check (openpi PyTorch backend: patched transformers installed)"
python - <<'PY'
from transformers.models.siglip import check
ok = check.check_whether_transformers_replace_is_installed_correctly()
print("transformers_replace installed correctly" if ok else "WARNING: transformers_replace NOT installed correctly")
PY

echo ">> Done. The '$ENV_NAME' env is ready."
echo ">> In new shells, run:  source env.sh   (activates the env + points caches at /scratch)"
