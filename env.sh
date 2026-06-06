# Source this before running the profiler:  source env.sh
# Activates the conda env and redirects all heavy caches to /scratch (home is ~20GB).

# --- conda ---
__CONDA="/scratch/ishirgarg/miniforge3"
if [ -f "$__CONDA/etc/profile.d/conda.sh" ]; then
  . "$__CONDA/etc/profile.d/conda.sh"
  conda activate edge-robotics
fi

# --- caches on /scratch (NOT on the 20GB home) ---
export PIP_CACHE_DIR=/scratch/ishirgarg/pip-cache
export OPENPI_DATA_HOME=/scratch/ishirgarg/openpi_cache   # checkpoints land here
export HF_HOME=/scratch/ishirgarg/hf_cache                # any HF tokenizer/model assets
export TMPDIR=/scratch/ishirgarg/tmp
mkdir -p "$TMPDIR"

# --- JAX/XLA ---
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "[env.sh] edge-robotics env active; caches -> /scratch; OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
