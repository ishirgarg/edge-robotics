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

# --- JAX (openpi imports it for config helpers only; keep it off the GPU) ---
export JAX_PLATFORMS=cpu

# --- torch.compile / Triton PERSISTENT caches (compile once, reuse forever) ---
# torch.compile recompiles per PROCESS, and the profiler launches many (time/nsys/sweep). These
# on-disk caches make the EXPENSIVE work — Inductor codegen, Triton kernel builds, and especially
# max-autotune's autotuning — compile-once-per-shape and reuse across every later process/run:
#   * TORCHINDUCTOR_CACHE_DIR  : generated kernels + the max-autotune autotune results
#   * TRITON_CACHE_DIR         : compiled Triton kernel binaries
#   * TORCHINDUCTOR_FX_GRAPH_CACHE=1 : cache the compiled FX graph itself
# (CUDA-graph *capture* still happens per process — that's cheap, ~ms — and a different prompt_len is
#  a different shape, so it autotunes once per shape, then hits the cache on every subsequent run.)
export TORCHINDUCTOR_CACHE_DIR=/scratch/ishirgarg/torch_cache/inductor
export TRITON_CACHE_DIR=/scratch/ishirgarg/torch_cache/triton
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "[env.sh] edge-robotics env active; caches -> /scratch; OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
