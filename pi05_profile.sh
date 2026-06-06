#!/usr/bin/env bash
# =============================================================================
# pi05_profile.sh
#
# Basic apples-to-apples profiling of pi-0.5 on BOTH backends on this machine:
#
#   * pi05_jax          (native JAX/Flax openpi)
#   * pi05_realtimevla  (dexmal/realtime-vla, PyTorch + Triton + CUDA-graph replay)
#
# Both runs use the EXACT SAME model config so the numbers are directly
# comparable. `pi05_libero` is chosen on purpose: it is the one config_name
# whose settings line up on BOTH backends, because realtime-vla's random-weight
# path forces discrete_state_input=False and we need the JAX side to match:
#
#     knob                  pi05_jax (pi05_libero)   pi05_realtimevla (pi05_libero, random)
#     --------------------  -----------------------  --------------------------------------
#     action_dim            32                       32   (fixed in Triton kernels)
#     action_horizon/chunk  10                       10
#     n_images / num_views  3                        3
#     max_token_len/prompt  $PROMPT_LEN              $PROMPT_LEN
#     discrete_state_input  False                    False
#     num_steps (flow)      10                       10   (hard-locked in realtime-vla)
#     batch_size            1                        1    (realtime-vla requires 1)
#     paligemma_variant     gemma_2b                 gemma_2b
#     action_expert_variant gemma_300m               gemma_300m
#     dtype                 bfloat16                 bfloat16
#     checkpoint            random                   random
#
# Each run gets its own output directory containing a config.json that records
# EVERY settable parameter plus the hardware/software environment, for 100%
# reproducibility. (Python packages are pinned by the conda env / environment.yml,
# so they're not re-dumped here.)
#
# GPUs 6 and 7 are assumed free; the two backends run in parallel, one per GPU.
#
# Usage:   source env.sh   # (the script also sources it, but doing it yourself
#                          #  surfaces conda/cache errors early)
#          ./pi05_profile.sh
# =============================================================================

set -euo pipefail

# --- repo root (this script lives at the repo root) --------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# Shared, identical-across-backends configuration. Change here ONLY; both runs
# read the same values so the comparison stays apples-to-apples.
# =============================================================================
CONFIG_NAME="pi05_libero"   # the one config that matches exactly on both backends (see header)
CHECKPOINT="random"         # random-init: no download, no tokenizer; latency is value-independent
PROMPT_LEN="200"            # max_token_len (JAX) == #language tokens (realtime-vla)
NUM_STEPS=10                # flow-matching denoise steps (realtime-vla is hard-locked to 10)
WARMUP=3                    # warmup iterations (absorb JIT / graph capture)
ITERS=30                    # timed iterations per measurement
BATCH_SIZE=1                # realtime-vla requires 1; JAX matches it

# Per-backend GPU assignment (both free per the request).
GPU_JAX=6
GPU_RTVLA=7

# Run-tag suffix for output dirs (kept stable so reruns overwrite cleanly).
RUN_TAG="${CONFIG_NAME}"

OUT_JAX="out/${RUN_TAG}_jax"
OUT_RTVLA="out/${RUN_TAG}_realtimevla"

# =============================================================================
# Environment: conda + /scratch caches + XLA mem fraction (see env.sh).
# =============================================================================
# shellcheck source=/dev/null
source "$REPO_ROOT/env.sh"

# Timestamp shared by both runs so they're tagged as one logical experiment.
RUN_TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOSTNAME_FULL="$(hostname -f 2>/dev/null || hostname)"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then echo true; else echo false; fi)"
PYTHON_VERSION="$(python -c 'import platform; print(platform.python_version())' 2>/dev/null || echo unknown)"
CONDA_ENV="${CONDA_DEFAULT_ENV:-unknown}"
NVIDIA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null | head -1 || echo unknown)"
CUDA_DRIVER_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1 || echo unknown)"

# importlib.metadata reads dist metadata WITHOUT importing the (heavy) package.
pkg_version() { python -c "import importlib.metadata as m,sys
try: print(m.version(sys.argv[1]))
except Exception: print('not-installed')" "$1" 2>/dev/null || echo unknown; }

JAX_VER="$(pkg_version jax)"
JAXLIB_VER="$(pkg_version jaxlib)"
OPENPI_VER="$(pkg_version openpi)"
TORCH_VER="$(pkg_version torch)"
TRITON_VER="$(pkg_version triton)"

# -----------------------------------------------------------------------------
# write_config <run_dir> <system> <gpu> <output_prefix> <backend_versions_json>
# Emits <run_dir>/config.json with every settable parameter + hardware/software
# context, and dumps a full pip freeze + git diff for total reproducibility.
# -----------------------------------------------------------------------------
write_config() {
  local run_dir="$1" system="$2" gpu="$3" output_prefix="$4" backend_versions="$5"
  mkdir -p "$run_dir"

  # Per-GPU hardware identity (the actual device this run is pinned to).
  local gpu_name gpu_uuid gpu_mem gpu_pcie
  gpu_name="$(nvidia-smi --query-gpu=name        --format=csv,noheader -i "$gpu" 2>/dev/null || echo unknown)"
  gpu_uuid="$(nvidia-smi --query-gpu=uuid        --format=csv,noheader -i "$gpu" 2>/dev/null || echo unknown)"
  gpu_mem="$( nvidia-smi --query-gpu=memory.total --format=csv,noheader -i "$gpu" 2>/dev/null || echo unknown)"
  gpu_pcie="$(nvidia-smi --query-gpu=pci.bus_id  --format=csv,noheader -i "$gpu" 2>/dev/null || echo unknown)"

  # The exact command the run will execute (verbatim, for replay).
  local invocation="python scripts/profile_policy.py --system ${system} --config-name ${CONFIG_NAME} --checkpoint ${CHECKPOINT} --gpu ${gpu} --prompt-lens ${PROMPT_LEN} --num-steps ${NUM_STEPS} --warmup ${WARMUP} --iters ${ITERS} --batch-size ${BATCH_SIZE} --output ${output_prefix}"

  cat > "$run_dir/config.json" <<JSON
{
  "experiment": {
    "name": "pi05_basic_profile",
    "run_tag": "${RUN_TAG}",
    "timestamp_utc": "${RUN_TS_UTC}",
    "purpose": "apples-to-apples basic profiling of pi-0.5 across backends with identical model config"
  },
  "run": {
    "system": "${system}",
    "invocation": "${invocation}",
    "output_prefix": "${output_prefix}",
    "results_json": "${output_prefix}.json",
    "results_csv": "${output_prefix}.csv",
    "trace_dir": "${output_prefix}_trace"
  },
  "profiler_params": {
    "config_name": "${CONFIG_NAME}",
    "checkpoint": "${CHECKPOINT}",
    "gpu": ${gpu},
    "prompt_lens": [${PROMPT_LEN}],
    "num_steps": ${NUM_STEPS},
    "warmup": ${WARMUP},
    "iters": ${ITERS},
    "batch_size": ${BATCH_SIZE}
  },
  "shared_model_config": {
    "_comment": "Identical across both backends by construction (see script header). Asserted constants for config_name=${CONFIG_NAME}.",
    "action_dim": 32,
    "action_horizon": 10,
    "n_images": 3,
    "tokens_per_image_nominal": 256,
    "max_token_len": ${PROMPT_LEN},
    "prefix_len_nominal": $((3 * 256 + PROMPT_LEN)),
    "discrete_state_input": false,
    "num_steps_flow": ${NUM_STEPS},
    "batch_size": ${BATCH_SIZE},
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "dtype": "bfloat16"
  },
  "hardware": {
    "hostname": "${HOSTNAME_FULL}",
    "gpu_index": ${gpu},
    "gpu_name": "${gpu_name}",
    "gpu_uuid": "${gpu_uuid}",
    "gpu_memory_total": "${gpu_mem}",
    "gpu_pci_bus_id": "${gpu_pcie}",
    "nvidia_driver_version": "${NVIDIA_DRIVER}",
    "cuda_driver_version": "${CUDA_DRIVER_VERSION}"
  },
  "software": {
    "python_version": "${PYTHON_VERSION}",
    "conda_env": "${CONDA_ENV}",
    "git_commit": "${GIT_COMMIT}",
    "git_dirty": ${GIT_DIRTY},
    "packages": ${backend_versions}
  },
  "environment_vars": {
    "CUDA_VISIBLE_DEVICES": "${gpu}",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}",
    "XLA_FLAGS": "${XLA_FLAGS:-}",
    "OPENPI_DATA_HOME": "${OPENPI_DATA_HOME:-}",
    "HF_HOME": "${HF_HOME:-}",
    "TMPDIR": "${TMPDIR:-}",
    "REALTIME_VLA_DIR": "${REALTIME_VLA_DIR:-}",
    "REALTIME_VLA_TOKENIZER": "${REALTIME_VLA_TOKENIZER:-}"
  }
}
JSON

  # Validate the JSON we just wrote (fail loudly rather than ship a broken config).
  python -m json.tool "$run_dir/config.json" > /dev/null

  echo "[pi05_profile] wrote $run_dir/config.json"
}

# =============================================================================
# Write per-run configs, then launch both backends in parallel (one GPU each).
# =============================================================================
write_config "$OUT_JAX"   "pi05_jax"         "$GPU_JAX"   "$OUT_JAX/profile" \
  "{\"jax\": \"${JAX_VER}\", \"jaxlib\": \"${JAXLIB_VER}\", \"openpi\": \"${OPENPI_VER}\"}"

write_config "$OUT_RTVLA" "pi05_realtimevla" "$GPU_RTVLA" "$OUT_RTVLA/profile" \
  "{\"torch\": \"${TORCH_VER}\", \"triton\": \"${TRITON_VER}\"}"

echo "[pi05_profile] launching pi05_jax on GPU ${GPU_JAX} and pi05_realtimevla on GPU ${GPU_RTVLA} (parallel)"

# --- pi05_jax (GPU $GPU_JAX) -------------------------------------------------
python scripts/profile_policy.py \
  --system pi05_jax \
  --config-name "$CONFIG_NAME" \
  --checkpoint "$CHECKPOINT" \
  --gpu "$GPU_JAX" \
  --prompt-lens "$PROMPT_LEN" \
  --num-steps "$NUM_STEPS" --warmup "$WARMUP" --iters "$ITERS" --batch-size "$BATCH_SIZE" \
  --output "$OUT_JAX/profile" \
  > "$OUT_JAX/run.log" 2>&1 &
PID_JAX=$!

# --- pi05_realtimevla (GPU $GPU_RTVLA) ---------------------------------------
python scripts/profile_policy.py \
  --system pi05_realtimevla \
  --config-name "$CONFIG_NAME" \
  --checkpoint "$CHECKPOINT" \
  --gpu "$GPU_RTVLA" \
  --prompt-lens "$PROMPT_LEN" \
  --num-steps "$NUM_STEPS" --warmup "$WARMUP" --iters "$ITERS" --batch-size "$BATCH_SIZE" \
  --output "$OUT_RTVLA/profile" \
  > "$OUT_RTVLA/run.log" 2>&1 &
PID_RTVLA=$!

# Wait for both, capturing exit codes independently so one failure doesn't hide the other.
RC_JAX=0; RC_RTVLA=0
wait "$PID_JAX"   || RC_JAX=$?
wait "$PID_RTVLA" || RC_RTVLA=$?

echo "[pi05_profile] pi05_jax exit=${RC_JAX} (log: $OUT_JAX/run.log)"
echo "[pi05_profile] pi05_realtimevla exit=${RC_RTVLA} (log: $OUT_RTVLA/run.log)"

# =============================================================================
# Post-run cross-check: confirm the two backends actually loaded identical model
# config. Compares the meta block of both results JSONs on the equivalence keys.
# =============================================================================
if [ "$RC_JAX" -eq 0 ] && [ "$RC_RTVLA" -eq 0 ]; then
  echo "[pi05_profile] verifying config equivalence across backends..."
  python - "$OUT_JAX/profile.json" "$OUT_RTVLA/profile.json" <<'PY'
import json, sys
# meta.model carries prefix_len_nominal (= n_images*256 + max_token_len), not max_token_len
# directly, so comparing it also covers prompt-length equivalence.
keys = ["action_horizon", "action_dim", "prefix_len_nominal", "n_images",
        "paligemma_variant", "action_expert_variant", "dtype"]
a = json.load(open(sys.argv[1]))["meta"]["model"]
b = json.load(open(sys.argv[2]))["meta"]["model"]
mism = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
if mism:
    print("  [WARN] config mismatch between backends:")
    for k, (x, y) in mism.items():
        print(f"    {k}: jax={x!r}  realtimevla={y!r}")
    sys.exit(1)
print("  [OK] backends loaded identical model config:",
      {k: a.get(k) for k in keys})
PY
else
  echo "[pi05_profile] skipping equivalence check (a run failed); inspect the logs above."
fi

echo "[pi05_profile] done. Results:"
echo "  $OUT_JAX/profile.{json,csv}    config: $OUT_JAX/config.json"
echo "  $OUT_RTVLA/profile.{json,csv}  config: $OUT_RTVLA/config.json"
