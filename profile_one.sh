#!/usr/bin/env bash
# =============================================================================
# profile_one.sh — profile ONE system at ONE config / prompt_len, end to end.
#
# The model runs ONLY in its real graphs-on form; the per-component split comes
# from a single Nsight Systems (nsys) capture. Because nsys must wrap the whole
# process (and CUPTI perturbs wall timing), the work is split into cheap stages
# this script sequences:
#
#   time        -> pristine wall: segmented E2E (breakdown vehicle) + per-component
#                  standalone                            ->  <out>.timing.json
#   time-native -> the DEPLOYED native single-compile E2E, in its OWN process. This is the
#                  HEADLINE latency (CROSS_CHECK_NATIVE=0 skips it -> headline falls back to the
#                  segmented wall, which is NOT deployment-faithful)  -> <out>.timing-native.json
#   nsys        -> bracket one steady-state segmented run UNDER `nsys profile`
#                                                        ->  <out>.nsys-rep
#   parse       -> NVTX per-phase split + kernel buckets ->  <out>.breakdown.json
#   report      -> merge -> <out>.json / <out>.csv + config.json (full reproducibility
#                  manifest: hardware/driver, dtype, model cfg, pkg versions, git) (+ summary)
#
# Usage:
#   ./profile_one.sh <system> <config_name> <prompt_len|native> <gpu> <out_dir>
#
# Example:
#   ./profile_one.sh openpi_torch pi05_libero native 6 out/libero_torch   # native max_token_len
#   ./profile_one.sh openpi_torch pi05_droid 200 6 out/droid_L200         # override prompt_len
#
# Env overrides: CHECKPOINT NUM_STEPS WARMUP ITERS BATCH_SIZE CROSS_CHECK_NATIVE
#   OPENPI_TORCH_COMPILE_MODE        (segmented breakdown vehicle; default max-autotune = deployed
#                                     fidelity. Set reduce-overhead only for fast iteration.)
#   OPENPI_TORCH_NATIVE_COMPILE_MODE (native headline; default max-autotune = openpi's deployed mode)
# =============================================================================
set -euo pipefail

SYSTEM="${1:?usage: profile_one.sh <system> <config_name> <prompt_len|native> <gpu> <out_dir>}"
CONFIG_NAME="${2:?missing config_name}"
PROMPT_LEN="${3:?missing prompt_len: an integer, or the word native for the config trained value}"
GPU="${4:?missing gpu}"
OUT_DIR="${5:?missing out_dir}"

# prompt_len="native"/"-"/"" -> omit --prompt-len so the CLI uses the config's native max_token_len.
PROMPT_ARGS=()
if [ "$PROMPT_LEN" != "native" ] && [ "$PROMPT_LEN" != "-" ] && [ -n "$PROMPT_LEN" ]; then
  PROMPT_ARGS=(--prompt-len "$PROMPT_LEN")
fi

CHECKPOINT="${CHECKPOINT:-random}"
NUM_STEPS="${NUM_STEPS:-10}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
[ -f "$REPO_ROOT/env.sh" ] && source "$REPO_ROOT/env.sh"

# Locate nsys: PATH first, then the CUDA 12.x toolkit default.
NSYS="$(command -v nsys || true)"
for c in /usr/local/cuda-12.6/bin/nsys /usr/local/cuda/bin/nsys; do
  [ -z "$NSYS" ] && [ -x "$c" ] && NSYS="$c"
done
if [ -z "$NSYS" ]; then
  echo "ERROR: nsys not found on PATH or in /usr/local/cuda*/bin." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/profile"
LOG="$OUT_DIR/run.log"
: > "$LOG"

run_py() {  # run profile_policy.py with the shared flags + extra args; tee to the log
  python scripts/profile_policy.py \
    --system "$SYSTEM" --config-name "$CONFIG_NAME" --checkpoint "$CHECKPOINT" --gpu "$GPU" \
    ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"} --num-steps "$NUM_STEPS" --warmup "$WARMUP" --iters "$ITERS" \
    --batch-size "$BATCH_SIZE" --output "$OUT" "$@" 2>&1 | tee -a "$LOG"
}

echo "[profile_one] $SYSTEM / $CONFIG_NAME / L=$PROMPT_LEN on GPU $GPU -> $OUT_DIR"

# --- 1. pristine wall timing (NOT under nsys) --------------------------------
echo "[profile_one] (stage: time — segmented breakdown vehicle + components)"
run_py --mode time

# --- 1b. native single-compile E2E = the HEADLINE latency (SEPARATE process) -
# openpi's native fused graph thrashes the cudagraph pool if run alongside the per-phase graphs, so
# it gets its own process. This is the deployed/headline number; CROSS_CHECK_NATIVE=0 skips it but
# then the headline falls back to the (slower, non-deployment-faithful) segmented wall.
if [ "${CROSS_CHECK_NATIVE:-1}" = "1" ]; then
  echo "[profile_one] (stage: time-native — DEPLOYED headline, separate process)"
  run_py --mode time-native || echo "[profile_one] native headline run failed (non-fatal); headline will fall back to segmented"
else
  echo "[profile_one] WARNING: CROSS_CHECK_NATIVE=0 — skipping the deployed native run; headline will be the segmented wall (NOT deployment-faithful)."
fi

# --- 2. nsys capture of one steady-state segmented run -----------------------
# --capture-range=cudaProfilerApi: nsys records ONLY between cudaProfilerStart/Stop (which the
# python brackets), excluding model load, torch.compile, cudagraph capture and warmup.
echo "[profile_one] (stage: nsys capture)"
# --cuda-graph-trace=node: without it nsys treats each CUDA graph as ONE opaque op, so the kernel
# summary (cuda_gpu_kern_sum) sees almost nothing. =node records per-kernel (per graph-node) timing
# INSIDE graphs, which is what makes the kernel-family buckets and the attributed_frac denominator
# meaningful. (NVTX GPU projection works either way.)
CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cpu "$NSYS" profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  -o "$OUT" --force-overwrite=true \
  python scripts/profile_policy.py \
    --system "$SYSTEM" --config-name "$CONFIG_NAME" --checkpoint "$CHECKPOINT" --gpu "$GPU" \
    ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"} --num-steps "$NUM_STEPS" --warmup "$WARMUP" --iters "$ITERS" \
    --batch-size "$BATCH_SIZE" --output "$OUT" --mode nsys 2>&1 | tee -a "$LOG"

# --- 3. parse the report -----------------------------------------------------
echo "[profile_one] (stage: parse)"
run_py --mode parse --nsys-rep "$OUT.nsys-rep"

# --- 4. merge + summarize ----------------------------------------------------
echo "[profile_one] (stage: report)"
run_py --mode report

echo "[profile_one] done -> $OUT.json  $OUT.csv  $OUT_DIR/config.json  (log: $LOG)"
