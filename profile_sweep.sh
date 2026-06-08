#!/usr/bin/env bash
# =============================================================================
# profile_sweep.sh — sweep profile_one.sh over backends x prompt lengths.
#
# Reuses profile_one.sh (one system / one config / one prompt_len) as the unit
# of work, so the sweep is just a loop. Edit the knobs below. All runs go to one
# free GPU SEQUENTIALLY (each grabs most of VRAM; concurrent runs would contend
# and pollute timings).
#
# Usage:   source env.sh   # (this script also sources it)
#          ./profile_sweep.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- knobs (edit here) -------------------------------------------------------
# Both torch backends. pi05_libero is the config that lines up across both:
# realtime-vla's random path forces discrete_state_input=False (chunk_size=10).
SYSTEMS=(pi05_openpi_torch pi05_realtimevla)
CONFIG_NAME="pi05_libero"
# Realistic inference prompt lengths to sweep (max_token_len / #language tokens).
PROMPT_LENS=(16 32 64)
GPU="${GPU:-6}"

export CHECKPOINT="${CHECKPOINT:-random}"
export NUM_STEPS="${NUM_STEPS:-10}"
export WARMUP="${WARMUP:-3}"
export ITERS="${ITERS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
# Segmented breakdown: reduce-overhead compiles fast (the %-split is mode-insensitive).
export OPENPI_TORCH_COMPILE_MODE="${OPENPI_TORCH_COMPILE_MODE:-reduce-overhead}"
# Native cross-check: openpi's repo default (max-autotune) = the faithful fully-optimized baseline.
# (realtime-vla's native is its own captured graph — the benchmark.py path.) Set CROSS_CHECK_NATIVE=0
# to skip the native baseline entirely and avoid its slow max-autotune compile during big sweeps.
export OPENPI_TORCH_NATIVE_COMPILE_MODE="${OPENPI_TORCH_NATIVE_COMPILE_MODE:-max-autotune}"

OUT_ROOT="${OUT_ROOT:-out/sweep_${CONFIG_NAME}}"

# shellcheck source=/dev/null
[ -f "$REPO_ROOT/env.sh" ] && source "$REPO_ROOT/env.sh"

echo "[sweep] systems=(${SYSTEMS[*]}) config=$CONFIG_NAME prompt_lens=(${PROMPT_LENS[*]}) gpu=$GPU"
echo "[sweep] out_root=$OUT_ROOT compile=$OPENPI_TORCH_COMPILE_MODE"

declare -a SUMMARY=()
for system in "${SYSTEMS[@]}"; do
  for L in "${PROMPT_LENS[@]}"; do
    out_dir="$OUT_ROOT/${system}_L${L}"
    echo ""
    echo "================================================================="
    echo "[sweep] $system  L=$L  -> $out_dir"
    echo "================================================================="
    rc=0
    ./profile_one.sh "$system" "$CONFIG_NAME" "$L" "$GPU" "$out_dir" || rc=$?
    SUMMARY+=("$system L=$L exit=$rc -> $out_dir/profile.json")
  done
done

echo ""
echo "[sweep] done:"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
