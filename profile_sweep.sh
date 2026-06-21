#!/usr/bin/env bash
# =============================================================================
# profile_sweep.sh — sweep profile_one.sh over (backend × config × prompt_len).
#
# Reuses profile_one.sh (one system / one config / one prompt_len) as the unit of
# work. All runs go to one free GPU SEQUENTIALLY (each grabs most of VRAM;
# concurrent runs would contend and pollute timings). Every run uses the DEPLOYED
# max-autotune fidelity by default (the headline is the native single-compile E2E).
#
# Override any knob via env (space-separated lists), e.g.:
#   CONFIGS="pi05_libero pi05_droid" PROMPT_LENS="native" GPU=3 ./profile_sweep.sh
#   SYSTEMS="openpi_torch" CONFIGS="pi05_libero" PROMPT_LENS="16 32 64" ./profile_sweep.sh
# Per-config real checkpoints (else random-init, fine for latency but not attention/eval):
#   CKPT_pi05_libero=/scratch/.../pi05_libero_torch ./profile_sweep.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- knobs (env-overridable; arrays via space-separated strings) -------------
read -ra SYSTEMS    <<< "${SYSTEMS:-openpi_torch realtime_vla}"          # backends
read -ra CONFIGS    <<< "${CONFIGS:-pi05_libero pi05_droid pi0_droid pi0_aloha}"  # model×dataset
read -ra PROMPT_LENS <<< "${PROMPT_LENS:-native}"   # 'native' = as trained (pi0=48/pi05=200), or ints
GPU="${GPU:-0}"

export CHECKPOINT="${CHECKPOINT:-random}"   # global default; per-config override via CKPT_<config>
export NUM_STEPS="${NUM_STEPS:-10}"
export WARMUP="${WARMUP:-3}"
export ITERS="${ITERS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
# BOTH compile paths default to max-autotune = openpi's deployed mode (max fidelity). The headline is
# the native E2E; the segmented breakdown is captured at the same mode so the breakdown reflects the
# deployed kernels. Override to reduce-overhead only for fast iteration (degrades breakdown fidelity).
export OPENPI_TORCH_COMPILE_MODE="${OPENPI_TORCH_COMPILE_MODE:-max-autotune}"
export OPENPI_TORCH_NATIVE_COMPILE_MODE="${OPENPI_TORCH_NATIVE_COMPILE_MODE:-max-autotune}"

OUT_ROOT="${OUT_ROOT:-out/sweep}"

# shellcheck source=/dev/null
[ -f "$REPO_ROOT/env.sh" ] && source "$REPO_ROOT/env.sh"

# realtime-vla supports ONLY pi05 configs (its Triton kernels bake the gemma sizes; RT_REGISTRY is
# pi05-only). Skip pi0_* on that backend rather than fail mid-run.
rt_supports() { case "$1" in pi05_*) return 0 ;; *) return 1 ;; esac; }

echo "[sweep] systems=(${SYSTEMS[*]}) configs=(${CONFIGS[*]}) prompt_lens=(${PROMPT_LENS[*]}) gpu=$GPU"
echo "[sweep] out_root=$OUT_ROOT compile=$OPENPI_TORCH_COMPILE_MODE (native=$OPENPI_TORCH_NATIVE_COMPILE_MODE)"

declare -a SUMMARY=()
for system in "${SYSTEMS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    if [ "$system" = "realtime_vla" ] && ! rt_supports "$config"; then
      echo "[sweep] skip $system/$config (realtime-vla is pi05-only)"
      SUMMARY+=("$system $config SKIPPED (pi05-only backend)")
      continue
    fi
    # Per-config checkpoint override: e.g. CKPT_pi05_droid=/path/to/pi05_droid_torch
    ckpt_var="CKPT_${config}"
    ckpt="${!ckpt_var:-$CHECKPOINT}"
    for L in "${PROMPT_LENS[@]}"; do
      out_dir="$OUT_ROOT/${system}__${config}__L${L}"
      echo ""
      echo "================================================================="
      echo "[sweep] $system  $config  L=$L  ckpt=$ckpt  -> $out_dir"
      echo "================================================================="
      rc=0
      CHECKPOINT="$ckpt" ./profile_one.sh "$system" "$config" "$L" "$GPU" "$out_dir" || rc=$?
      SUMMARY+=("$system $config L=$L exit=$rc -> $out_dir/profile.json")
    done
  done
done

echo ""
echo "[sweep] done:"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
