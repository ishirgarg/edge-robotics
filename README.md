# edge-robotics

Profiler-driven latency study for robotics foundation models **at the edge**. v1 profiles
**pi-0.5** (native JAX/Flax, from [openpi](https://github.com/Physical-Intelligence/openpi))
on a single GPU and reports a per-phase breakdown:

| Vision (ms / %) | VLM (ms / %) | Action (ms / %) | E2E (ms) | Freq (Hz) |

The model is run **unmodified** (the same `module_jit(sample_actions)` path `openpi.Policy`
uses) on dummy, spec-conformant inputs. Phase timings come from the JAX/CUDA profiler and a
non-invasive `num_steps` regression — not from editing the model's forward pass.

---

## How latency is measured (read this)

One pi-0.5 inference = **Vision** (SigLIP image encode) → **VLM** (Gemma prefill, builds the
KV-cache) → **Action** (the `jax.lax.while_loop` flow-matching denoise loop, `num_steps` iters).

All timing is **device-synced**: every measured region ends with `jax.block_until_ready`, so we
time real GPU completion, not async dispatch (this is *not* a naive `time.time()` call). Warmup
iterations absorb JIT compilation.

- **E2E (ms)** — median steady-state wall time of the full `sample_actions`. **Freq = 1000/E2E.**
- **Action** — from a **`num_steps` regression** (`edge_robotics/profiling/regression.py`):
  time inference at several denoise-step counts; only the denoise loop repeats, so
  `time(k) = intercept + slope·k`. **Action = slope · num_steps**, **Vision+VLM = intercept**.
  Uses only the public `sample_actions(num_steps=k)` API — zero internal poking.
- **Vision vs VLM** — the optimized, fused XLA trace does **not** label kernels as img-vs-llm
  (they appear as `gemm_fusion_dot_*` / generic names), so it can't separate them on its own.
  We split them with a minimal **vision probe**: directly time the public SigLIP encoder
  (`model.PaliGemma.img`) over the camera images, then **VLM = (Vision+VLM intercept) − Vision**.
  Toggle with `--no-probe-vision` (then Vision+VLM is reported combined).
- **JAX profiler trace** (`--jax-trace`) — captures a TensorBoard/XProf trace (CUDA kernels via
  CUPTI included) as an artifact and a **total-GPU-time cross-check** vs E2E. View with
  `tensorboard --logdir <trace_dir>`. (Parsing is robust: it reads the `*.trace.json.gz`.)
- **CUDA profiler** (`--nsys`) — prints the `nsys profile …` command to wrap the run for a
  kernel-level timeline (nsys is not installed on this box; the JAX trace already has kernels).

The table prints, per row, the method used and the `sum_of_phases` vs measured-E2E discrepancy.

---

## Hardware note

The request said "RTX 6000"; this machine actually has **8× NVIDIA RTX A6000 (48 GB)**.
We pin to one GPU via `--gpu` (indices **4–7 are idle**). Profiling runs on the A6000.

## Layout note (disk)

Home here is only ~20 GB, so the conda env, checkpoints, and pip cache live on **`/scratch`**
(see `env.sh`). Project code stays in this folder. The `openpi/` dependency is cloned in-tree.

---

## Setup (conda only — no uv)

```bash
cd /home/eecs/ishirgarg/edge-robotics

# 1. clone the openpi dependency in-tree (must exist before env create)
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
    https://github.com/Physical-Intelligence/openpi.git openpi

# 2. create the env (python 3.11 + JAX/torch stack; skips openpi's heavy unused deps)
GIT_LFS_SKIP_SMUDGE=1 conda env create -f environment.yml

# 3. finish: editable openpi + edge_robotics (--no-deps) + sanity import
conda activate edge-robotics
bash install.sh
```

`environment.yml` pins the subset of openpi's deps needed for JAX inference and deliberately
omits `lerobot`/`tensorflow`/`gym-aloha`/`dlimp` (only used for data-loading/training).
`install.sh` does `pip install --no-deps -e ./openpi` so those omitted deps aren't pulled.

---

## Usage

```bash
source env.sh   # activate env + point caches (checkpoints, etc.) at /scratch

# prompt-length sweep on idle GPU 4, real DROID checkpoint:
python scripts/profile_policy.py \
  --system pi05_jax \
  --config-name pi05_droid \
  --checkpoint gs://openpi-assets/checkpoints/pi05_droid \
  --gpu 4 \
  --prompt-lens 16 32 64 128 200 \
  --num-steps 10 --warmup 3 --iters 30 --batch-size 1 \
  --regression-steps 1 2 4 8 \
  --output out/pi05_droid \
  --jax-trace
```

Fast plumbing test with no download (tiny dummy model, random weights):

```bash
python scripts/profile_policy.py --system pi05_jax --config-name debug_pi05 \
  --checkpoint random --gpu 4 --prompt-lens 32 128 --num-steps 10 \
  --warmup 2 --iters 5 --regression-steps 1 2 4 --output out/smoke
```

### Key flags

| flag | meaning |
|------|---------|
| `--config-name` | `pi05_droid` (ah=15), `pi05_libero` (ah=10), `pi05_base`/`pi05_aloha` (ah=50), `debug_pi05` (tiny) |
| `--checkpoint` | `gs://…` URI, local path, or `random` (random-init, no download) |
| `--gpu` | CUDA index to pin (4–7 idle here) |
| `--prompt-lens` | `max_token_len` values to sweep (the pi-0.5 prompt-length knob) |
| `--num-steps` | flow-matching denoise steps for the headline E2E |
| `--regression-steps` | step counts for the Action regression |
| `--jax-trace` / `--no-jax-trace` | capture + cross-check a JAX profiler trace |
| `--probe-vision` / `--no-probe-vision` | split Vision from VLM via the SigLIP probe |
| `--nsys` / `--no-nsys` | print the Nsight Systems command |

### Outputs

- stdout: the 5-metric table (one row per prompt length) + per-row method/notes.
- `<output>.json`: full stats (mean/median/p50/p90/p99/std), regression fit, trace summary, metadata.
- `<output>.csv`: one row per `(prompt_len, phase)` for plotting.
- `<output>_trace/L<n>/…`: JAX traces (with `--jax-trace`); view via `tensorboard --logdir <output>_trace`.

Why prompt length matters: it sets the padded token-sequence length (`max_token_len`); the LLM
processes all padded positions, so larger prompts grow the VLM prefill (~quadratic in
`768 + max_token_len`) and each Action step (attends over the cached prefix), while Vision stays
flat — surfacing how the bottleneck shifts.

---

## Extending (modular by file)

- **New system**: add `edge_robotics/systems/<name>.py` implementing `ProfiledSystem` and wire it
  into the tiny dispatch in `cli._make_system`.
- **Network simulation**: see `network/` (stub) — wrap openpi's websocket client/server.
- The profiling primitives (`profiling/{walltime,regression,jax_profiler,cuda_profiler}.py`) are
  system-agnostic and reusable.
