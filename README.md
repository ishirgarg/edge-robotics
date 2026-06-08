# edge-robotics

Profiler-driven latency study for robotics foundation models **at the edge**. It profiles
**pi-0.5** on a single GPU, running the model **only in its real, fully-optimized CUDA-graphs-on
form**, and attributes time to each component with **NVIDIA Nsight Systems (nsys)** — never by
re-running the model eager to peek inside it.

For each run it reports:

- **E2E (ms / Freq)** — wall latency of the full inference (graphs on).
- **Component breakdown** — Vision / VLM / Action GPU time from a single nsys capture.
- **Per-component standalone** — each component timed on its own, also fully optimized (graphs on).

Two backends (pick with `--system`):

| `--system` | what | component attribution |
|------------|------|-----------------------|
| `pi05_openpi_torch` | openpi's native **PyTorch** port (`PI0Pytorch`) | **nsys NVTX GPU projection** over per-phase CUDA graphs — clean Vision/VLM/Action split, graphs ON |
| `pi05_realtimevla` | PyTorch + Triton, from [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) (fused Triton kernels + CUDA-graph replay) | **same NVTX split** — we re-capture its single graph as three per-stage sub-graphs (bitwise-identical to native), plus kernel-family buckets |

> The JAX backend was removed. This repo now profiles the two PyTorch paths.

---

## How latency is measured (read this)

One pi-0.5 inference = **Vision** (SigLIP image encode ×3) → **VLM** (Gemma-2b prefill, builds the
KV-cache) → **Action** (flow-matching denoise loop, `num_steps` × Gemma-300m action expert).

### Why the model is restructured into per-phase graphs

openpi compiles `sample_actions` as **one** `torch.compile` unit (a single fused CUDA graph). NVTX
ranges pushed at the python level **cannot** attribute kernels *inside* a single fused graph: the
whole thing launches under one `cudaGraphLaunch` that falls outside any NVTX window opened at replay
time, so the split collapses (verified empirically — the NVTX ranges come back essentially empty).

The fix — the only thing that survives CUDA graphs — is to run the three phases as **separately
compiled / cudagraph callables glued in eager python**, with the NVTX range pushed **around** each
call (never inside a compiled region). nsys `nvtx_gpu_proj_sum` then attributes GPU device time per
phase. Cross-phase tensors (prefix embeds, the KV cache, masks, state) are `.clone()`d out of each
graph's static pool, and `cudagraph_mark_step_begin()` is called before each graph invocation, as
CUDA-graph trees require. This is what `infer_segmented` in
`edge_robotics/systems/pi05_openpi_torch.py` does.

Validated on the real model (pi05_libero, A6000, graphs on): the per-phase NVTX projection sums to
**~99%** of E2E, and the segmented path's wall time is **within noise of** openpi's native
single-compile path — so segmenting buys an attributable breakdown at no measurable latency cost.

### The numbers

- **E2E (segmented, headline)** — median device-synced **wall** time of the per-phase-graph
  inference. **Freq = 1000/E2E.** Warmup absorbs `torch.compile` + cudagraph capture.
- **E2E (native, cross-check)** — the **faithful, fully-optimized repo baseline**, exactly as each
  repo ships it: openpi → `torch.compile(sample_actions, mode="max-autotune")` (openpi's own
  `Pi0Config` default); realtime-vla → its single captured CUDA graph (`infer.forward`, the
  `benchmark.py` path). The gap vs segmented is the segmentation overhead (≈0 when both use the same
  mode). The segmented breakdown defaults to `reduce-overhead` for fast compiles
  (`OPENPI_TORCH_COMPILE_MODE`); the native baseline keeps `max-autotune`
  (`OPENPI_TORCH_NATIVE_COMPILE_MODE`) so it stays faithful to the original.
- **Component breakdown** — GPU ms/infer per phase from nsys `nvtx_gpu_proj_sum`. `%` is each
  phase's share. `attributed_frac` (share of GPU time landing in a phase) and `residual` (inter-phase
  glue: noise sampling, the `x_t+dt·v_t` update, D2D copies) are reported, not hidden.
- **Per-component standalone** — each phase timed in isolation, fully optimized (graphs on), with the
  upstream inputs precomputed once and reused.
- **Kernel-family buckets** (`pi05_realtimevla`, and also available for openpi-torch) — every GPU
  kernel classified by name into attention / gemm / conv / elementwise / other. Graph-safe (nsys
  reports real kernel names even inside graphs). Heuristic — always sanity-check the `other` bucket.

### The four stages (one nsys capture, clean wall numbers)

Because nsys must wrap the whole process (and CUPTI perturbs wall timing), one profiling run is split
into cheap stages the shell sequences (`--mode`):

1. **time** — pristine wall: segmented E2E (headline) + per-component standalone → `<out>.timing.json`
2. **time-native** — openpi's native single-compile E2E, in its OWN process (cross-check) →
   `<out>.timing-native.json`. Separate because the native fused graph and the per-phase graphs
   thrash the shared cudagraph pool if run together (dynamo recompiles every call). Skip with
   `CROSS_CHECK_NATIVE=0`.
3. **nsys** — bracket one steady-state segmented run UNDER `nsys profile` → `<out>.nsys-rep`
4. **parse** — NVTX split + kernel buckets from the report → `<out>.breakdown.json`
5. **report** — merge → `<out>.json` / `<out>.csv` (+ printed summary)

`--capture-range=cudaProfilerApi` means nsys records **only** between `cudaProfilerStart/Stop` (which
the python brackets), so model load, compile, cudagraph capture and warmup are excluded.

---

## Setup (conda only — no uv)

```bash
cd /home/eecs/ishirgarg/edge-robotics
bash install.sh
```

`install.sh` is idempotent: clones `openpi` and `realtime-vla` in-tree (both git-ignored, never
committed), creates the `edge-robotics` conda env from `environment.yml`, installs `openpi` +
`edge_robotics` editable `--no-deps`, copies openpi's patched transformers (`transformers_replace`,
required by `PI0Pytorch`), and sanity-checks imports.

> openpi is still a dependency (the PyTorch port lives in it and pulls JAX for config helpers only —
> `JAX_PLATFORMS=cpu` keeps JAX off the GPU). Override realtime-vla's location with `REALTIME_VLA_DIR`.

`nsys` must be available — on PATH or at `/usr/local/cuda-*/bin/nsys` (the scripts auto-detect it).

---

## Usage

```bash
source env.sh   # activate env + point caches at /scratch + JAX_PLATFORMS=cpu

# one system, one config, one prompt length (all four stages, end to end):
./profile_one.sh pi05_openpi_torch pi05_libero 200 6 out/libero_torch_L200
#                <system>          <config>     <L> <gpu> <out_dir>
```

Env overrides for `profile_one.sh`: `CHECKPOINT` (default `random`), `NUM_STEPS` (10), `WARMUP` (3),
`ITERS` (20), `BATCH_SIZE` (1), `OPENPI_TORCH_COMPILE_MODE`.

`OPENPI_TORCH_COMPILE_MODE`: `reduce-overhead` (default — CUDA graphs, compiles fast),
`max-autotune` (openpi's default — also graphs, compiles for many minutes), or `none`/`eager`
(graphs off; the NVTX split won't be available).

### Sweep over backends × prompt lengths

```bash
./profile_sweep.sh        # edit SYSTEMS / CONFIG_NAME / PROMPT_LENS / GPU at the top
```

`profile_sweep.sh` just loops `profile_one.sh` (the one-config unit of work) sequentially on one GPU.
Why prompt length matters: it sets the padded token-sequence length (`max_token_len`); the LLM
processes all padded positions, so larger prompts grow the VLM prefill (~quadratic in
`768 + max_token_len`) and each Action step (attends over the cached prefix), while Vision stays flat.

### Run a single stage directly

```bash
python scripts/profile_policy.py --system pi05_openpi_torch --config-name pi05_libero \
  --checkpoint random --gpu 6 --prompt-len 200 --mode time --output out/x/profile
```

### realtime-vla backend

Same `profile_one.sh`, `--system pi05_realtimevla`. The smoke path needs **no checkpoint and no
tokenizer** (random weights + random language embeds, like realtime-vla's `benchmark.py`). Caveats
(see also `out/*.json` `meta`):

- The repo captures the **whole** forward as one CUDA graph. We get the Vision/VLM/Action split by
  re-capturing it as **three per-stage sub-graphs** (the stages talk only through persistent in-place
  buffers, so this is **bitwise-identical** to their single graph — verified — and adds ~0 overhead).
  Their original single graph is kept as the `infer_native` cross-check.
- **Flow steps are hard-locked to 10**; `--num-steps != 10` is ignored.
- **Triton kernels are tuned for RTX 4090/5090**; they run on A6000 etc. but won't hit advertised latencies.
- **Real weights** (`--checkpoint <converted.pkl>`) need the HF-gated `paligemma-3b-pt-224` tokenizer
  (`REALTIME_VLA_TOKENIZER`); experimental/unvalidated.

---

## Outputs

- stdout: a per-run summary (E2E segmented + native, component breakdown, per-component standalone, buckets).
- `<out>/config.json`: **full reproducibility manifest** — hardware target (GPU name, compute
  capability, memory, UUID, driver + CUDA versions), `model_weight_dtype` (e.g. `bfloat16`), model
  config (variants, dims, prompt/horizon), compile modes, all package versions
  (torch/triton/transformers/openpi/jax/numpy/nsys), git commit + dirty flag, the invocation, and the
  result-affecting env vars. (Same `meta` the results carry, written as its own file.)
- `<out>/profile.json`: `meta` (= the manifest above) + `result` (E2E stats mean/median/p50/p90/p99,
  segmentation overhead, NVTX per-phase ms/%, kernel buckets + top kernels, per-component standalone ms).
- `<out>/profile.csv`: long format — one row per `(metric, phase)` for plotting across a sweep.
- `<out>/profile.nsys-rep`: the raw nsys capture; open in the Nsight Systems GUI for the full timeline.
- `<out>/profile.{timing,breakdown}.json`: the intermediate per-stage artifacts.

---

## Hardware / disk notes

- This machine has **8× NVIDIA RTX A6000 (48 GB)**; pin one with the `<gpu>` arg (check `nvidia-smi`
  for a free device — a real ~3B checkpoint needs the GPU mostly idle; `debug_pi05` fits anywhere).
- Home is ~20 GB, so the conda env / checkpoints / caches live on **`/scratch`** (see `env.sh`).
  `openpi/` and `realtime-vla/` are cloned in-tree but git-ignored; `install.sh` pulls them.

## Extending (modular by file)

- **New system**: add `edge_robotics/systems/<name>.py` implementing `ProfiledSystem` (return a
  `LoadedSystem` with `infer_segmented` + optional `infer_native`/`component_profiler` and
  `nvtx_phases`), and wire it into `cli._make_system` + `cli.SYSTEM_PHASES`.
- The profiling primitives (`profiling/{walltime,nsys}.py`) are system-agnostic and reusable.
