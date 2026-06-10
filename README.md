# edge-robotics

Profiler-driven latency study for robotics foundation models **at the edge**. It profiles
**pi-0.5** on a single GPU, running the model **only in its real, fully-optimized CUDA-graphs-on
form**, and attributes time to each component with **NVIDIA Nsight Systems (nsys)** — never by
re-running the model eager to peek inside it.

For each run it reports:

- **E2E (ms / Freq)** — headline wall latency of the full inference, measured on the **DEPLOYED**
  native single-compile (max-autotune) path (graphs on) — the latency the robot actually sees.
- **Component breakdown** — Vision / VLM / Action GPU time from a single nsys capture of the
  segmented (per-phase-graph) run (the within-run breakdown vehicle).
- **Per-component standalone** — each component timed on its own (secondary cross-check; graphs on).
- **Per-phase × kernel-family** — attention vs GEMM (weights/activations) vs elementwise vs memory
  ops, *crossed with* the Vision/VLM/Action phases, plus a compute-GEMM vs memory-bound-GEMV split.
  Answers "attention vs W/A, backbone vs action part."
- **System & overhead** — kernels/infer, CUDA-graph vs eager launches, GPU-busy-vs-wall utilization,
  launch-API overhead, SM-coverage. Answers "CUDA launch overheads, utilization."
- **Roofline** — analytic ideal lower bound (`max(FLOPs/peak, Bytes/BW)` per operator, after
  [NVlabs/vla-perf](https://github.com/NVlabs/vla-perf)) with arithmetic intensity, compute-vs-memory
  bound, and **how far from ideal** we are (MFU/MBU, efficiency). Answers "compare with roofline."
- **Server↔edge transfer** — per-inference bytes the action expert conditions on (the VLM's prefix
  KV cache + masks + state) if the VLM ran on a server and the action expert on the edge. Answers
  "how much network bandwidth for a VLM-on-server / VLA-on-edge split." (`edge_robotics/bandwidth.py`)
- **Attention study** *(separate tool)* — with REAL weights, how the action expert attends to vision
  vs language vs proprioception. For discrete-state pi05 (state folded into the prompt) proprioception
  is separated from language. See `scripts/attention_heatmaps.py`.

The framework is config-driven: `--config-name` is resolved via openpi `get_config`, so any
**pi0/pi05 × DROID/ALOHA/LIBERO** config works on the openpi-torch backend with no new code, and the
roofline is **quantization-aware** (a `QuantScheme` for int8/fp8/int4 variants). Two backends
(pick with `--system`):

| `--system` | what | component attribution |
|------------|------|-----------------------|
| `openpi_torch` | openpi's native **PyTorch** port (`PI0Pytorch`); serves any pi0/pi05 × dataset config | **nsys NVTX GPU projection** over per-phase CUDA graphs — clean Vision/VLM/Action split, graphs ON |
| `realtime_vla` | PyTorch + Triton, from [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) (fused Triton kernels + CUDA-graph replay); **pi05 only** | **same NVTX split** — we re-capture its single graph as three per-stage sub-graphs (bitwise-identical to native), plus kernel-family buckets |

> The `pi05_openpi_torch` / `pi05_realtimevla` names are accepted as aliases. The JAX backend was removed.

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
`edge_robotics/systems/openpi_torch.py` does.

Validated on the real model (pi05_libero, A6000, graphs on): the per-phase NVTX projection sums to
**~99%** of E2E, and the segmented path's wall time is **within noise of** openpi's native
single-compile path — so segmenting buys an attributable breakdown at no measurable latency cost.

### The numbers

- **E2E (native, HEADLINE)** — median device-synced **wall** time of the **deployed** path, exactly
  as each repo ships it: openpi → `torch.compile(sample_actions, mode="max-autotune")` (openpi's own
  `Pi0Config` default); realtime-vla → its single captured CUDA graph (`infer.forward`). **Freq =
  1000/E2E.** Warmup absorbs `torch.compile` + cudagraph capture. This is the headline because it
  carries no segmentation artifacts. (If `CROSS_CHECK_NATIVE=0` skips it, the headline falls back to
  the segmented wall, flagged `headline_source=segmented`.)
- **E2E (segmented, breakdown vehicle)** — wall time of the per-phase-graph inference; its purpose is
  the within-run NVTX split, not the headline (it carries `.clone()`/eager-glue overhead). Both paths
  default to `max-autotune` (`OPENPI_TORCH_COMPILE_MODE` / `OPENPI_TORCH_NATIVE_COMPILE_MODE`) so the
  per-phase kernels are the deployed kernels and the seg-vs-native gap is a clean segmentation cost.
  Set `OPENPI_TORCH_COMPILE_MODE=reduce-overhead` only for fast iteration (degrades breakdown fidelity).
- **Component breakdown** — GPU ms/infer per phase from nsys `nvtx_gpu_proj_sum`. `%` is each
  phase's share. `attributed_frac` (share of GPU time landing in a phase) and `residual` (inter-phase
  glue: noise sampling, the `x_t+dt·v_t` update, D2D copies) are reported, not hidden.
- **Per-component standalone** — each phase timed in isolation, fully optimized (graphs on), with the
  upstream inputs precomputed once and reused.
- **Kernel-family buckets** (`realtime_vla`, and also available for openpi-torch) — every GPU
  kernel classified by name into attention / gemm / conv / quantize / elementwise / other (the
  `quantize` family + int8/fp8 GEMM keys cover quantized backends). Graph-safe (nsys reports real
  kernel names even inside graphs). Heuristic — always sanity-check the `other` bucket.

### The four stages (one nsys capture, clean wall numbers)

Because nsys must wrap the whole process (and CUPTI perturbs wall timing), one profiling run is split
into cheap stages the shell sequences (`--mode`):

1. **time** — pristine wall: segmented E2E (breakdown vehicle) + per-component standalone → `<out>.timing.json`
2. **time-native** — the DEPLOYED native single-compile E2E = the **HEADLINE**, in its OWN process →
   `<out>.timing-native.json`. Separate because the native fused graph and the per-phase graphs
   thrash the shared cudagraph pool if run together (dynamo recompiles every call). `CROSS_CHECK_NATIVE=0`
   skips it, but then the headline falls back to the (non-deployment-faithful) segmented wall.
3. **nsys** — bracket one steady-state segmented run UNDER `nsys profile` → `<out>.nsys-rep`
4. **parse** — NVTX split + kernel buckets from the report → `<out>.breakdown.json`
5. **report** — merge → `<out>.json` / `<out>.csv` (+ printed summary)

`--capture-range=cudaProfilerApi` means nsys records **only** between `cudaProfilerStart/Stop` (which
the python brackets), so model load, compile, cudagraph capture and warmup are excluded.

---

## Bottleneck, overhead & roofline analysis

The same single nsys capture (parsed from its `.sqlite`, `edge_robotics/profiling/kernel_analysis.py`)
and an analytic model (`edge_robotics/roofline.py`) answer three questions beyond the phase split:

**1. Attention vs weights/activations, backbone vs action.** Every GPU kernel is attributed to its
phase (vision/vlm/action) AND its family (attention / gemm / elementwise / memory-ops) by the
`correlationId → launch → NVTX-window` projection (the only thing that survives CUDA graphs — kernels
run async, long after the eager NVTX range, so they're matched through the launch that issued them).
The gemm bucket is further split into **compute-bound GEMM vs memory-bound GEMV** (cuBLAS dispatches
GEMV for batch-1 decode). On A6000/L64, openpi-torch: vision and VLM are GEMM-dominated; the **action
expert is ~40% memory-bound GEMV** (the batch-1 flow-matching decode).

**2. CUDA launch overhead & utilization.** CUDA graphs replay **~5400 kernels via only ~12 graph
launches + ~31 eager launches** per inference, so launch overhead is amortised to ~nothing:
**non-GPU time is ~2% of wall** (the pipeline is GPU-bound). Reported: kernels/infer, graph vs eager
launches, GPU-busy/wall utilization (~96%), launch-API CPU time (async-overlapped, off the critical
path), mean/median kernel duration, and a time-weighted SM-coverage proxy (~0.87 — the tiny decode
kernels don't fill all 84 SMs).

**3. Roofline — how far from ideal.** Per operator `T = max(FLOPs/peak_compute, Bytes/peak_BW)`
(after [NVlabs/vla-perf](https://github.com/NVlabs/vla-perf)), summed per phase, vs the A6000's
154.8 dense-bf16 TFLOPS / 768 GB/s (ridge ≈ 202 FLOP/byte). pi-0.5 on A6000:

| phase  | OI (FLOP/byte) | bound        | ideal ms | measured GPU ms | efficiency (MFU/MBU) |
|--------|----------------|--------------|----------|------------------|----------------------|
| vision | ~322           | **compute**  | ~4.6     | ~15.0            | 31% (MFU 28%)        |
| vlm    | ~530           | **compute**  | ~22.3    | ~32.9            | 68% (MFU 67%)        |
| action | ~7             | **memory**   | ~14.5    | ~35.3 (openpi)   | 41% (MBU 41%)        |

The VLM prefill is the best-utilised (67% MFU). The **action expert is memory-bound** (tiny OI: 10
query tokens re-reading all expert weights + the fp32 adaRMS modulation weights every denoise step) —
and is where the headroom is: openpi-torch hits 41% of its memory roofline, while **realtime-vla's
fused Triton kernels hit ~93%** of the same roofline (15.5 ms vs 35.3 ms) — the clearest concrete
optimization signal in the study. End-to-end, the measured wall is ~50% of the analytic ideal.

> Roofline hardware is auto-detected (A6000; unknown GPUs warn before defaulting); override for other
> targets with `EDGE_ROBOTICS_PEAK_BF16_TFLOPS` / `EDGE_ROBOTICS_MEM_BW_GBPS` (e.g. to model a Jetson).
> The roofline is **quantization-aware**: a `QuantScheme` (weights/activations/kv/compute dtype, read
> from the run's meta) rescales the bytes and selects the int8/fp8/int4 tensor-core peak; bf16 (the
> default) is byte-identical to the dense model.

**4. Server↔edge transfer (`edge_robotics/bandwidth.py`).** If the heavy VLM ran on a server and only
the action expert on the edge robot, the per-inference data crossing the network is what the expert
**conditions on** — dominated by the VLM's prefix **KV cache** (gemma_2b prefill: depth × {K,V} ×
prefix_len × kv_heads(MQA=1) × head_dim) plus the pad mask and (pi0) state. For pi05_libero that's
**~17 MiB/inference** (≈200 MB/s at the deployed rate; the openpi-torch backend cross-checks it by
measuring the real KV tensors — analytic == measured). The alternative vision-on-server split ships
only the ~3 MiB image embeddings instead — a concrete bandwidth trade-off for disaggregated VLA.

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

# one system, one config (prompt_len 'native' = the config's trained value; all stages, end to end):
./profile_one.sh openpi_torch pi05_libero native 6 out/libero_torch
#                <system>      <config>     <L>    <gpu> <out_dir>     # <L> can be an int to sweep
```

Env overrides for `profile_one.sh`: `CHECKPOINT` (default `random`), `NUM_STEPS` (10), `WARMUP` (3),
`ITERS` (20), `BATCH_SIZE` (1), `CROSS_CHECK_NATIVE` (1), `OPENPI_TORCH_COMPILE_MODE`.

`OPENPI_TORCH_COMPILE_MODE`: `max-autotune` (default — openpi's deployed mode, CUDA graphs; compiles
for a few minutes), `reduce-overhead` (CUDA graphs, fast compile — fast iteration only), or
`none`/`eager` (graphs off; the NVTX split won't be available).

### Sweep over backends × configs × prompt lengths

```bash
# the matrix (env-overridable, space-separated lists); pi0_* auto-skipped on the pi05-only realtime_vla:
CONFIGS="pi05_libero pi05_droid pi0_droid pi0_aloha" SYSTEMS="openpi_torch realtime_vla" \
  PROMPT_LENS="native" GPU=0 ./profile_sweep.sh
# per-config real checkpoints (else random-init): CKPT_pi05_libero=/scratch/.../pi05_libero_torch ...
```

`profile_sweep.sh` just loops `profile_one.sh` (the one-config unit of work) sequentially on one GPU.
Why prompt length matters: it sets the padded token-sequence length (`max_token_len`); the LLM
processes all padded positions, so larger prompts grow the VLM prefill (~quadratic in
`768 + max_token_len`) and each Action step (attends over the cached prefix), while Vision stays flat.

### Run a single stage directly

```bash
python scripts/profile_policy.py --system openpi_torch --config-name pi05_libero \
  --checkpoint random --gpu 6 --mode time --output out/x/profile   # omit --prompt-len for native
```

### realtime-vla backend

Same `profile_one.sh`, `--system realtime_vla`. The smoke path needs **no checkpoint and no
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

## Real pi05_libero weights & the LIBERO attention study

Latency is value-independent, so the profiler defaults to `--checkpoint random`. But the **real
trained weights** are needed for a faithful "deployed model" run and, crucially, for the **attention
study** (random weights give meaningless attention). Fetch + convert the public checkpoint once:

```bash
source env.sh
python scripts/get_pi05_libero_torch.py        # downloads gs://openpi-assets/checkpoints/pi05_libero
                                                # (anon gcsfs, ~12GB) + converts JAX->torch safetensors
```

Then profile the real weights (identical latency, now a faithful artifact), or run the attention study:

```bash
# real-weights profiling run (the full deep stack, on the real deployed model):
CHECKPOINT=/scratch/ishirgarg/openpi_cache/pi05_libero_torch \
  ./profile_one.sh openpi_torch pi05_libero native 6 out/real_libero

# attention heatmaps: how the action expert attends to vision vs language vs proprioception,
# with the REAL weights on a REAL LIBERO frame (eager — an interpretability artifact, not a timing):
python scripts/attention_heatmaps.py \
    --checkpoint /scratch/ishirgarg/openpi_cache/pi05_libero_torch --gpu 6 --out out/attention/pi05_libero
```

**What the attention study finds** (real pi05_libero, real LIBERO frame "put the white mug on the
left plate…"): the action expert attends **~45% to language, ~32% to vision, ~23% to its own action
tokens** — the task instruction drives action most. Within vision, the **wrist camera (~25%) far
outweighs the base camera (~7%)**. The absent 3rd camera (masked) and padded language tokens get
**0.0** (a built-in correctness check; per-query softmax mass sums to 1.0). For **pi05_libero there is
no proprioception modality** (`discrete_state_input=False` → state is neither a prompt nor a suffix
token; the policy conditions on vision + language only). For **discrete-state pi05** (pi05_droid /
aloha / base) the state IS folded into the prompt, so the study recovers that token span and reports a
separate **proprioception** bucket distinct from language (pi0 instead carries state as a continuous
suffix token). Outputs:
`attention.json` + four plots (by-modality bar, layer×modality heatmap, attention-vs-denoise-step,
and a spatial overlay of action→base-camera attention on the real image). Measured proprioception
separation on the real **pi05_droid** weights: vision 23% / language 23% / action-self 32% /
**proprioception 21%** — the in-prompt state, previously lumped into "language," is now its own bucket.

---

## Multi-model matrix (pi0 / pi05 × DROID / ALOHA / LIBERO)

`--config-name` is resolved via openpi `get_config`, so the openpi-torch backend profiles any
pi0/pi05 × dataset config unchanged. Fetch + convert real checkpoints (each ~14 GB; `source env.sh`
first so they land on `/scratch`):

```bash
source env.sh
for cfg in pi05_libero pi05_droid pi0_droid pi0_aloha_sim; do
  python scripts/get_pi0_torch.py --config-name $cfg --out /scratch/ishirgarg/openpi_cache/${cfg}_torch
done
# profiling matrix (latency + breakdown + roofline + system + bandwidth), real weights, native prompt:
CONFIGS="pi05_libero pi05_droid pi0_droid pi0_aloha_sim" \
  CKPT_pi05_libero=/scratch/ishirgarg/openpi_cache/pi05_libero_torch \
  ... ./profile_sweep.sh   # pi0_* auto-skipped on the pi05-only realtime_vla backend
```

Camera/state layout per dataset (driven by openpi's `*Inputs`): LIBERO/DROID = base + 1 wrist (3rd
cam masked), state 8; ALOHA = high + 2 wrists (all real), state 14. pi0 carries state as a continuous
action-expert suffix token; pi05 either folds discretized state into the prompt (DROID/base) or omits
it (LIBERO).

### Matrix results (RTX A6000, real weights, max-autotune deployed path, native prompt)

| config | headline (deployed) | breakdown vision/vlm/action | E2E vs roofline | server→edge KV/infer |
|---|---|---|---|---|
| pi05_libero | 89.6 ms / 11.2 Hz | 18% / 46% / 36% | 50% of ideal | 17.0 MiB (199 MB/s) |
| pi05_droid  | 92.5 ms / 10.8 Hz | 17% / 44% / 39% | 49% of ideal | 17.0 MiB (193 MB/s) |
| pi0_droid   | 75.8 ms / 13.2 Hz | 22% / 42% / 36% | 46% of ideal | 14.3 MiB (198 MB/s) |
| pi0_aloha_sim | 81.3 ms / 12.3 Hz | 20% / 40% / 40% | 44% of ideal | 14.3 MiB (185 MB/s) |

Takeaways: the **VLM prefill dominates** (40–46%), with the action expert close behind (36–40%); pi0
configs are **faster than pi05** (75–81 vs 90–92 ms) because their shorter prompt (`max_token_len`
48 vs 200) shrinks the prefill (and the KV transfer, 14.3 vs 17.0 MiB); every config sits at **~44–50%
of its analytic roofline** — the same large headroom the single-config study found, concentrated in
the memory-bound action expert. Headline is the native deployed (max-autotune) path for all.

## Evaluation (accuracy)

Two complementary measures (the "both" eval): simulator-free action-error, and closed-loop sim success.

**1. Offline action-error** (simulator-free) — runs the real inference path on sampled frames and compares
the predicted action chunk to the dataset's ground-truth (normalized like training — quantile for pi05,
z-score for pi0):

```bash
python scripts/eval_offline.py --checkpoint /scratch/.../pi05_libero_torch \
    --config-name pi05_libero --gpu 0 --out out/eval/pi05_libero
```

Measured **pi05_libero: normalized RMSE 0.079 / MAE 0.026** (12 frames × 10-step horizon). It's wired for
LIBERO (its public LeRobot parquet carries ground-truth actions); DROID/ALOHA need their real episodes.

**2. Closed-loop sim success-rate** — rolls the policy out *in* the LIBERO simulator and reports task
success. The model is served over openpi's websocket protocol (`serve_libero_torch.py` reuses openpi's
real LIBERO transform pipeline — Normalize / LiberoInputs / Unnormalize / LiberoOutputs — straight off the
torch checkpoint), and `eval_libero_sim.py` drives the sim client (same rollout loop as openpi's
`examples/libero/main.py`). The sim stack (robosuite/bddl/mujoco) lives in a **separate py3.8 conda env**;
rendering is headless via `MUJOCO_GL=egl` (no Docker, no X server):

```bash
# terminal 1 — policy server (main env, torch checkpoint, pick a GPU):
CUDA_VISIBLE_DEVICES=1 python scripts/serve_libero_torch.py \
    --checkpoint /scratch/.../pi05_libero_torch --config-name pi05_libero --port 8000
# terminal 2 — LIBERO sim client (py3.8 libero-sim env, headless EGL):
conda activate /scratch/ishirgarg/envs/libero-sim
MUJOCO_GL=egl CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
LIBERO_CONFIG_PATH=/scratch/ishirgarg/libero_config PYTHONPATH=openpi/third_party/libero \
python scripts/eval_libero_sim.py --task-suite-name libero_spatial \
    --num-tasks 10 --num-trials-per-task 10 --no-save-video --out out/sim/pi05_libero
```

Measured **pi05_libero on libero_spatial: 100% (100/100, all 10 tasks × 10 trials)** — in line with the
published pi05_libero numbers for this suite. DROID has no simulator; ALOHA sim (`gym_aloha`/`dm_control`)
is not wired into the client yet.

## Outputs

- stdout: a per-run summary (E2E segmented + native, component breakdown, per-component standalone, buckets).
- `<out>/config.json`: **full reproducibility manifest** — hardware target (GPU name, compute
  capability, memory, UUID, driver + CUDA versions), `model_weight_dtype` (e.g. `bfloat16`), model
  config (variants, dims, prompt/horizon), compile modes, all package versions
  (torch/triton/transformers/openpi/jax/numpy/nsys), git commit + dirty flag, the invocation, and the
  result-affecting env vars. (Same `meta` the results carry, written as its own file.)
- `<out>/profile.json`: `meta` (= the manifest above) + `result` (E2E stats mean/median/p50/p90/p99,
  segmentation overhead, NVTX per-phase ms/%, kernel buckets + top kernels, per-component standalone ms,
  and the deep analysis: `kernel_analysis` (per-phase×family, gemm/gemv split, `system` overhead/util)
  and `roofline` (per-phase OI/bound/ideal-ms + measured MFU/MBU/efficiency)).
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
  `LoadedSystem` with `infer_native` [the headline] + `infer_segmented` + optional `component_profiler`,
  and `nvtx_phases`), and wire it into `cli._make_system`. Phases flow from `LoadedSystem.nvtx_phases`
  (persisted into meta and read back by the offline `parse`/`report` stages) — there is no phase table to edit.
- The profiling primitives (`profiling/{walltime,nsys}.py`) are system-agnostic and reusable.
- **Deep analysis** is system-agnostic too: `profiling/kernel_analysis.py` (per-phase×family + system
  overhead from the nsys SQLite) and `roofline.py` (analytic lower bound from model dims + hardware
  peaks) work for any system whose meta carries the model dims and which emits NVTX phases.
- **Attention study**: `attention.py` (eager capture + modality bucketing + plots) and `libero_obs.py`
  (real LIBERO frame → model `Observation` + token layout), driven by `scripts/attention_heatmaps.py`.
