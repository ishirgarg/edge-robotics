# edge-robotics

Profiler-driven latency study for robotics foundation models **at the edge**. It profiles
**pi-0.5** on a single GPU and reports a per-phase breakdown:

| Vision (ms / %) | VLM (ms / %) | Action (ms / %) | E2E (ms) | Freq (Hz) |

Two backends are profiled as **separate models** (pick with `--system`):

| `--system` | what | phase attribution |
|------------|------|-------------------|
| `pi05_jax` | native JAX/Flax, from [openpi](https://github.com/Physical-Intelligence/openpi) | JAX profiler trace bucketed by `jax.named_scope` (CUDA graphs off) |
| `pi05_realtimevla` | PyTorch + Triton, from [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) (fused kernels + CUDA-graph replay) | `torch.cuda.Event` timing of the eager Triton stages (graphs off); E2E from the graph |

In both, the model's forward math is run **unmodified** on dummy, spec-conformant inputs — the only
instrumentation is inert timing annotations around the public stages. Phase timings are a **direct
measurement**, not a `num_steps` regression or statistical inference. The same task variant on the
two backends is one "model" each, e.g. `out/pi05_droid` (JAX) vs `out/pi05_droid_realtimevla`.

---

## How latency is measured (read this)

One pi-0.5 inference = **Vision** (SigLIP image encode) → **VLM** (Gemma prefill, builds the
KV-cache) → **Action** (the `jax.lax.while_loop` flow-matching denoise loop, `num_steps` iters).

- **E2E (ms)** — median steady-state **wall** time of the full `sample_actions`. **Freq = 1000/E2E.**
  Timing is **device-synced**: every measured region ends with `jax.block_until_ready`, so we time
  real GPU completion, not async dispatch. Warmup iterations absorb JIT compilation.
- **Vision / VLM / Action (ms, %)** — **GPU device time per phase, read straight off the JAX
  profiler trace.** At load time we wrap the two public submodule calls in `jax.named_scope`
  (`edge_robotics/systems/pi05_jax.py:apply_namescope_patch`):
  `PaliGemma.img → "vision"`, and `PaliGemma.llm → "vlm"` (prefix prefill, no kv-cache) or
  `"action"` (denoise step, with kv-cache). These are inert annotations — the forward pass is
  never reimplemented. `parse_trace` then sums each GPU event's `dur` by the scope tag carried in
  its `args["name"]` path (e.g. `…/vision/…`). **%** is each phase's share of (Vision+VLM+Action).
- **Why CUDA graphs are disabled** — the profiler can only attribute per-op time if kernels keep
  their metadata. With XLA command buffers on, the bulk of kernels are captured into CUDA graphs
  that strip per-op names, and attribution collapses to ~0%. The entrypoint therefore sets
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` **before importing jax**. (Trade-off: this adds a
  little kernel-launch overhead, so E2E here is marginally above a graphs-on deployment.)
- **Trustworthiness signals** (printed per row): `attributed_frac` (share of GPU time that landed
  in a phase — typically ~97%; the small remainder is inter-phase glue: input transpose, noise
  sampling, the `x_t+dt·v_t` update, `action_out_proj`, D2D copies), `residual` ms, and
  `GPU_total` (summed device time) vs measured wall `E2E`.

The raw trace is kept under `<output>_trace/L<n>/…`; view it with `tensorboard --logdir <output>_trace`.

---

## Hardware note

The request said "RTX 6000"; this machine actually has **8× NVIDIA RTX A6000 (48 GB)**.
We pin to one GPU via `--gpu`. (GPU occupancy varies — check `nvidia-smi` for a free device;
a real ~3B pi-0.5 checkpoint needs the GPU to be mostly idle, the tiny `debug_pi05` fits anywhere.)

## Layout note (disk)

Home here is only ~20 GB, so the conda env, checkpoints, and pip cache live on **`/scratch`**
(see `env.sh`). Project code stays in this folder. The `openpi/` and `realtime-vla/` dependencies
are cloned in-tree but **git-ignored** — they aren't shipped with this repo; `install.sh` pulls them.

---

## Setup (conda only — no uv)

```bash
cd /home/eecs/ishirgarg/edge-robotics

# 1. clone the openpi dependency in-tree. It MUST exist before env create —
#    environment.yml installs `-e ./openpi/packages/openpi-client`. (install.sh
#    also auto-clones openpi as a fallback, but that runs after env create.)
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
    https://github.com/Physical-Intelligence/openpi.git openpi

# 2. create the env (python 3.11 + JAX/torch stack; skips openpi's heavy unused deps)
GIT_LFS_SKIP_SMUDGE=1 conda env create -f environment.yml

# 3. finish: editable openpi + edge_robotics (--no-deps) + sanity import.
#    Also auto-clones realtime-vla (and openpi, if step 1 was skipped).
conda activate edge-robotics
bash install.sh
```

Both `openpi/` and `realtime-vla/` are git-ignored — they're cloned by the steps above,
never committed to this repo.

`environment.yml` pins the subset of openpi's deps needed for JAX inference and deliberately
omits `lerobot`/`tensorflow`/`gym-aloha`/`dlimp` (only used for data-loading/training).
`install.sh` does `pip install --no-deps -e ./openpi` so those omitted deps aren't pulled.

`install.sh` also clones [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) in-tree
(for the `pi05_realtimevla` backend). It's imported via `sys.path` (copy-in files, not a package)
and needs only `torch`/`triton`/`transformers`, already in the env. Override its location with
`REALTIME_VLA_DIR`.

---

## Usage

```bash
source env.sh   # activate env + point caches (checkpoints, etc.) at /scratch

# prompt-length sweep on an idle GPU, real DROID checkpoint:
python scripts/profile_policy.py \
  --system pi05_jax \
  --config-name pi05_droid \
  --checkpoint gs://openpi-assets/checkpoints/pi05_droid \
  --gpu 4 \
  --prompt-lens 16 32 64 128 200 \
  --num-steps 10 --warmup 3 --iters 30 --batch-size 1 \
  --output out/pi05_droid
```

Fast plumbing test with no download (tiny dummy model, random weights):

```bash
python scripts/profile_policy.py --system pi05_jax --config-name debug_pi05 \
  --checkpoint random --gpu 4 --prompt-lens 32 128 --num-steps 10 \
  --warmup 2 --iters 5 --output out/smoke
```

### realtime-vla backend (PyTorch + Triton)

Same CLI, `--system pi05_realtimevla`. The smoke path needs **no checkpoint and no tokenizer**
(random weights + random language embeds, exactly like realtime-vla's `benchmark.py`):

```bash
python scripts/profile_policy.py \
  --system pi05_realtimevla \
  --config-name pi05_droid \
  --checkpoint random \
  --gpu 4 \
  --prompt-lens 16 32 64 128 200 \
  --num-steps 10 --warmup 3 --iters 30 \
  --output out/pi05_droid_realtimevla
```

Caveats for this backend (see also the `out/*.json` `meta`):

- **E2E is CUDA-graph replay (the fast path); the phase split is from the eager `record_run()`**
  (graphs off — graph replay is opaque). Percentages are sound; absolute per-stage ms are the eager
  numbers, so eager-sum > graph E2E (the per-row note shows this gap = the graph speedup). This is
  the same trade-off the JAX backend makes by disabling graphs for attribution.
- **Flow steps are hard-locked to 10** in realtime-vla's forward; `--num-steps != 10` is ignored.
- **Triton kernels are tuned for RTX 4090/5090**; they run on other GPUs (e.g. A6000) but won't hit
  the advertised latencies.
- **Real weights** (`--checkpoint <converted.pkl>`) need the HF-gated `paligemma-3b-pt-224`
  tokenizer (`REALTIME_VLA_TOKENIZER`) and a `.pkl` from `realtime-vla/convert_from_jax_pi05.py`.
  This path is implemented but experimental/unvalidated; the smoke path above is the supported one.

Reruns are self-overwriting: the same `--output` truncates `<output>.{json,csv}` and the whole
`<output>_trace/` dir is wiped at startup, so stale results from a previous run never linger.

### Key flags

| flag | meaning |
|------|---------|
| `--config-name` | `pi05_droid` (ah=15), `pi05_libero` (ah=10), `pi05_base`/`pi05_aloha` (ah=50), `debug_pi05` (tiny) |
| `--checkpoint` | `gs://…` URI, local path, or `random` (random-init, no download) |
| `--gpu` | CUDA index to pin |
| `--prompt-lens` | `max_token_len` values to sweep (the pi-0.5 prompt-length knob) |
| `--num-steps` | flow-matching denoise steps |
| `--warmup` / `--iters` | warmup (compile) and timed iterations per measurement |
| `--batch-size` | inference batch size |
| `--trace-dir` | where to write JAX traces (default `<output>_trace`; wiped per run) |

The JAX profiler trace (with CUDA graphs disabled for attribution) is always captured — it *is*
the phase measurement, not an optional artifact.

### Outputs

- stdout: the 5-metric table (one row per prompt length) + per-row method/notes.
- `<output>.json`: full E2E stats (mean/median/p50/p90/p99/std), per-phase trace buckets +
  residual + attributed fraction + top kernels, and run metadata.
- `<output>.csv`: one row per `(prompt_len, phase)` for plotting.
- `<output>_trace/L<n>/…`: raw JAX traces; view via `tensorboard --logdir <output>_trace`.

Why prompt length matters: it sets the padded token-sequence length (`max_token_len`); the LLM
processes all padded positions, so larger prompts grow the VLM prefill (~quadratic in
`768 + max_token_len`) and each Action step (attends over the cached prefix), while Vision stays
flat — surfacing how the bottleneck shifts.

---

## Extending (modular by file)

- **New system**: add `edge_robotics/systems/<name>.py` implementing `ProfiledSystem` and wire it
  into the tiny dispatch in `cli._make_system`.
- **Network simulation**: see `network/` (stub) — wrap openpi's websocket client/server.
- The profiling primitives (`profiling/{walltime,jax_profiler}.py`) are
  system-agnostic and reusable.
