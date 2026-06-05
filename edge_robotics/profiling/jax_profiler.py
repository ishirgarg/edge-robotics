"""JAX profiler capture + per-phase device-time attribution.

`profile_inference` runs inference under `jax.profiler` (which records CUDA kernels via CUPTI)
and writes a TensorBoard/XProf logdir that also contains a gzipped Chrome-trace JSON
(`*.trace.json.gz`).

`parse_trace` reads that JSON directly (stdlib gzip+json — no xprof/tensorboard API needed) and
buckets GPU device time into the Vision / VLM / Action phases. The split is possible because the
system tags the public image/LLM submodule calls with `jax.named_scope` (see
`systems/pi05_jax.py:apply_namescope_patch`); the scope shows up on every device event as a
`/`-delimited path segment of `args["name"]`, e.g.
    jit(fun)/jit(main)/vision/_Module/Transformer/while/body/...
so we sum `dur` per matching segment.

IMPORTANT: this only works when XLA command buffers (CUDA graphs) are DISABLED
(`XLA_FLAGS=--xla_gpu_enable_command_buffer=`, set before `import jax` — the entrypoint does
this). With CUDA graphs on, the bulk of kernels are captured into graphs that strip per-op
metadata (no `name`, only a `cuda_graph_id`), and attribution collapses to ~0%.

`parse_trace` never raises; on failure it returns {"ok": False, "error": ...}.
"""

from __future__ import annotations

import collections
import contextlib
import glob
import gzip
import json
import os
import shutil
from collections.abc import Callable
from typing import Any

# The phase scopes, matched as exact path-segments of the trace event name. Mirrors
# systems/pi05_jax.PHASE_SCOPES (kept local to avoid importing jax/openpi here).
PHASE_SCOPES = ("vision", "vlm", "action")


@contextlib.contextmanager
def capture_trace(logdir: str):
    import jax

    os.makedirs(logdir, exist_ok=True)
    jax.profiler.start_trace(logdir)
    try:
        yield
    finally:
        jax.profiler.stop_trace()


def profile_inference(
    infer: Callable[[], Any],
    block: Callable[[Any], Any],
    logdir: str,
    *,
    warmup: int,
    iters: int,
) -> str:
    """Warm up (compile), then capture `iters` steady-state inferences. Returns logdir.

    The logdir is wiped first so a rerun never parses a stale trace from a previous run (jax
    writes a fresh timestamped subdir each time, which would otherwise accumulate).
    """
    shutil.rmtree(logdir, ignore_errors=True)
    for _ in range(warmup):
        block(infer())
    with capture_trace(logdir):
        for _ in range(iters):
            block(infer())
    return logdir


def find_trace_json(logdir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(logdir, "plugins", "profile", "*", "*.trace.json.gz")))
    return hits[-1] if hits else None


def _gpu_pids(events: list[dict]) -> set[int]:
    """Pids whose process name marks them as a GPU device (e.g. '/device:GPU:0')."""
    gpu_pids: set[int] = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            label = str(e.get("args", {}).get("name", "")).lower()
            if "device:gpu" in label or "gpu" in label:
                gpu_pids.add(e.get("pid"))
    return gpu_pids


def _scope_of(args: dict) -> str | None:
    """Return the phase scope tagged on this op, matched as an exact path-segment of args['name']
    (so 'action_out_proj' is NOT mistaken for 'action'), or None if unscoped."""
    segs = set(str(args.get("name", "")).split("/"))
    for s in PHASE_SCOPES:
        if s in segs:
            return s
    return None


def parse_trace(logdir: str, *, iters: int) -> dict:
    """Read the Chrome-trace JSON; bucket GPU device time into Vision/VLM/Action (ms per infer).

    Returns a dict with ok=True and:
      phases_ms_per_infer : {vision, vlm, action}
      residual_ms_per_infer : device time not under any phase scope (inter-phase glue)
      total_gpu_ms_per_infer : all GPU device time / iters (cross-check vs E2E)
      attributed_frac : fraction of GPU time that landed in a phase (sanity; want >> 0)
      top_kernels : heaviest kernels by name (debugging)
    """
    try:
        path = find_trace_json(logdir)
        if not path:
            return {"ok": False, "error": f"no *.trace.json.gz under {logdir}"}
        with gzip.open(path, "rt") as f:
            events = json.load(f).get("traceEvents", [])
        gpu_pids = _gpu_pids(events)

        per_scope_us: dict[str, float] = collections.defaultdict(float)
        per_name_us: dict[str, float] = collections.defaultdict(float)
        total_gpu_us = 0.0
        for e in events:
            if e.get("ph") != "X" or "dur" not in e:
                continue
            if gpu_pids and e.get("pid") not in gpu_pids:
                continue
            dur = float(e["dur"])
            total_gpu_us += dur
            per_name_us[e.get("name", "")] += dur
            per_scope_us[_scope_of(e.get("args", {}) or {}) or "_residual"] += dur

        n = max(int(iters), 1)
        phases_ms = {s: per_scope_us.get(s, 0.0) / 1000.0 / n for s in PHASE_SCOPES}
        residual_ms = per_scope_us.get("_residual", 0.0) / 1000.0 / n
        total_ms = total_gpu_us / 1000.0 / n
        attributed = sum(per_scope_us.get(s, 0.0) for s in PHASE_SCOPES)
        attributed_frac = attributed / total_gpu_us if total_gpu_us > 0 else 0.0
        top = sorted(per_name_us.items(), key=lambda kv: -kv[1])[:25]
        return {
            "ok": True,
            "trace_json": path,
            "gpu_pids": sorted(p for p in gpu_pids if p is not None),
            "phases_ms_per_infer": phases_ms,
            "residual_ms_per_infer": residual_ms,
            "total_gpu_ms_per_infer": total_ms,
            "attributed_frac": attributed_frac,
            "top_kernels": [{"name": k, "ms_per_infer": v / 1000.0 / n} for k, v in top],
        }
    except Exception as exc:  # noqa: BLE001 - best-effort, must not break a run
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
