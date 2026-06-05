"""JAX profiler capture + trace parsing.

`profile_inference` runs UNMODIFIED inference under `jax.profiler` (which records CUDA kernels
via CUPTI, so the same trace doubles as CUDA-level data) and writes a TensorBoard/XProf logdir
that also contains a gzipped Chrome-trace JSON (`*.trace.json.gz`).

`parse_trace` reads that JSON directly (stdlib gzip+json — no xprof/tensorboard API needed) and
reports total GPU device time per inference plus the heaviest kernels. NOTE: the optimized,
fused XLA graph does NOT label kernels as img/llm (they show as `gemm_fusion_dot_*`,
`loop_convert_fusion_*`, generic `name=jit(fun)/jit(main)`), so the trace can NOT separate
Vision from VLM on its own. That split is done by the direct vision probe (see systems/pi05_jax
`vision_infer`) + the num_steps regression. The trace here is a captured artifact + a GPU-time
cross-check against the wall-clock E2E. It never raises; on failure it returns {"ok": False}.
"""

from __future__ import annotations

import collections
import contextlib
import glob
import gzip
import json
import os
from collections.abc import Callable
from typing import Any


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
    """Warm up (compile), then capture `iters` steady-state inferences. Returns logdir."""
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


def parse_trace(logdir: str, *, iters: int) -> dict:
    """Read the Chrome-trace JSON; return total GPU device ms/infer + top kernels (cross-check)."""
    try:
        path = find_trace_json(logdir)
        if not path:
            return {"ok": False, "error": f"no *.trace.json.gz under {logdir}"}
        with gzip.open(path, "rt") as f:
            events = json.load(f).get("traceEvents", [])
        gpu_pids = _gpu_pids(events)

        per_name_us: dict[str, float] = collections.defaultdict(float)
        total_gpu_us = 0.0
        for e in events:
            if e.get("ph") != "X" or "dur" not in e:
                continue
            if gpu_pids and e.get("pid") not in gpu_pids:
                continue
            per_name_us[e.get("name", "")] += float(e["dur"])
            total_gpu_us += float(e["dur"])

        n = max(int(iters), 1)
        top = sorted(per_name_us.items(), key=lambda kv: -kv[1])[:25]
        return {
            "ok": True,
            "trace_json": path,
            "gpu_pids": sorted(p for p in gpu_pids if p is not None),
            "total_gpu_ms_per_infer": (total_gpu_us / 1000.0) / n,
            "top_kernels": [{"name": k, "ms_per_infer": (v / 1000.0) / n} for k, v in top],
        }
    except Exception as exc:  # noqa: BLE001 - best-effort, must not break a run
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
