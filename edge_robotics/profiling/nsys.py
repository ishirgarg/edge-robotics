"""NVIDIA Nsight Systems (nsys) based profiling: NVTX phase attribution with CUDA graphs ON.

The previous approach ran the model EAGER (graphs off) and timed submodules with cuda-events,
because a single fused CUDA graph hides per-op boundaries. This module replaces that: the model
runs in its real graphs-on form and we attribute GPU time to phases with two complementary,
nvidia-native mechanisms read back from a single nsys capture:

  1. NVTX GPU projection (`nvtx_gpu_proj_sum`)  -> GPU device time per NVTX range
       (vision / vlm / action). This is the authoritative component split. It works with CUDA
       graphs ONLY when each phase is a SEPARATE compiled/cudagraph callable and the NVTX range
       is pushed/popped in EAGER glue AROUND the call (never inside the compiled region) — a
       single fused graph launches all its kernels under one `cudaGraphLaunch` that falls outside
       any NVTX window pushed at replay time, so the split collapses. The torch backend is
       structured accordingly (see systems/openpi_torch.py).
  2. Kernel-family buckets (`cuda_gpu_kern_sum`) -> every GPU kernel classified into
       attention / gemm / conv / quantize / elementwise / other by name. Backend-agnostic; a useful
       cross-cut for any backend (and the fallback for a truly opaque fused graph with no NVTX split).

In-process side (this file): helpers the profiled script calls to bracket exactly one steady-state
region for nsys (`profiler_capture`, paired with `nsys --capture-range=cudaProfilerApi`) and to
emit NVTX ranges (`nvtx_range`). Offline side: `parse_nvtx_gpu_proj` / `parse_kernel_buckets`
shell out to `nsys stats` and parse its CSV. Neither parser raises; on failure they return
{"ok": False, "error": ...} so a run is never broken by post-processing.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import shutil
import subprocess
from collections.abc import Iterator

import torch

# Kernel-name -> family. First match wins, so ORDER MATTERS (specific before generic). Heuristic
# and brittle by nature — always sanity-check the size of the "other" bucket before trusting it.
# Covers cuBLAS/cutlass/cuDNN (openpi-torch), Triton (realtime-vla), and quantized kernels.
_KERNEL_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    # quantize/dequantize/rescale kernels are pure quantization OVERHEAD — surface them rather than
    # bury them in "other". First, so an fp8/int8 cast isn't swallowed by a later 'gemm' match.
    ("quantize", ("quant", "dequant", "to_fp8", "to_int8", "fp8_cast", "rescale", "dynamic_scale")),
    # 'attn' + realtime-vla's QK^T kernel (matmul_abT_scale) before 'gemm' so attention isn't called gemm.
    ("attention", ("flash", "fmha", "sdpa", "attention", "attn", "softmax", "scaled_dot", "abt_scale")),
    # conv BEFORE gemm: cuDNN/cutlass convolutions run as implicit-GEMM (names carry 'implicit_gemm'/
    # 'winograd'), so they'd be swallowed by the generic 'gemm' match and leave the conv bucket empty.
    ("conv", ("conv", "cudnn", "nchw", "nhwc", "implicit_gemm", "winograd")),
    ("gemm", ("gemm", "gemv", "cutlass", "cublas", "addmm", "bmm", "_mm", "matmul", "wgrad", "dgrad",
              "sgemm", "hgemm", "igemm", "imma", "dp4a", "i8i8", "s8s8", "marlin", "awq", "machete")),
    # NOTE: no bare 'triton' here — it forced every Triton kernel into elementwise; inductor's fused
    # elementwise kernels still match via 'fused'/'add'/'mul', while Triton matmul/attn route correctly.
    ("elementwise", ("elementwise", "vectorized", "silu", "gelu", "rms", "norm", "layer_norm",
                     "copy", "cast", "add", "mul", "fused")),
]


def under_nsys() -> bool:
    """True when this process is running under `nsys profile` (it injects this env var)."""
    return bool(os.environ.get("NSYS_PROFILING_SESSION_ID"))


def nsys_bin() -> str | None:
    """Locate the nsys binary: PATH first, then the CUDA 12.x toolkit default."""
    found = shutil.which("nsys")
    if found:
        return found
    for cand in ("/usr/local/cuda-12.6/bin/nsys", "/usr/local/cuda/bin/nsys"):
        if os.path.exists(cand):
            return cand
    return None


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push/pop an NVTX range. MUST wrap the call to a compiled phase in EAGER code (not inside a
    torch.compile region) for nsys to attribute the graph's kernels to it (see module docstring)."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def profiler_capture() -> Iterator[None]:
    """Bracket exactly the steady-state region nsys should record. Pair with
    `nsys --capture-range=cudaProfilerApi --capture-range-end=stop`: nsys ignores everything before
    cudaProfilerStart (model load, torch.compile, cudagraph capture, warmup) and after stop.
    Device-synced on both edges so the captured window is precisely the timed kernels."""
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()


def _run_stats_csv(rep_path: str, report: str) -> list[dict]:
    """Run `nsys stats --report <report> --format csv` and return parsed rows (list of dicts).
    Raises on nsys failure; callers wrap this and degrade to {"ok": False}."""
    nsys = nsys_bin()
    if nsys is None:
        raise FileNotFoundError("nsys not found on PATH or in /usr/local/cuda*/bin")
    if not os.path.exists(rep_path):
        raise FileNotFoundError(f"nsys report not found: {rep_path}")
    out = subprocess.run(
        [nsys, "stats", "--report", report, "--format", "csv", "--force-export=true", rep_path],
        capture_output=True, text=True, check=True,
    ).stdout
    # nsys prepends non-CSV status lines ("Generating SQLite...", "Processing..."); the real CSV
    # starts at the header row. Find it by the report's known leading column.
    lines = out.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(("Range,", "Time (%),", '"Time (%)"'))), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def _f(row: dict, *keys: str) -> float:
    for k in keys:
        if k in row and row[k] not in ("", None):
            try:
                return float(str(row[k]).replace(",", ""))
            except ValueError:
                pass
    return 0.0


def _sum_total_ms(rep_path: str, report: str, n: int) -> float:
    return sum(_f(r, "Total Time (ns)") for r in _run_stats_csv(rep_path, report)) / 1e6 / n


def parse_nvtx_gpu_proj(rep_path: str, *, iters: int, phases: tuple[str, ...]) -> dict:
    """Per-phase GPU device time (ms/infer) from the NVTX GPU Projection Summary.

    Returns parse_trace-shaped dict: phases_ms_per_infer, total_gpu_ms_per_infer, attributed_frac.
    `attributed_frac` = Σ(phase projection) / total GPU op time — a COVERAGE cross-check. NOTE: each
    phase's "Total Proj Time" is a per-range SPAN (first-op-start to last-op-end, so it includes
    intra-range idle gaps), while the denominator is a SUM of op durations (kernels + memcpy/memset),
    so this can slightly EXCEED 1.0 (~3-5% on real captures). For the exact kernels-only attributed
    fraction (≤1 by construction), use kernel_analysis.system.phase_attributed_frac.
    """
    try:
        rows = _run_stats_csv(rep_path, "nvtx_gpu_proj_sum")
        n = max(int(iters), 1)
        per: dict[str, float] = {p: 0.0 for p in phases}
        for r in rows:
            name = str(r.get("Range", "")).lstrip(":").strip()
            if name in per:
                per[name] += _f(r, "Total Proj Time (ns)") / 1e6 / n  # ns -> ms/infer
        # Denominator for an honest attributed_frac. The numerator (NVTX GPU projection) counts
        # kernels + memcpy + memset within each range, so the denominator must include BOTH kernel
        # time (cuda_gpu_kern_sum) AND GPU memory-op time (cuda_gpu_mem_time_sum) — otherwise
        # attributed_frac can exceed 1.0 (the memops show up in the numerator but not the denominator).
        total_gpu_ms = 0.0
        try:
            total_gpu_ms += _sum_total_ms(rep_path, "cuda_gpu_kern_sum", n)
            try:
                total_gpu_ms += _sum_total_ms(rep_path, "cuda_gpu_mem_time_sum", n)
            except Exception:  # noqa: BLE001  — memops report optional; kernels dominate the basis
                pass
        except Exception:  # noqa: BLE001
            total_gpu_ms = sum(per.values())
        attributed = sum(per.values())
        return {
            "ok": attributed > 0,
            "method": "nsys-nvtx-gpu-proj",
            "phases_ms_per_infer": {p: per.get(p, 0.0) for p in phases},
            "residual_ms_per_infer": max(total_gpu_ms - attributed, 0.0),
            "total_gpu_ms_per_infer": total_gpu_ms,
            "attributed_frac": (attributed / total_gpu_ms) if total_gpu_ms > 0 else 0.0,
            "nsys_report": rep_path,
            "error": None if attributed > 0 else "no NVTX ranges projected (graphs hide them?)",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _family_of(kernel_name: str) -> str:
    low = kernel_name.lower()
    for fam, keys in _KERNEL_FAMILIES:
        if any(k in low for k in keys):
            return fam
    return "other"


def parse_kernel_buckets(rep_path: str, *, iters: int, top: int = 25) -> dict:
    """Classify every GPU kernel into attention/gemm/conv/elementwise/other (ms/infer).

    Backend-agnostic and graph-safe (nsys reports real kernel names even inside CUDA graphs); the
    a backend-agnostic cross-cut, and the fallback for a truly opaque fused graph with no NVTX split.
    """
    try:
        rows = _run_stats_csv(rep_path, "cuda_gpu_kern_sum")
        n = max(int(iters), 1)
        buckets: dict[str, float] = {}
        named: list[tuple[str, float]] = []
        for r in rows:
            name = str(r.get("Name", ""))
            ms = _f(r, "Total Time (ns)") / 1e6 / n
            fam = _family_of(name)
            buckets[fam] = buckets.get(fam, 0.0) + ms
            named.append((name, ms))
        named.sort(key=lambda kv: -kv[1])
        total = sum(buckets.values())
        return {
            "ok": total > 0,
            "method": "nsys-kernel-buckets",
            "buckets_ms_per_infer": buckets,
            "total_gpu_ms_per_infer": total,
            "top_kernels": [{"name": k, "ms_per_infer": v} for k, v in named[:top]],
            "nsys_report": rep_path,
            "error": None if total > 0 else "no kernels found in report",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
