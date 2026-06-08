"""Assemble the per-run result from the graphs-on measurements.

One run == one (system, config, prompt_len). The result carries, all from the model's REAL
graphs-on form:

  - e2e_segmented : device-synced wall of the per-phase-graph E2E (headline). freq = 1000/median.
  - e2e_native    : wall of openpi's native single-compile E2E (cross-check; may be absent). The
                    gap segmented-vs-native is the segmentation overhead of running phases as
                    separate graphs (the price of an NVTX-attributable breakdown).
  - breakdown_nvtx: GPU ms/infer per phase from nsys NVTX GPU projection (the component split).
  - kernel_buckets: GPU ms/infer per kernel family from nsys (graph-safe; the only split for an
                    opaque fused graph).
  - components_standalone: each phase timed standalone in its graphs-on form (independent of E2E).

Anything not measured (e.g. no nsys rep yet) is left None; nothing is inferred.
"""

from __future__ import annotations

import numpy as np


def stats(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def _pct_split(phases_ms: dict) -> dict:
    total = sum(v for v in phases_ms.values() if v)
    if total <= 0:
        return {k: None for k in phases_ms}
    return {k: 100.0 * v / total for k, v in phases_ms.items()}


def build_result(
    *,
    e2e_segmented: list[float] | None,
    e2e_native: list[float] | None,
    breakdown_nvtx: dict | None,
    kernel_buckets: dict | None,
    components_standalone: dict | None,
) -> dict:
    """Combine whatever measurements are present into one JSON-serializable result dict."""
    res: dict = {}

    if e2e_segmented:
        seg = stats(e2e_segmented)
        res["e2e_segmented_ms"] = seg
        res["freq_hz"] = 1000.0 / seg["median"] if seg["median"] > 0 else float("nan")
    if e2e_native:
        res["e2e_native_ms"] = stats(e2e_native)

    seg_med = res.get("e2e_segmented_ms", {}).get("median")
    nat_med = res.get("e2e_native_ms", {}).get("median")
    if seg_med and nat_med:
        res["segmentation_overhead_pct"] = 100.0 * (seg_med / nat_med - 1.0)

    if breakdown_nvtx is not None and breakdown_nvtx.get("ok"):
        p = breakdown_nvtx["phases_ms_per_infer"]
        res["breakdown_nvtx"] = {
            "phases_ms_per_infer": p,
            "phases_pct": _pct_split(p),
            "attributed_frac": breakdown_nvtx.get("attributed_frac"),
            "residual_ms_per_infer": breakdown_nvtx.get("residual_ms_per_infer"),
            "total_gpu_ms_per_infer": breakdown_nvtx.get("total_gpu_ms_per_infer"),
            "method": breakdown_nvtx.get("method"),
        }
    elif breakdown_nvtx is not None:
        res["breakdown_nvtx_error"] = breakdown_nvtx.get("error")

    if kernel_buckets is not None and kernel_buckets.get("ok"):
        b = kernel_buckets["buckets_ms_per_infer"]
        res["kernel_buckets"] = {
            "buckets_ms_per_infer": b,
            "buckets_pct": _pct_split(b),
            "total_gpu_ms_per_infer": kernel_buckets.get("total_gpu_ms_per_infer"),
            "top_kernels": kernel_buckets.get("top_kernels", []),
            "method": kernel_buckets.get("method"),
        }
    elif kernel_buckets is not None:
        res["kernel_buckets_error"] = kernel_buckets.get("error")

    if components_standalone is not None and components_standalone.get("ok"):
        p = components_standalone["phases_ms_per_infer"]
        res["components_standalone"] = {
            "phases_ms_per_infer": p,
            "phases_pct": _pct_split(p),
            "method": components_standalone.get("method"),
        }
    elif components_standalone is not None:
        res["components_standalone_error"] = components_standalone.get("error")

    return res
