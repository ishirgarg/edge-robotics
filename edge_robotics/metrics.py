"""Assemble the per-run result from the graphs-on measurements.

One run == one (system, config, prompt_len). The result carries, all from the model's REAL
graphs-on form:

  - e2e_native    : device-synced wall of the deployed native single-compile (max-autotune) E2E.
                    This is the HEADLINE latency/throughput (freq = 1000/median) — the latency the
                    robot experiences, with no segmentation artifacts.
  - e2e_segmented : wall of the per-phase-graph E2E (the BREAKDOWN VEHICLE; carries glue/clone
                    overhead). The gap segmented-vs-native at the same compile mode is the
                    segmentation overhead (the price of an NVTX-attributable breakdown).
  - breakdown_nvtx: GPU ms/infer per phase from nsys NVTX GPU projection (the component split).
  - kernel_buckets: GPU ms/infer per kernel family from nsys (graph-safe; the only split for an
                    opaque fused graph).
  - components_standalone: each phase timed standalone (secondary cross-check; no cross-phase overlap).

If the native cross-check wasn't run, the headline falls back to the segmented wall (flagged via
`headline_source`). Anything not measured (e.g. no nsys rep yet) is left None; nothing is inferred.
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
    kernel_analysis: dict | None = None,
    roofline: dict | None = None,
    bandwidth: dict | None = None,
) -> dict:
    """Combine whatever measurements are present into one JSON-serializable result dict."""
    res: dict = {}

    if e2e_segmented:
        res["e2e_segmented_ms"] = stats(e2e_segmented)
    if e2e_native:
        res["e2e_native_ms"] = stats(e2e_native)

    seg_med = res.get("e2e_segmented_ms", {}).get("median")
    nat_med = res.get("e2e_native_ms", {}).get("median")
    # HEADLINE latency/throughput = the deployed native single-compile path (no segmentation
    # artifacts). Fall back to the segmented wall only if the native cross-check wasn't run.
    headline = nat_med or seg_med
    if headline:
        res["headline_latency_ms"] = headline
        res["headline_source"] = "native" if nat_med else "segmented"
        res["freq_hz"] = 1000.0 / headline if headline > 0 else float("nan")
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

    # Deep kernel/system analysis (per-phase x family, GEMM/GEMV split, launch/util overheads).
    if kernel_analysis is not None and kernel_analysis.get("ok"):
        res["kernel_analysis"] = {
            k: kernel_analysis[k] for k in
            ("per_phase_family_ms", "family_order", "gemm_split_ms", "system", "method")
            if k in kernel_analysis
        }
    elif kernel_analysis is not None:
        res["kernel_analysis_error"] = kernel_analysis.get("error")

    # Analytic roofline lower bound + measured efficiency (MFU/MBU, ideal-vs-achieved).
    if roofline is not None:
        res["roofline"] = roofline

    # Server<->edge transfer sizing (KV cache + conditioning data per inference).
    if bandwidth is not None:
        res["bandwidth"] = bandwidth

    return res
