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


def _attach(res: dict, key: str, src: dict | None, build) -> None:
    """Attach res[key]=build(src) when src succeeded, else res[key+'_error'], else nothing."""
    if src is None:
        return
    if src.get("ok"):
        res[key] = build(src)
    else:
        res[f"{key}_error"] = src.get("error")


def build_result(
    *,
    e2e_segmented: list[float] | None,
    e2e_native: list[float] | None,
    breakdown_nvtx: dict | None,
    kernel_buckets: dict | None,
    components_standalone: dict | None,
    kernel_analysis: dict | None = None,
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

    _attach(res, "breakdown_nvtx", breakdown_nvtx, lambda s: {
        "phases_ms_per_infer": s["phases_ms_per_infer"],
        "phases_pct": _pct_split(s["phases_ms_per_infer"]),
        "attributed_frac": s.get("attributed_frac"),
        "residual_ms_per_infer": s.get("residual_ms_per_infer"),
        "total_gpu_ms_per_infer": s.get("total_gpu_ms_per_infer"),
        "method": s.get("method"),
    })

    _attach(res, "kernel_buckets", kernel_buckets, lambda s: {
        "buckets_ms_per_infer": s["buckets_ms_per_infer"],
        "buckets_pct": _pct_split(s["buckets_ms_per_infer"]),
        "total_gpu_ms_per_infer": s.get("total_gpu_ms_per_infer"),
        "top_kernels": s.get("top_kernels", []),
        "method": s.get("method"),
    })

    _attach(res, "components_standalone", components_standalone, lambda s: {
        "phases_ms_per_infer": s["phases_ms_per_infer"],
        "phases_pct": _pct_split(s["phases_ms_per_infer"]),
        "method": s.get("method"),
    })

    # Deep kernel/system analysis (per-phase x family, GEMM/GEMV split, launch/util overheads).
    _attach(res, "kernel_analysis", kernel_analysis, lambda s: {
        k: s[k] for k in
        ("per_phase_family_ms", "family_order", "gemm_split_ms", "system", "method")
        if k in s
    })

    # NOTE: server<->edge transfer sizing (res["bandwidth"]) is attached by cli._run_report AFTER this
    # assembler, because it needs the headline freq this function computes (freq_hz).

    return res
