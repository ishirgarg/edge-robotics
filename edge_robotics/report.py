"""Render the per-run summary and export JSON/CSV. One run == one (system, config, prompt_len)."""

from __future__ import annotations

import csv
import json
import os
from typing import Any


def _ms(d: dict | None, key: str = "median") -> str:
    if not d:
        return "—"
    v = d.get(key)
    return f"{v:.2f}" if v is not None else "—"


def _phase_cells(phases_ms: dict | None, phases_pct: dict | None, order: tuple[str, ...]) -> list[str]:
    out = []
    for p in order:
        if not phases_ms or phases_ms.get(p) is None:
            out.append("—")
            continue
        ms = phases_ms[p]
        pct = (phases_pct or {}).get(p)
        out.append(f"{ms:.2f} ({pct:.1f}%)" if pct is not None else f"{ms:.2f}")
    return out


def _phase_line(phases_ms: dict | None, phases_pct: dict | None, order: tuple[str, ...]) -> str:
    """Render 'name=cell  name=cell ...' for an arbitrary phase list (no hardcoded phase names)."""
    cells = _phase_cells(phases_ms, phases_pct, order)
    return "  " + "  ".join(f"{name}={cell}" for name, cell in zip(order, cells))


def render_summary(meta: dict, result: dict) -> str:
    _ph = meta.get("nvtx_phases")  # [] (opaque graph) stays empty; missing falls back to the default
    order = tuple(_ph) if _ph is not None else ("vision", "vlm", "action")
    lines: list[str] = []
    lines.append(
        f"Device: {meta.get('device_kind','?')} | system={meta.get('system','?')} | "
        f"config={meta.get('config_name','?')} | prompt_len={meta.get('prompt_len','?')} | "
        f"num_steps={meta.get('num_steps','?')} | batch={meta.get('batch_size','?')} | "
        f"iters={meta.get('iters','?')} | compile={meta.get('compile_mode','?')}"
    )
    lines.append("")

    seg = result.get("e2e_segmented_ms")
    nat = result.get("e2e_native_ms")
    seg_mode = meta.get("compile_mode")
    nat_mode = meta.get("native_compile_mode")
    # HEADLINE = the faithful deployed native path; segmented is the breakdown vehicle.
    if nat:
        nat_tag = f", {nat_mode}" if nat_mode else ""
        lines.append(f"E2E (native deployed path, HEADLINE{nat_tag}): {_ms(nat)} ms   "
                     f"(p90 {_ms(nat,'p90')}, min {_ms(nat,'min')})")
    if seg:
        seg_tag = f", {seg_mode}" if seg_mode else ""
        ov = result.get("segmentation_overhead_pct")
        # Only call the gap "segmentation overhead" when both paths used the SAME compile mode.
        same_mode = (seg_mode == nat_mode) or (seg_mode is None and nat_mode is None)
        if ov is not None:
            delta = f"  [segmentation {ov:+.1f}%]" if same_mode else f"  [Δ vs native {ov:+.1f}%, modes differ]"
        else:
            delta = ""
        label = "breakdown vehicle" if nat else "headline — native unavailable"
        lines.append(f"E2E (segmented, {label}{seg_tag}): {_ms(seg)} ms{delta}")
    if result.get("freq_hz") is not None:
        src = result.get("headline_source", "?")
        lines.append(f"Freq (headline={src})     : {result['freq_hz']:.2f} Hz")
    lines.append("")

    bd = result.get("breakdown_nvtx")
    if bd:
        lines.append("Component breakdown (nsys NVTX GPU projection, graphs ON):")
        lines.append(_phase_line(bd["phases_ms_per_infer"], bd.get("phases_pct"), order))
        # Honest cross-checks: the per-phase GPU projection should sum to ~the E2E wall and ~the
        # total GPU kernel time. (Σphases can slightly exceed kernel-sum: projection counts the full
        # per-range GPU-busy span incl. memops/overheads, kernel-sum counts kernels only.)
        phases_sum = sum(v for v in bd["phases_ms_per_infer"].values() if v)
        gpu = bd.get("total_gpu_ms_per_infer")
        xs = [f"Σphases={phases_sum:.2f}ms"]
        if gpu:
            xs.append(f"nsys GPU kernels={gpu:.2f}ms")
        if seg and seg.get("median"):
            xs.append(f"E2E wall={seg['median']:.2f}ms")
        lines.append("  cross-check: " + "  |  ".join(xs))
    elif result.get("breakdown_nvtx_error"):
        lines.append(f"Component breakdown (NVTX): unavailable ({result['breakdown_nvtx_error']})")

    comp = result.get("components_standalone")
    if comp:
        lines.append("Per-component standalone (graphs ON, isolated — secondary cross-check, no overlap):")
        lines.append(_phase_line(comp["phases_ms_per_infer"], comp.get("phases_pct"), order))

    kb = result.get("kernel_buckets")
    if kb:
        b = kb["buckets_ms_per_infer"]
        pct = kb.get("buckets_pct", {})
        order_k = ("attention", "gemm", "conv", "quantize", "elementwise", "other")
        parts = [f"{k}={b[k]:.2f}({pct.get(k,0):.0f}%)" for k in order_k if k in b]
        lines.append("Kernel-family buckets (nsys, graphs ON):")
        lines.append("  " + "  ".join(parts))
    elif result.get("kernel_buckets_error"):
        lines.append(f"Kernel buckets: unavailable ({result['kernel_buckets_error']})")

    _render_kernel_analysis(result.get("kernel_analysis"), order, lines)
    _render_bandwidth(result.get("bandwidth"), lines)

    return "\n".join(lines)


def _render_bandwidth(bw: dict | None, lines: list[str]) -> None:
    """Server->edge transfer: per-inference bytes the action expert conditions on (KV cache + state)."""
    if not bw:
        return
    det = bw.get("kv_cache_detail", {})
    lines.append("")
    lines.append("Server->edge transfer (VLM on server, action expert on edge):")
    kv = f"  prefix KV cache: {bw['kv_cache_mib']:.2f} MiB"
    if bw.get("kv_cache_measured_mib") is not None:
        kv += f"  (measured {bw['kv_cache_measured_mib']:.2f} MiB)"
    kv += (f"  [{det.get('layers')}L x K,V x {det.get('prefix_len')}tok x "
           f"kv_heads={det.get('kv_heads')} x d{det.get('head_dim')}, {det.get('dtype')}]")
    lines.append(kv)
    lines.append(f"  + pad mask {bw['prefix_pad_mask_bytes'] / 1024:.1f} KiB + state {bw['state_bytes']} B"
                 f"  =>  total {bw['total_conditioning_mib']:.2f} MiB / inference")
    if bw.get("required_bandwidth_mbytes_per_s") is not None:
        lines.append(f"  at {bw['freq_hz']:.1f} Hz  =>  {bw['required_bandwidth_mbytes_per_s']:.1f} MB/s "
                     f"({bw['required_bandwidth_mbit_per_s']:.0f} Mbit/s)")
    lines.append(f"  (alt split — vision on server: ship {bw['alt_vision_split_mib']:.2f} MiB image embeds)")


def _render_kernel_analysis(ka: dict | None, order: tuple[str, ...], lines: list[str]) -> None:
    """Per-phase x kernel-family (attention vs weights/activations), GEMM/GEMV split, and the
    system-overhead / utilization summary (launch overhead, GPU-busy vs wall, SM-fill)."""
    if not ka:
        return
    fam_ms = ka.get("per_phase_family_ms") or {}
    fam_order = [f for f in ka.get("family_order", []) if any(f in fam_ms.get(p, {}) for p in order)]
    if fam_ms and fam_order:
        lines.append("")
        lines.append("Per-phase x kernel-family (nsys SQLite projection, GPU ms/infer):")
        head = "  " + f"{'phase':8}" + "".join(f"{f:>12}" for f in fam_order) + f"{'total':>10}"
        lines.append(head)
        for p in order:
            fams = fam_ms.get(p)
            if not fams:
                continue
            cells = "".join(f"{fams.get(f, 0.0):>12.2f}" for f in fam_order)
            lines.append("  " + f"{p:8}" + cells + f"{sum(fams.values()):>10.2f}")
    gs = ka.get("gemm_split_ms") or {}
    if gs:
        parts = [f"{p}: compute={d.get('gemm_compute',0):.2f} gemv={d.get('gemv_memory',0):.2f}"
                 for p, d in gs.items() if (d.get('gemm_compute') or d.get('gemv_memory'))]
        if parts:
            lines.append("GEMM split — compute-bound GEMM vs memory-bound GEMV (batch-1 decode), ms/infer:")
            lines.append("  " + "   |   ".join(parts))
    s = ka.get("system") or {}
    if s:
        lines.append("System & overhead:")
        lines.append(f"  {s.get('kernels_per_infer')} kernels/infer replayed by "
                     f"{s.get('graph_launches_per_infer')} CUDA-graph + {s.get('eager_launches_per_infer')} "
                     f"eager launches (graphs amortize launch overhead)")
        util = s.get("gpu_utilization_under_nsys")
        nonpct = s.get("non_gpu_pct")
        parts = [f"GPU-busy={s.get('gpu_busy_ms_per_infer', 0):.2f}ms"]
        if nonpct is not None:
            parts.append(f"non-GPU(idle+overhead vs pristine wall)={s.get('non_gpu_ms_per_infer', 0):.2f}ms ({nonpct:.1f}%)")
        if util is not None:
            parts.append(f"util(under nsys)={100*util:.0f}%")
        lines.append("  " + "  |  ".join(parts))
        smc = s.get("sm_coverage_weighted")
        lines.append(f"  mean kernel={s.get('mean_kernel_us') or 0:.1f}us  median={s.get('median_kernel_us') or 0:.1f}us"
                     + (f"  |  SM-coverage (CTAs vs {s.get('sm_count')} SMs, time-wtd)={smc:.2f}" if smc is not None else ""))
        lines.append(f"  launch-API CPU time={s.get('launch_api_cpu_ms_per_infer', 0):.2f}ms "
                     "(async-overlapped with GPU; off the critical path)")
        tk = s.get("tiny_kernel_time_pct")
        kd = (f"  kernel dist: p90={s.get('p90_kernel_us') or 0:.1f}us  max={s.get('max_kernel_us') or 0:.1f}us"
              f"  |  idle-gap={s.get('idle_gap_ms_per_infer', 0):.2f}ms")
        if tk is not None:
            kd += f"  |  tiny(<5us)={100*s.get('tiny_kernel_frac', 0):.0f}% of kernels / {tk:.0f}% of GPU time"
        lines.append(kd)
        mc = s.get("memcpy_mib_per_infer")
        if mc is not None:
            lines.append(f"  data movement: memcpy={mc:.2f} MiB/infer (H2D {s.get('memcpy_h2d_mib_per_infer', 0):.2f},"
                         f" D2H {s.get('memcpy_d2h_mib_per_infer', 0):.2f}, D2D {s.get('memcpy_d2d_mib_per_infer', 0):.2f})"
                         f" + memset {s.get('memset_mib_per_infer', 0):.2f} MiB")


def write_outputs(output_prefix: str, meta: dict, result: dict) -> tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)) or ".", exist_ok=True)
    json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"

    with open(json_path, "w") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2, default=_json_default)

    # Long-format CSV: one row per (metric, phase) for easy plotting across a sweep.
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "config_name", "prompt_len", "num_steps", "metric", "phase", "ms", "pct"])
        base = [meta.get("system"), meta.get("config_name"), meta.get("prompt_len"), meta.get("num_steps")]

        def emit(metric: str, phases_ms: dict | None, phases_pct: dict | None):
            if not phases_ms:
                return
            for k, v in phases_ms.items():
                pct = (phases_pct or {}).get(k)
                w.writerow(base + [metric, k, "" if v is None else f"{v:.4f}",
                                   "" if pct is None else f"{pct:.2f}"])

        if result.get("headline_latency_ms") is not None:
            w.writerow(base + ["e2e", "headline", f"{result['headline_latency_ms']:.4f}",
                               result.get("headline_source", "")])
        seg = result.get("e2e_segmented_ms")
        if seg:
            w.writerow(base + ["e2e", "segmented", f"{seg['median']:.4f}", ""])
        nat = result.get("e2e_native_ms")
        if nat:
            w.writerow(base + ["e2e", "native", f"{nat['median']:.4f}", ""])
        if result.get("breakdown_nvtx"):
            emit("breakdown_nvtx", result["breakdown_nvtx"]["phases_ms_per_infer"],
                 result["breakdown_nvtx"].get("phases_pct"))
        if result.get("components_standalone"):
            emit("components_standalone", result["components_standalone"]["phases_ms_per_infer"],
                 result["components_standalone"].get("phases_pct"))
        if result.get("kernel_buckets"):
            emit("kernel_buckets", result["kernel_buckets"]["buckets_ms_per_infer"],
                 result["kernel_buckets"].get("buckets_pct"))

        # Deep analysis: per-phase x family (phase tagged as "phase:family"), GEMM/GEMV split.
        ka = result.get("kernel_analysis")
        if ka:
            for p, fams in (ka.get("per_phase_family_ms") or {}).items():
                for fam, v in fams.items():
                    w.writerow(base + ["per_phase_family", f"{p}:{fam}", f"{v:.4f}", ""])
            for p, d in (ka.get("gemm_split_ms") or {}).items():
                for k, v in d.items():
                    w.writerow(base + ["gemm_split", f"{p}:{k}", f"{v:.4f}", ""])
            for k, v in (ka.get("system") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    w.writerow(base + ["system", k, f"{v:.4f}", ""])

        # Server<->edge transfer sizing (bytes; rate in MB/s).
        bw = result.get("bandwidth")
        if bw:
            for k in ("kv_cache_bytes", "kv_cache_bytes_measured", "total_conditioning_bytes",
                      "alt_vision_split_bytes"):
                if bw.get(k) is not None:
                    w.writerow(base + ["bandwidth", k, str(bw[k]), ""])
            if bw.get("required_bandwidth_mbytes_per_s") is not None:
                w.writerow(base + ["bandwidth", "required_mbytes_per_s",
                                   f"{bw['required_bandwidth_mbytes_per_s']:.4f}", ""])

    return json_path, csv_path


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(o)
