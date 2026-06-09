"""Deep kernel + system analysis from the nsys-exported SQLite (graphs ON, single capture).

`nsys stats` gives global summaries; the questions "attention vs weights/activations, backbone vs
action part" and "launch overhead / utilization" need the kernel timeline CROSSED with the NVTX
phases — which `nsys stats` won't do. We read the SQLite directly.

The one non-obvious thing (and why naive time-overlap fails): with CUDA graphs the kernels execute
on the GPU timeline asynchronously, LONG after the eager NVTX range was pushed/popped on the CPU. So
a kernel's GPU timestamp does NOT fall inside its phase's NVTX window. The fix is exactly nsys's own
projection: every kernel carries a `correlationId` linking it to the CUDA runtime launch
(`cudaGraphLaunch` for graph replays, `cudaLaunchKernel` for eager glue) that issued it; that launch
DOES fall inside the phase's NVTX CPU window. So we attribute kernel -> launch -> NVTX phase.
Validated against `nvtx_gpu_proj_sum`: this reproduces the per-phase GPU time (kernels + memops) and
the global kernel total to the millisecond.

Outputs (all per-inference):
  * per_phase_family   — phase x {attention, gemm, conv, elementwise, memory_ops, other} GPU ms.
                         Answers attention-vs-W/A within backbone(vision+vlm) vs action.
  * gemm_split         — within gemm, compute-GEMM vs memory-bound GEMV (batch-1 decode), per phase.
  * system             — kernels/infer, graph vs eager launches, GPU-busy vs wall (utilization),
                         launch-API CPU time, mean/median kernel duration, and an SM-coverage proxy
                         (are there enough CTAs to cover the SMs?).
Never raises: on any failure returns {"ok": False, "error": ...}.
"""

from __future__ import annotations

import bisect
import contextlib
import os
import re
import sqlite3
import statistics
import subprocess

from .nsys import _family_of, nsys_bin

_FAMILY_ORDER = ("attention", "gemm", "conv", "quantize", "elementwise", "memory_ops", "other")
_TEMPLATE_ARGS = re.compile(r"<[^<>]*>")


def _identifier(name: str) -> str:
    """Kernel identifier with all <...> template args stripped (innermost-first, handles nesting).

    Needed because a cuBLAS GEMM kernel can carry a *descriptor type* like
    `cublasGemvTensorStridedBatched` inside its template args — matching "gemv" against the raw
    demangled name would misclassify that GEMM as a GEMV. Stripping template args leaves just the
    real kernel identifier (e.g. `internal::gemvx::kernel`), which is what we test."""
    prev = None
    out = name
    while out != prev:
        prev = out
        out = _TEMPLATE_ARGS.sub("", out)
    return out


def _ensure_sqlite(rep_path: str) -> str:
    """Return the .sqlite for a .nsys-rep, exporting it if nsys stats hasn't already."""
    base = rep_path[:-9] if rep_path.endswith(".nsys-rep") else rep_path
    sqlite_path = base + ".sqlite"
    if os.path.exists(sqlite_path):
        return sqlite_path
    nsys = nsys_bin()
    if nsys is None:
        raise FileNotFoundError("nsys not found; cannot export sqlite")
    r = subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                        "--output", sqlite_path, rep_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nsys export failed: {r.stderr.strip() or r.stdout.strip()}")
    return sqlite_path


def _gemm_is_gemv(name: str) -> bool:
    # cuBLAS dispatches GEMV (matrix-vector) for batch-1 / tiny-M matmuls precisely because they are
    # MEMORY-bound (the decode regime), vs the tiled tensor-core GEMMs that are compute-bound. Test
    # the template-stripped identifier so the cuBLAS descriptor type doesn't cause false positives.
    return "gemv" in _identifier(name).lower()


def _merge_busy(intervals: list[tuple[int, int]]) -> tuple[int, int, int]:
    """Union coverage, min-start and max-end (ns) of a set of [start,end] intervals (handles overlap)."""
    if not intervals:
        return 0, 0, 0
    intervals.sort()
    busy = 0
    cs, ce = intervals[0]
    lo = cs
    for s, e in intervals[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    return busy, lo, ce


def analyze_sqlite(rep_or_sqlite: str, *, iters: int, phases: tuple[str, ...], sm_count: int = 84,
                   pristine_wall_ms: float | None = None) -> dict:
    try:
        sqlite_path = rep_or_sqlite if rep_or_sqlite.endswith(".sqlite") else _ensure_sqlite(rep_or_sqlite)
        n = max(int(iters), 1)
        with contextlib.closing(sqlite3.connect(sqlite_path)) as con:
            cur = con.cursor()

            # --- NVTX phase windows (CPU timeline), correlationId -> launch start, string table. ---
            nvtx: dict[str, list[tuple[int, int]]] = {}
            for nm, st, en in cur.execute(
                "SELECT COALESCE(s.value, e.text), e.start, e.end FROM NVTX_EVENTS e "
                "LEFT JOIN StringIds s ON e.textId = s.id WHERE e.eventType = 59"
            ):
                if nm in phases:
                    nvtx.setdefault(nm, []).append((st, en))
            for p in nvtx:
                nvtx[p].sort()
            starts_by_phase = {p: [s for s, _ in ivs] for p, ivs in nvtx.items()}

            launch_start = dict(cur.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
            strings = dict(cur.execute("SELECT id, value FROM StringIds"))

            def phase_of(corr: int) -> str | None:
                t = launch_start.get(corr)
                if t is None:
                    return None
                for p, ivs in nvtx.items():
                    i = bisect.bisect_right(starts_by_phase[p], t) - 1
                    if i >= 0 and ivs[i][0] <= t <= ivs[i][1]:
                        return p
                return None

            # --- Kernels: family + gemv split + phase, SM-coverage (CTAs vs SMs), busy intervals. ---
            per_phase_family: dict[str, dict[str, float]] = {p: {} for p in phases}
            gemm_split: dict[str, dict[str, float]] = {p: {"gemm_compute": 0.0, "gemv_memory": 0.0} for p in phases}
            durations_ns: list[int] = []
            busy_intervals: list[tuple[int, int]] = []
            smcov_num = smcov_den = 0.0  # kernel-time-weighted SM coverage
            attributed_kernel_ns = 0     # kernel time landing in SOME phase (coverage signal)

            for corr, st, en, dname, sname, gx, gy, gz in cur.execute(
                "SELECT correlationId, start, end, demangledName, shortName, gridX, gridY, gridZ "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ):
                dur = en - st
                durations_ns.append(dur)
                busy_intervals.append((st, en))
                name = strings.get(dname) or strings.get(sname) or ""
                fam = _family_of(name)
                p = phase_of(corr)
                if p is not None:
                    attributed_kernel_ns += dur
                    per_phase_family[p][fam] = per_phase_family[p].get(fam, 0.0) + dur
                    if fam == "gemm":
                        key = "gemv_memory" if _gemm_is_gemv(name) else "gemm_compute"
                        gemm_split[p][key] += dur
                blocks = (gx or 1) * (gy or 1) * (gz or 1)
                smcov_num += min(blocks, sm_count) / sm_count * dur
                smcov_den += dur

            # --- Memcpy / memset: GPU memory ops, attributed to phases too (family memory_ops). ---
            for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
                try:
                    rows = cur.execute(f"SELECT correlationId, start, end FROM {table}").fetchall()
                except sqlite3.OperationalError:
                    rows = []
                for corr, st, en in rows:
                    busy_intervals.append((st, en))
                    p = phase_of(corr)
                    if p is not None:
                        per_phase_family[p]["memory_ops"] = per_phase_family[p].get("memory_ops", 0.0) + (en - st)

            # --- Launch / API counts + CPU time (async-overlapped; see note in `system` below). ---
            api = {strings.get(nid, str(nid)).split("_v")[0]: (cnt, dur) for nid, cnt, dur in cur.execute(
                "SELECT nameId, COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId")}
            graph_launches = api.get("cudaGraphLaunch", (0, 0))[0]
            eager_launches = api.get("cudaLaunchKernel", (0, 0))[0]
            launch_api_ns = (api.get("cudaGraphLaunch", (0, 0))[1] or 0) + (api.get("cudaLaunchKernel", (0, 0))[1] or 0)

            # Wall = span of ALL GPU activity (kernels + memops), so busy <= wall by construction.
            gpu_busy_ns, span_lo, span_hi = _merge_busy(busy_intervals)
            wall_ns = span_hi - span_lo
            n_kernels = len(durations_ns)
            total_kernel_ns = sum(durations_ns)
            # Fraction of kernel time that landed in SOME NVTX phase. If this is ~0 the per-phase
            # split silently failed (NVTX/schema drift, or a backend with no phases) even though
            # kernels exist — without this signal an all-empty split looks identical to a good one.
            attributed_frac = (attributed_kernel_ns / total_kernel_ns) if total_kernel_ns else 0.0

        def msinf(ns: float) -> float:
            return ns / 1e6 / n

        per_phase_family = {p: {f: msinf(v) for f, v in fams.items()} for p, fams in per_phase_family.items()}
        gemm_split_ms = {p: {k: msinf(v) for k, v in d.items()} for p, d in gemm_split.items()}

        system = {
            "kernels_per_infer": round(n_kernels / n, 1),
            "graph_launches_per_infer": round(graph_launches / n, 1),
            "eager_launches_per_infer": round(eager_launches / n, 1),
            "gpu_busy_ms_per_infer": msinf(gpu_busy_ns),
            "gpu_active_span_ms_per_infer": msinf(wall_ns),
            "gpu_utilization_under_nsys": (gpu_busy_ns / wall_ns) if wall_ns else None,
            # CPU residence time inside the launch calls. With CUDA graphs this is ASYNC-OVERLAPPED
            # with GPU execution (it can block only when the submit queue is full), so it is NOT on
            # the critical path — see non_gpu_* below for the real wall-clock overhead.
            "launch_api_cpu_ms_per_infer": msinf(launch_api_ns),
            "mean_kernel_us": (statistics.fmean(durations_ns) / 1e3) if durations_ns else None,
            "median_kernel_us": (statistics.median(durations_ns) / 1e3) if durations_ns else None,
            # Grid-level SM coverage only (enough CTAs to put >=1 block on each SM); time-weighted.
            # Saturates at 1.0 once blocks>=SMs and does NOT capture intra-SM warp/register occupancy.
            "sm_coverage_weighted": (smcov_num / smcov_den) if smcov_den else None,
            "sm_count": sm_count,
            # Coverage of the per-phase split: ~1.0 = every kernel attributed to a phase; near 0 =
            # the NVTX projection failed (don't trust per_phase_family/gemm_split then).
            "phase_attributed_frac": round(attributed_frac, 4),
        }
        # Real wall-clock overhead: PRISTINE (non-nsys) wall minus GPU-busy. A small NEGATIVE value
        # is possible (CUPTI slightly inflates GPU-busy vs the un-profiled run) — reported signed, not
        # clamped, so it stays honest; `non_gpu_valid` flags when the comparison is trustworthy.
        if pristine_wall_ms:
            busy_ms = msinf(gpu_busy_ns)
            system["pristine_wall_ms_per_infer"] = pristine_wall_ms
            system["non_gpu_ms_per_infer"] = pristine_wall_ms - busy_ms
            system["non_gpu_pct"] = 100.0 * (pristine_wall_ms - busy_ms) / pristine_wall_ms
            system["non_gpu_valid"] = busy_ms <= pristine_wall_ms

        return {
            "ok": n_kernels > 0,
            "method": "nsys-sqlite corrId->launch->NVTX projection",
            "sqlite": sqlite_path,
            "per_phase_family_ms": per_phase_family,
            "family_order": list(_FAMILY_ORDER),
            "gemm_split_ms": gemm_split_ms,
            "system": system,
            "error": None if n_kernels > 0 else "no kernels in sqlite",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
