"""CLI config + orchestration for the edge-robotics profiler.

One invocation profiles ONE (system, config, prompt_len). Sweeps are the shell's job
(profile_sweep.sh). The model always runs in its real graphs-on form; attribution comes from
Nsight Systems. Because nsys must wrap the whole process and CUPTI perturbs wall timing, the work
is split across four cheap modes the shell drives in sequence:

  time   : load model, time segmented E2E (the breakdown vehicle) + per-component standalone
           -> <out>.timing.json. NOT under nsys, so the wall numbers are pristine. (The deployed
           native E2E = the HEADLINE latency is timed separately by `time-native`.)
  nsys   : load model, warmup, bracket ONE steady-state segmented inference with cudaProfilerStart/
           Stop + NVTX. Run by the shell UNDER `nsys profile`; nsys writes <out>.nsys-rep.
  parse  : no model load. Parse <out>.nsys-rep -> NVTX per-phase split + kernel-family buckets ->
           <out>.breakdown.json.
  report : no model load. Merge timing + breakdown -> <out>.json / <out>.csv and print a summary.

This module is import-light: defining Config must not import torch/openpi, so the entrypoint can
pin CUDA_VISIBLE_DEVICES before any CUDA context is created. Heavy imports live inside run().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Fallback NVTX phase names for the offline `parse`/`report` stages when the persisted meta lacks
# them. The AUTHORITATIVE source is LoadedSystem.nvtx_phases, recorded into timing.json by `time`.
_DEFAULT_PHASES: tuple[str, ...] = ("vision", "vlm", "action")


@dataclass
class Config:
    # --- required identity flags ---
    system: Literal["openpi_torch", "realtime_vla", "pi05_openpi_torch", "pi05_realtimevla"]
    """Backend to profile: 'openpi_torch' (openpi's native PyTorch port; serves any pi0/pi05 x
    dataset config, full NVTX phase split) or 'realtime_vla' (dexmal/realtime-vla Triton; pi05 only,
    NVTX split via re-captured per-stage sub-graphs). The 'pi05_*' names are accepted aliases."""
    config_name: str
    """openpi config name = model-family x dataset (e.g. pi0_droid, pi0_aloha, pi05_droid,
    pi05_aloha, pi05_libero; debug_pi05 for a fast smoke test). Resolved via openpi get_config."""
    checkpoint: str
    """Checkpoint: local path/gs:// URI, or the literal 'random' for random-init (no download)."""
    gpu: int
    """CUDA device index to pin."""

    # --- operational knobs ---
    mode: Literal["time", "time-native", "nsys", "parse", "report"] = "time"
    """Pipeline stage (see module docstring). The shell drives them in sequence. `time-native` is a
    SEPARATE process from `time` on purpose: openpi's single fused graph and the per-phase segmented
    graphs thrash the shared cudagraph pool if exercised together, so the native cross-check is run
    in isolation."""
    prompt_len: int | None = None
    """Override max_token_len for this run. None (default) uses the config's native value
    ('as trained': pi0=48, pi05=200). Set an int only to sweep prompt length."""
    num_steps: int = 10
    """Flow-matching denoise steps."""
    warmup: int = 3
    """Warmup iterations (absorb torch.compile + cudagraph capture) before timing."""
    iters: int = 20
    """Timed iterations (also the #inferences bracketed for nsys)."""
    batch_size: int = 1
    """Inference batch size."""
    output: str = "out/profile"
    """Output prefix. Writes <prefix>.{timing,breakdown}.json, <prefix>.nsys-rep, <prefix>.{json,csv}."""
    nsys_rep: str | None = None
    """Path to the .nsys-rep for `parse` (default <output>.nsys-rep)."""


def _make_system(name: str):
    if name in ("realtime_vla", "pi05_realtimevla"):
        from .systems.realtime_vla import RealtimeVlaSystem

        return RealtimeVlaSystem()
    if name in ("openpi_torch", "pi05_openpi_torch"):
        from .systems.openpi_torch import OpenpiTorchSystem

        return OpenpiTorchSystem()
    raise ValueError(f"unknown system '{name}'")


def _load(config: Config):
    system = _make_system(config.system)
    return system.load(
        config_name=config.config_name,
        checkpoint=config.checkpoint,
        prompt_len=config.prompt_len,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
    )


def _environment_meta(config: Config) -> dict:
    """Full reproducibility manifest: hardware target, driver/toolkit, package versions, git state,
    the exact invocation, and the env vars that change results. Gathered once, in the `time` stage.
    Best-effort throughout — a missing tool degrades a field to None, never breaks a run."""
    import importlib.metadata as ilm
    import os
    import platform
    import socket
    import subprocess
    import sys
    import time

    import torch

    def _pkg(name: str):
        try:
            return ilm.version(name)
        except Exception:  # noqa: BLE001
            return None

    def _sh(cmd: list[str]):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_commit = _sh(["git", "-C", repo, "rev-parse", "HEAD"])
    git_dirty = bool(_sh(["git", "-C", repo, "status", "--porcelain"]))

    # GPU/driver: torch for the device torch actually uses; nvidia-smi for the host driver string.
    dev: dict = {}
    driver_api = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        dev = {
            "name": p.name,
            "compute_capability": f"{p.major}.{p.minor}",
            "total_memory_mib": round(p.total_memory / (1024**2)),
            "multi_processor_count": p.multi_processor_count,
            "uuid": str(getattr(p, "uuid", None)) if getattr(p, "uuid", None) else None,
        }
        try:
            driver_api = torch._C._cuda_getDriverVersion()  # noqa: SLF001  (e.g. 12060)
        except Exception:  # noqa: BLE001
            driver_api = None
    nvidia_driver = _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nvidia_driver:
        nvidia_driver = nvidia_driver.splitlines()[0].strip()

    from .profiling.nsys import nsys_bin

    nsys_path = nsys_bin()
    nsys_ver = None
    if nsys_path:
        v = _sh([nsys_path, "--version"])
        nsys_ver = v.splitlines()[0].strip() if v else None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "invocation": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": {"commit": git_commit, "dirty": git_dirty, "repo": repo},
        "hardware": {
            "gpu_index_pinned": config.gpu,
            "device": dev,
            "nvidia_driver_version": nvidia_driver,
            "cuda_driver_api_version": driver_api,
            "cuda_toolkit_torch_built": torch.version.cuda,
        },
        "packages": {
            "torch": torch.__version__,
            "triton": _pkg("triton"),
            "transformers": _pkg("transformers"),
            "numpy": _pkg("numpy"),
            "openpi": _pkg("openpi"),
            "jax": _pkg("jax"),
            "jaxlib": _pkg("jaxlib"),
            "nsys": nsys_ver,
        },
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "env": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES", "OPENPI_TORCH_COMPILE_MODE", "OPENPI_TORCH_NATIVE_COMPILE_MODE",
                "CROSS_CHECK_NATIVE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_FX_GRAPH_CACHE", "JAX_PLATFORMS",
            )
        },
    }


def _top_meta(config: Config, loaded) -> dict:
    m = loaded.meta
    return {
        "system": loaded.name,  # canonical backend id (config.system may be an alias)
        "config_name": config.config_name,
        "checkpoint": m.get("checkpoint"),
        "real_weights": m.get("real_weights", False),
        "device_kind": m.get("device_kind"),
        "n_devices": m.get("n_devices"),
        # EFFECTIVE values actually run (a backend may resolve/lock these), not the raw request.
        "prompt_len": loaded.prompt_len,
        "num_steps": loaded.num_steps,
        "batch_size": config.batch_size,
        "warmup": config.warmup,
        "iters": config.iters,
        "compile_mode": m.get("compile_mode"),
        "native_compile_mode": m.get("native_compile_mode"),
        "graphs_on": m.get("graphs_on"),
        "backend": m.get("backend"),
        "backend_version": m.get("torch_version"),
        "attribution": m.get("attribution", "n/a"),
        # Authoritative phase identity from the loaded system; offline stages read this back.
        "nvtx_phases": list(loaded.nvtx_phases),
        "model_weight_dtype": m.get("dtype"),
        "compute_dtype": m.get("compute_dtype", m.get("dtype")),
        "proprioception": m.get("proprioception"),
        "discrete_state_input": m.get("discrete_state_input"),
        "model": {
            k: m.get(k)
            for k in (
                "action_horizon", "action_dim", "paligemma_variant", "action_expert_variant",
                "dtype", "compute_dtype", "n_images", "prefix_len_nominal", "max_token_len", "pi05",
                "discrete_state_input", "proprioception", "kv_cache_bytes_measured",
            )
        },
        "environment": _environment_meta(config),
    }


def run(config: Config) -> None:
    dispatch = {
        "time": _run_time, "time-native": _run_time_native,
        "nsys": _run_nsys, "parse": _run_parse, "report": _run_report,
    }
    if config.mode not in dispatch:
        raise ValueError(f"unknown mode '{config.mode}'")
    dispatch[config.mode](config)


def _run_time(config: Config) -> None:
    """Pristine wall timing (no nsys): segmented E2E (breakdown vehicle) + per-component standalone.

    Deliberately does NOT exercise the native path — that shares the cudagraph pool with the
    per-phase graphs and would thrash. The native cross-check is its own process (`time-native`).
    """
    from .profiling.walltime import time_callable

    loaded = _load(config)
    print(f"[time] {config.system}/{config.config_name} L={config.prompt_len} — timing segmented E2E...")
    seg = time_callable(loaded.infer_segmented, loaded.block, warmup=config.warmup, iters=config.iters)

    comp = None
    if loaded.component_profiler is not None:
        print("[time] per-component standalone (graphs ON)...")
        comp = loaded.component_profiler(warmup=config.warmup, iters=config.iters)

    payload = {
        "meta": _top_meta(config, loaded),
        "e2e_segmented_ms": seg,
        "components_standalone": comp,
    }
    path = f"{config.output}.timing.json"
    _write_json(path, payload)
    sm = sorted(seg)[len(seg) // 2]
    print(f"[time] segmented median ~{sm:.2f} ms -> wrote {path}")


def _run_time_native(config: Config) -> None:
    """Cross-check ONLY: time openpi's native single-compile E2E, in isolation (own process).

    Run separately from `time` so the native fused graph never coexists with the per-phase graphs
    (which makes dynamo recompile every call). The native graph also needs more cudagraph warmup.
    """
    from .profiling.walltime import time_callable

    loaded = _load(config)
    if loaded.infer_native is None:
        print(f"[time-native] {config.system} has no native single-compile path; skipping.")
        _write_json(f"{config.output}.timing-native.json", {"e2e_native_ms": None})
        return
    nat_warmup = max(config.warmup, 8)
    print(f"[time-native] timing native single-compile E2E (warmup={nat_warmup})...")
    nat = time_callable(loaded.infer_native, loaded.block, warmup=nat_warmup, iters=config.iters)
    _write_json(f"{config.output}.timing-native.json", {"e2e_native_ms": nat})
    nm = sorted(nat)[len(nat) // 2]
    print(f"[time-native] native median ~{nm:.2f} ms")


def _run_nsys(config: Config) -> None:
    """Bracket exactly one steady-state run of the segmented E2E for nsys to capture."""
    from .profiling.nsys import profiler_capture, under_nsys

    if not under_nsys():
        print("[nsys] WARNING: not running under `nsys profile` — no capture will be produced. "
              "Launch via profile_one.sh (it wraps this process in nsys).")
    loaded = _load(config)
    print(f"[nsys] warmup x{config.warmup}, then capturing {config.iters} segmented inferences...")
    for _ in range(config.warmup):
        loaded.block(loaded.infer_segmented())
    with profiler_capture():
        for _ in range(config.iters):
            loaded.infer_segmented()
    print("[nsys] capture window closed.")


def _run_parse(config: Config) -> None:
    """Parse the nsys report into NVTX per-phase split + kernel-family buckets + the DEEP analysis
    (per-phase x kernel-family, GEMM/GEMV split, launch/utilization overheads) — all no model load."""
    from .profiling.kernel_analysis import analyze_sqlite
    from .profiling.nsys import parse_kernel_buckets, parse_nvtx_gpu_proj

    rep = config.nsys_rep or f"{config.output}.nsys-rep"
    # Phases + pristine wall + SM count all come from timing.json (written by `time`); parse does not
    # load the model. Phases are the loaded system's own nvtx_phases (authoritative), not a table.
    timing = _read_json(f"{config.output}.timing.json") or {}
    tmeta = timing.get("meta") or {}
    # Distinguish an intentionally-empty phase list ([] = opaque graph, buckets-only) from a missing
    # key (fall back to the default split). `or` would wrongly turn [] into the default.
    _ph = tmeta.get("nvtx_phases")
    phases = tuple(_ph) if _ph is not None else _DEFAULT_PHASES
    breakdown = parse_nvtx_gpu_proj(rep, iters=config.iters, phases=phases) if phases else None
    buckets = parse_kernel_buckets(rep, iters=config.iters)

    seg = timing.get("e2e_segmented_ms")
    pristine = sorted(seg)[len(seg) // 2] if seg else None
    sm = (((tmeta.get("environment") or {}).get("hardware") or {}).get("device") or {}).get(
        "multi_processor_count")
    kernel_analysis = analyze_sqlite(
        rep, iters=config.iters, phases=phases,
        sm_count=int(sm) if sm else 84, pristine_wall_ms=pristine)

    payload = {"breakdown_nvtx": breakdown, "kernel_buckets": buckets, "kernel_analysis": kernel_analysis}
    path = f"{config.output}.breakdown.json"
    _write_json(path, payload)
    if breakdown and breakdown.get("ok"):
        p = breakdown["phases_ms_per_infer"]
        ssum = sum(v for v in p.values() if v)
        print(f"[parse] NVTX vision={p.get('vision',0):.2f} vlm={p.get('vlm',0):.2f} "
              f"action={p.get('action',0):.2f} ms (Σ={ssum:.2f} vs {breakdown.get('total_gpu_ms_per_infer',0):.2f}ms GPU kernels)")
    elif phases:
        print(f"[parse] NVTX breakdown unavailable: {(breakdown or {}).get('error')}")
    if buckets.get("ok"):
        print(f"[parse] kernel buckets: { {k: round(v,2) for k,v in buckets['buckets_ms_per_infer'].items()} }")
    if kernel_analysis and kernel_analysis.get("ok"):
        s = kernel_analysis["system"]
        frac = s.get("phase_attributed_frac")
        if frac is not None and frac < 0.9:
            print(f"[parse] WARNING: only {100*frac:.0f}% of kernel time attributed to NVTX phases — "
                  "the per-phase split may be unreliable (NVTX/schema drift or an opaque graph).")
        print(f"[parse] system: {s['kernels_per_infer']} kernels/infer via {s['graph_launches_per_infer']} graph "
              f"+ {s['eager_launches_per_infer']} eager launches; GPU-busy {s['gpu_busy_ms_per_infer']:.2f}ms; "
              f"non-GPU {s.get('non_gpu_pct', float('nan')):.1f}%; SM-coverage {s.get('sm_coverage_weighted', 0):.2f}")
    elif kernel_analysis:
        print(f"[parse] deep kernel analysis unavailable: {kernel_analysis.get('error')}")
    print(f"[parse] wrote {path}")


def _run_report(config: Config) -> None:
    """Merge timing + breakdown into the final <output>.{json,csv} and print a summary."""
    import os

    from .metrics import build_result
    from .report import render_summary, write_outputs

    from . import roofline as _roofline

    timing = _read_json(f"{config.output}.timing.json") or {}
    native = _read_json(f"{config.output}.timing-native.json") or {}
    breakdown = _read_json(f"{config.output}.breakdown.json") or {}
    meta = timing.get("meta") or {"system": config.system, "config_name": config.config_name,
                                  "prompt_len": config.prompt_len, "num_steps": config.num_steps,
                                  "batch_size": config.batch_size, "iters": config.iters}

    # Roofline: ideal lower bound from model dims + hardware peaks, merged with the MEASURED per-phase
    # GPU time (nsys NVTX projection) and the pristine E2E wall -> MFU/MBU and how-far-from-ideal.
    bd_nvtx = breakdown.get("breakdown_nvtx") or {}
    phases_gpu_ms = bd_nvtx.get("phases_ms_per_infer") if bd_nvtx.get("ok") else None
    # Roofline-vs-wall uses the DEPLOYED (native, headline) wall; per-phase GPU times come from the
    # segmented nsys capture. Fall back to the segmented wall if the native cross-check wasn't run.
    def _median(xs):
        return sorted(xs)[len(xs) // 2] if xs else None

    e2e_wall = _median(native.get("e2e_native_ms")) or _median(timing.get("e2e_segmented_ms"))
    try:
        roofline = _roofline.analyze(meta, phases_gpu_ms=phases_gpu_ms, e2e_wall_ms=e2e_wall)
    except Exception as exc:  # noqa: BLE001  — roofline is supplementary; never gate the core report
        print(f"[report] WARNING: roofline computation failed ({exc}); omitting.")
        roofline = None

    # Server<->edge transfer sizing: per-inference bytes the action expert conditions on (prefix KV
    # cache + masks + state) if the VLM ran on a server. Rate uses the deployed (headline) freq.
    from . import bandwidth as _bandwidth
    try:
        bandwidth = _bandwidth.analyze(meta, freq_hz=(1000.0 / e2e_wall if e2e_wall else None))
    except Exception as exc:  # noqa: BLE001
        print(f"[report] WARNING: bandwidth sizing failed ({exc}); omitting.")
        bandwidth = None

    result = build_result(
        e2e_segmented=timing.get("e2e_segmented_ms"),
        e2e_native=native.get("e2e_native_ms"),
        breakdown_nvtx=breakdown.get("breakdown_nvtx"),
        kernel_buckets=breakdown.get("kernel_buckets"),
        components_standalone=timing.get("components_standalone"),
        kernel_analysis=breakdown.get("kernel_analysis"),
        roofline=roofline,
        bandwidth=bandwidth,
    )
    print("\n" + render_summary(meta, result))
    json_path, csv_path = write_outputs(config.output, meta, result)

    # Standalone reproducibility manifest (hardware target, driver/toolkit, package versions, git
    # state, dtype, model config, env vars, invocation) — same `meta` the results carry, written as
    # its own file so it travels with the run. Restores the old pi05_profile.sh config.json.
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(config.output)) or ".", "config.json")
    _write_json(cfg_path, meta)
    print(f"\n[report] wrote {json_path}, {csv_path}, and {cfg_path}")


def _write_json(path: str, payload: dict) -> None:
    import json
    import os

    from .report import _json_default

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _read_json(path: str) -> dict | None:
    import json
    import os

    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    """Console-script entry. Prefer scripts/profile_policy.py for GPU pinning."""
    import os

    import tyro

    config = tyro.cli(Config)
    # Hard-pin BEFORE any CUDA context is created: --gpu must win. Warn (don't silently honor) a
    # conflicting pre-set value, which would otherwise run on the wrong GPU while meta records --gpu.
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset not in (None, "", str(config.gpu)):
        print(f"[cli] WARNING: CUDA_VISIBLE_DEVICES={preset!r} preset; overriding with --gpu {config.gpu}.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    run(config)


if __name__ == "__main__":
    main()
