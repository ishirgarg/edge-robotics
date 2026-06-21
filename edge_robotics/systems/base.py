"""Minimal interface a profilable system must provide.

Light on abstraction by design. The profiler runs the model in its REAL graphs-on form and uses
nvidia/CUDA tooling (Nsight Systems) to attribute time — it never runs the model eager just to see
inside it. A loaded system therefore exposes up to three graphs-on callables plus metadata:

  * `infer_native` — the FAITHFUL deployed E2E: the model's own single fused torch.compile path
    (max-autotune), exactly as the repo ships it. This is the HEADLINE latency/throughput number —
    the latency the robot actually experiences — because it carries no segmentation artifacts.
  * `infer_segmented` — the same inference restructured so a single nsys capture can be split per
    component: SEPARATELY-compiled per-phase callables glued in eager python, each wrapped in an
    NVTX range (vision / vlm / action). nsys `nvtx_gpu_proj_sum` then attributes GPU time per phase.
    (See profiling/nsys.py for why per-phase graphs + eager NVTX is the only thing that survives
    CUDA graphs.) Its wall carries glue/clone overhead, so it is the BREAKDOWN VEHICLE, not the
    headline; the gap vs `infer_native` (same compile mode) is the segmentation overhead.
  * `component_profiler` — OPTIONAL secondary cross-check ONLY: each component timed standalone in
    its graphs-on form (ms/infer). No cross-phase overlap, so it will differ from the within-run
    NVTX split — it must never be used as the authoritative breakdown.

`nvtx_phases` lists the NVTX range names the segmented path emits — the authoritative, per-system
source of phase identity (the offline `parse`/`report` stages read it back from persisted meta, so
nothing hardcodes a phase list). Empty tuple => opaque fused graph that can't be NVTX-split, which
degrades to kernel-family buckets only. (Both current backends DO split into vision/vlm/action;
realtime-vla achieves it by re-capturing its fused graph as three per-stage sub-graphs.)
`block(out)` forces device completion for honest wall timing.

`meta` is the carrier for everything the offline stages need without re-loading
the model: dims (action_horizon/action_dim/max_token_len/n_images/paligemma_variant/...), the model
family axes (`pi05`, `discrete_state_input`, `proprioception` in {suffix_state, prompt_discrete,
none}), the precision/quant axis (`compute_dtype`), and the compile modes.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedSystem:
    """A loaded model ready to profile, in its real graphs-on form."""

    name: str
    config_name: str
    prompt_len: int  # the EFFECTIVE max_token_len this run used (resolved from the config if not overridden)
    num_steps: int  # the EFFECTIVE flow steps actually run (a backend may lock this, e.g. realtime-vla=10)
    batch_size: int
    # Graphs-on E2E restructured into per-phase NVTX ranges so one nsys capture yields the component
    # split. The BREAKDOWN VEHICLE (carries glue/clone overhead) — not the headline wall.
    infer_segmented: Callable[[], Any]
    block: Callable[[Any], Any]
    nvtx_phases: tuple[str, ...] = ()
    # The faithful deployed single-compile E2E (max-autotune). The HEADLINE latency/throughput.
    infer_native: Callable[[], Any] | None = None
    # Optional per-component standalone graphs-on timing: component_profiler(warmup, iters) -> dict.
    # SECONDARY cross-check only (no cross-phase overlap); never the authoritative breakdown.
    component_profiler: Callable[..., dict] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ProfiledSystem(abc.ABC):
    @abc.abstractmethod
    def load(
        self,
        *,
        config_name: str,
        checkpoint: str,
        prompt_len: int | None,
        num_steps: int,
        batch_size: int,
    ) -> LoadedSystem:
        """Load the model and return a LoadedSystem. Mimics the repo's own inference path.

        `prompt_len` is `None` to use the config's native `max_token_len` (the faithful "run as
        trained" default — pi0=48, pi05=200); pass an int only to override it for a sweep.
        """
