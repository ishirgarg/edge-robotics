"""Minimal interface a profilable system must provide.

Light on abstraction by design. The profiler runs the model in its REAL graphs-on form and uses
nvidia/CUDA tooling (Nsight Systems) to attribute time — it never runs the model eager just to see
inside it. A loaded system therefore exposes up to three graphs-on callables plus metadata:

  * `infer_segmented` — the headline E2E inference, structured so a single nsys capture can be
    split per component. For the torch backend this is the model run as SEPARATELY-compiled
    per-phase CUDA graphs glued in eager python, each wrapped in an NVTX range (vision / vlm /
    action). nsys `nvtx_gpu_proj_sum` then attributes GPU time per phase. (See profiling/nsys.py
    for why per-phase graphs + eager NVTX is the only thing that survives CUDA graphs.)
  * `infer_native` — OPTIONAL cross-check: the model's own single fused torch.compile path. Same
    result, no NVTX split; the wall gap vs `infer_segmented` is the segmentation overhead.
  * `component_profiler` — OPTIONAL: time each component standalone in its fully-optimized
    (graphs-on) form, returning ms/infer per phase. Independent of the E2E NVTX split.

`nvtx_phases` lists the NVTX range names the segmented path emits (empty for an opaque fused graph
that can't be split, e.g. realtime-vla — that backend degrades to kernel-family buckets only).
`block(out)` forces device completion for honest wall timing.
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
    prompt_len: int
    num_steps: int
    batch_size: int
    # Headline graphs-on E2E. For the torch backend this emits NVTX phase ranges so one nsys
    # capture yields the component split; it is ALSO the callable timed for the headline wall.
    infer_segmented: Callable[[], Any]
    block: Callable[[Any], Any]
    nvtx_phases: tuple[str, ...] = ()
    # Optional cross-check: the model's native single-compile E2E (no NVTX split).
    infer_native: Callable[[], Any] | None = None
    # Optional per-component standalone graphs-on timing: component_profiler(warmup, iters) -> dict.
    component_profiler: Callable[..., dict] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ProfiledSystem(abc.ABC):
    @abc.abstractmethod
    def load(
        self,
        *,
        config_name: str,
        checkpoint: str,
        prompt_len: int,
        num_steps: int,
        batch_size: int,
    ) -> LoadedSystem:
        """Load the model and return a LoadedSystem. Mimics the repo's own inference path."""
