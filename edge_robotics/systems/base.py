"""Minimal interface a profilable system must provide.

Light on abstraction by design: a system just knows how to (1) load itself and (2) hand
back a callable that runs ONE inference at a given number of denoise steps, plus a `block`
function the timers use to force device completion.

Phase attribution (Vision/VLM/Action) is the SYSTEM's responsibility, exposed via the optional
`phase_profiler` callable on `LoadedSystem` — the orchestration loop (cli.run) is framework-agnostic
and just calls it. Each backend instruments its own forward without reimplementing the math:
  * pi05_jax       -> JAX profiler trace bucketed by `jax.named_scope` (CUDA graphs off).
  * pi05_realtimevla -> CUDA-event timing around the eager Triton stage fns (graphs off for the
                        split; the graph path is still used for E2E).
Both return the same dict shape (see profiling/jax_profiler.py:parse_trace) so metrics.build_row
treats them uniformly.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedSystem:
    """A loaded model ready to profile.

    infer_at(k) -> a zero-arg callable that runs one full inference with k denoise steps and
    returns device array(s) (NOT yet synced). `block(out)` forces completion (jax.block_until_ready).
    """

    name: str
    config_name: str
    prompt_len: int
    num_steps: int
    batch_size: int
    infer_at: Callable[[int], Callable[[], Any]]
    block: Callable[[Any], Any]
    meta: dict[str, Any] = field(default_factory=dict)
    phase_profiler: Callable[..., dict] | None = None
    """Backend-specific Vision/VLM/Action attribution. Called as
    `phase_profiler(warmup=int, iters=int, logdir=str) -> dict` (parse_trace shape). When None,
    the row degrades to E2E/Freq only."""

    def infer(self) -> Callable[[], Any]:
        """The configured-num_steps inference callable (what E2E / trace use)."""
        return self.infer_at(self.num_steps)


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
