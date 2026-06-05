"""Minimal interface a profilable system must provide.

Light on abstraction by design: a system just knows how to (1) load itself and (2) hand
back a callable that runs ONE unmodified inference at a given number of denoise steps,
plus a `block` function the timers use to force device completion. Everything phase-related
is derived by the profiler from these callables (see edge_robotics/profiling/*), so systems
never need to expose model internals.
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
    # Optional: a zero-arg callable that runs ONLY the vision encoder (SigLIP) over all camera
    # images, for the Vision-vs-VLM split (the fused profiler trace can't separate them). None
    # if the system doesn't expose it. See edge_robotics/systems/pi05_jax.py.
    vision_infer: Callable[[], Any] | None = None

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
