#!/usr/bin/env python
"""Serve a real pi0/pi05 TORCH checkpoint over the openpi websocket protocol (for closed-loop sim eval).

openpi's own scripts/serve_policy.py imports openpi.training.checkpoints -> data_loader -> lerobot, and
lerobot is intentionally not installed in the profiling env (we load models directly via safetensors).
lerobot is never used at *serving* time: create_trained_policy loads the torch model, builds the LIBERO
input/output transforms from the data config, and reads norm_stats.json from the checkpoint assets. So we
stub the lerobot import and reuse openpi's real create_trained_policy -> the served actions go through the
exact same Normalize / LiberoInputs / Unnormalize / LiberoOutputs pipeline as the upstream eval.

    source env.sh
    CUDA_VISIBLE_DEVICES=1 python scripts/serve_libero_torch.py \
        --checkpoint /scratch/ishirgarg/openpi_cache/pi05_libero_torch \
        --config-name pi05_libero --port 8000
"""

import logging
import sys
import types

import tyro


def _stub_lerobot() -> None:
    """Satisfy `import lerobot...` (pulled in transitively) without the heavy git-pinned package."""
    for name in ("lerobot", "lerobot.common", "lerobot.common.datasets",
                 "lerobot.common.datasets.lerobot_dataset"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def main(checkpoint: str, config_name: str = "pi05_libero", port: int = 8000,
         device: str | None = None) -> None:
    """device=None lets create_trained_policy auto-pick cuda (or cpu if no GPU); set CUDA_VISIBLE_DEVICES to pin."""
    _stub_lerobot()
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name), checkpoint, pytorch_device=device)
    logging.info("policy ready (%s); serving on 0.0.0.0:%d", config_name, port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
