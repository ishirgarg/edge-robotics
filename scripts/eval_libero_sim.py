#!/usr/bin/env python
"""Closed-loop LIBERO success-rate eval (simulator-in-the-loop) against a running policy server.

This is the second half of the "Both" eval: where eval_offline.py scores one-step action error vs a
dataset, this rolls the policy out IN the LIBERO simulator and reports task success rate. The rollout
loop follows openpi's examples/libero/main.py (same 180-degree image rotate, resize-with-pad, 8-D proprio
state = eef_pos + axisangle(eef_quat) + gripper_qpos, replan-every-K, done==success), and queries the
same websocket policy server; the additions here are caps on #tasks / #trials to bound wall time, a
machine-readable JSON result, and an optional video toggle. Run in the py3.8 libero-sim env, server up first:

    # terminal 1 (main env):  CUDA_VISIBLE_DEVICES=1 python scripts/serve_libero_torch.py --checkpoint ... --port 8000
    # terminal 2 (libero-sim env):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=/scratch/ishirgarg/libero_config \
    PYTHONPATH=openpi/third_party/libero python scripts/eval_libero_sim.py \
        --task-suite-name libero_spatial --num-tasks 10 --num-trials-per-task 10 --out out/sim/pi05_libero
"""

import collections
import json
import logging
import math
import os
import pathlib

import imageio
import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _quat2axisangle(quat):
    """Copied from robosuite (matches openpi examples/libero/main.py)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_env(task, resolution, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def main(host: str = "0.0.0.0", port: int = 8000, resize_size: int = 224, replan_steps: int = 5,
         task_suite_name: str = "libero_spatial", num_tasks: int = -1, num_trials_per_task: int = 10,
         num_steps_wait: int = 10, seed: int = 7, out: str = "out/sim/run", save_video: bool = True,
         video_dir: str = "/scratch/ishirgarg/tmp/libero_videos") -> None:
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    n_tasks = suite.n_tasks if num_tasks < 0 else min(num_tasks, suite.n_tasks)
    max_steps = _MAX_STEPS[task_suite_name]
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)
    logging.info("connected to server %s:%d; suite=%s tasks=%d trials/task=%d", host, port,
                 task_suite_name, n_tasks, num_trials_per_task)
    vdir = pathlib.Path(video_dir) / task_suite_name  # namespace by suite so re-runs don't collide
    if save_video:
        vdir.mkdir(parents=True, exist_ok=True)

    total_ep = total_succ = 0
    per_task = []
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = _get_env(task, LIBERO_ENV_RESOLUTION, seed)
        t_ep = t_succ = 0
        for ep in range(num_trials_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            plan = collections.deque()
            success = False
            replay = []
            t = 0
            while t < max_steps + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
                    if save_video:
                        replay.append(img)
                    if not plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"])),
                            "prompt": str(task.language),
                        }
                        chunk = client.infer(element)["actions"]
                        assert len(chunk) >= replan_steps, \
                            f"server returned {len(chunk)} actions < replan_steps={replan_steps}"
                        plan.extend(chunk[:replan_steps])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                    if done:
                        success = True  # in LIBERO done==task success; only the real-step path sets this
                        break
                    t += 1
                except Exception as e:  # match main.py: a sim/inference error ends the episode as a failure
                    logging.error("episode error (task %d ep %d): %s", task_id, ep, e)
                    break
            t_ep += 1
            total_ep += 1
            if success:
                t_succ += 1
                total_succ += 1
            if save_video and replay:
                tag = "success" if success else "failure"
                imageio.mimwrite(vdir / f"t{task_id}_ep{ep}_{tag}.mp4",
                                 [np.asarray(x) for x in replay], fps=10)
            logging.info("task %d ep %d -> %s | running %d/%d (%.1f%%)", task_id, ep,
                         "success" if success else "fail", total_succ, total_ep, 100 * total_succ / total_ep)
        env.close()
        per_task.append({"task_id": task_id, "language": task.language, "episodes": t_ep,
                         "successes": t_succ, "success_rate": t_succ / t_ep if t_ep else 0.0})
        logging.info("task %d done: %d/%d", task_id, t_succ, t_ep)

    res = {
        "task_suite_name": task_suite_name, "num_tasks": n_tasks, "num_trials_per_task": num_trials_per_task,
        "replan_steps": replan_steps, "resize_size": resize_size, "seed": seed,
        "total_episodes": total_ep, "total_successes": total_succ,
        "success_rate": total_succ / total_ep if total_ep else 0.0, "per_task": per_task,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "sim_eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n{task_suite_name}: success {total_succ}/{total_ep} = {100 * res['success_rate']:.1f}%")
    print(f"wrote {out}/sim_eval.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
