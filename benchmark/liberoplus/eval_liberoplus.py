#!/usr/bin/env python3

"""Evaluate one official LIBERO-Plus task episode against a MiBot server."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import logging
import os
import pathlib
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import imageio.v2 as imageio
import numpy as np
import tyro

from catalog import DEFAULT_LIBEROPLUS_ROOT, SUITES, load_catalog
from model2liberoplus_interface import LiberoPlusPolicyArgs, MiBotLiberoPlusPolicy
from official_runtime import (
    DEFAULT_ASSETS_ROOT,
    DEFAULT_CONFIG_DIR,
    load_official_runtime,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10086

    suite: str = "libero_spatial"
    task_index: int = 0
    episode_index: int = 0

    action_chunk_size: int = 10
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 0
    resolution: int = 256
    seed: int = 7
    model_seed: int = 42
    socket_timeout: float = 600.0
    render_gpu_device_id: int = -1

    result_path: str = "outputs/liberoplus_episode.json"
    video_path: Optional[str] = None
    vis: bool = False
    env_only: bool = False

    liberoplus_root: str = str(DEFAULT_LIBEROPLUS_ROOT)
    libero_config_dir: str = str(DEFAULT_CONFIG_DIR)
    assets_root: str = str(DEFAULT_ASSETS_ROOT)


def _atomic_write_json(path: pathlib.Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def _catalog_task(args: Args):
    if args.suite not in SUITES:
        raise ValueError(f"Unknown suite {args.suite!r}. Available: {', '.join(SUITES)}")
    for task in load_catalog(args.liberoplus_root):
        if task.suite == args.suite and task.task_index == args.task_index:
            return task
    raise IndexError(f"Task index {args.task_index} is out of range for {args.suite}")


def _frame_for_video(observation: Dict[str, Any]) -> np.ndarray:
    frame = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    if np.issubdtype(frame.dtype, np.floating):
        if frame.size and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main(args: Args) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Official LIBERO-Plus init states use the pre-PyTorch-2.6 pickle format.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    result_path = pathlib.Path(args.result_path).resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    result: Dict[str, Any] = {
        "status": "starting",
        "suite": args.suite,
        "task_index": args.task_index,
        "episode_index": args.episode_index,
        "host": args.host,
        "port": args.port,
        "started_at": started_at,
        "env_only": args.env_only,
    }

    environment = None
    policy = None
    replay_frames = []
    try:
        if args.task_index < 0 or args.episode_index < 0:
            raise ValueError("task_index and episode_index must be non-negative")
        if args.num_steps_wait < 0 or args.max_steps < 0:
            raise ValueError("num_steps_wait and max_steps must be non-negative")
        if args.resolution <= 0:
            raise ValueError("resolution must be positive")

        spec = _catalog_task(args)
        result.update(spec.to_dict())

        benchmark, environment_class, get_libero_path = load_official_runtime(
            liberoplus_root=args.liberoplus_root,
            config_dir=args.libero_config_dir,
            assets_root=args.assets_root,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[args.suite]()
        official_task = suite.get_task(args.task_index)
        if official_task.name != spec.name:
            raise RuntimeError(
                "Classification and official benchmark order differ at "
                f"{args.suite}[{args.task_index}]: "
                f"{spec.name!r} vs {official_task.name!r}"
            )

        initial_states = suite.get_task_init_states(args.task_index)
        if args.episode_index >= len(initial_states):
            raise IndexError(
                f"Episode index {args.episode_index} is out of range; "
                f"task has {len(initial_states)} initial state(s)"
            )
        bddl_path = pathlib.Path(get_libero_path("bddl_files")) / (
            official_task.problem_folder
        ) / official_task.bddl_file
        result.update(
            {
                "instruction": str(official_task.language),
                "bddl_path": str(bddl_path),
                "initial_state_count": len(initial_states),
                "official_benchmark_path": str(pathlib.Path(benchmark.__file__).resolve()),
                "assets_root": str(pathlib.Path(args.assets_root).resolve()),
            }
        )

        environment = environment_class(
            bddl_file_name=str(bddl_path),
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        environment.seed(args.seed)
        environment.reset()
        observation = environment.set_init_state(initial_states[args.episode_index])

        done = bool(environment.check_success())
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = environment.step(LIBERO_DUMMY_ACTION)
        done = bool(done or environment.check_success())

        max_steps = args.max_steps or DEFAULT_MAX_STEPS[args.suite]
        executed_steps = 0
        action_min = None
        action_max = None

        if not args.env_only and not done:
            policy = MiBotLiberoPlusPolicy(
                LiberoPlusPolicyArgs(
                    host=args.host,
                    port=args.port,
                    suite=args.suite,
                    action_chunk_size=args.action_chunk_size,
                    replan_steps=args.replan_steps,
                    seed=args.model_seed,
                    socket_timeout=args.socket_timeout,
                )
            )
            policy.reset()

            for _ in range(max_steps):
                if args.vis:
                    replay_frames.append(_frame_for_video(observation))
                action = policy.predict(observation, str(official_task.language))
                current_min = float(np.min(action))
                current_max = float(np.max(action))
                action_min = current_min if action_min is None else min(action_min, current_min)
                action_max = current_max if action_max is None else max(action_max, current_max)
                observation, _, done, _ = environment.step(action.tolist())
                executed_steps += 1
                if done:
                    break

        if args.vis and replay_frames:
            video_path = (
                pathlib.Path(args.video_path).resolve()
                if args.video_path
                else result_path.with_suffix(".mp4")
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(video_path, replay_frames, fps=10)
            result["video_path"] = str(video_path)

        result.update(
            {
                "status": "completed",
                "success": bool(done),
                "max_steps": max_steps,
                "num_steps_wait": args.num_steps_wait,
                "executed_steps": executed_steps,
                "model_requests": 0 if policy is None else policy.request_count,
                "action_chunk_shape": None
                if policy is None
                else policy.last_chunk_shape,
                "first_action": None if policy is None else policy.first_action,
                "executed_action_min": action_min,
                "executed_action_max": action_max,
            }
        )
        LOGGER.info(
            "Episode complete: suite=%s task=%d success=%s steps=%d requests=%d",
            args.suite,
            args.task_index,
            result["success"],
            executed_steps,
            result["model_requests"],
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if policy is not None:
            policy.close()
        if environment is not None:
            try:
                environment.close()
            except Exception:
                LOGGER.exception("Failed to close LIBERO-Plus environment")
        result["duration_sec"] = round(time.monotonic() - start_time, 3)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(result_path, result)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    tyro.cli(main)
