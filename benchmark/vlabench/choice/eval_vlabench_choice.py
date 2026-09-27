#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataclasses
import json
import logging
import os
from typing import Optional, List, Tuple, Dict, Any

import tyro

from VLABench.evaluation.evaluator import Evaluator
from VLABench.robots import *   # noqa: F401,F403
from VLABench.tasks import *    # noqa: F401,F403

from model2vlabench_interface_choice import (
    MibotChoicePolicyArgs,
    MibotQwenChoiceVLABenchPolicy,
)


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10086
    task_id: str = "vlabench_choice"

    replan_steps: int = 5
    action_chunk_size: int = 10
    image_size: int = 480

    tasks: Optional[str] = None
    eval_track: Optional[str] = None
    episode_config_path: Optional[str] = None
    n_episode: int = 50

    metrics: str = "success_rate intention_score progress_score"
    save_dir: Optional[str] = None
    visulization: bool = True

    gripper_threshold: float = 0.2
    gripper_open_value: float = 0.04

    pos_offset_x: float = 0.0
    pos_offset_y: float = -0.4
    pos_offset_z: float = 0.78


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json format: {path}, type={type(data)}")
    return data


def _load_track_episode_config(eval_track: str) -> Dict[str, Any]:
    vlabench_root = os.getenv("VLABENCH_ROOT", None)
    if not vlabench_root:
        raise EnvironmentError("VLABENCH_ROOT is not set. Please export VLABENCH_ROOT=path/to/VLABench")

    candidates = [
        os.path.join(vlabench_root, "configs/evaluation/tracks", f"{eval_track}.json"),
        os.path.join(vlabench_root, "VLABench/configs/evaluation/tracks", f"{eval_track}.json"),
    ]
    for track_path in candidates:
        if os.path.exists(track_path):
            return _load_json(track_path)

    raise FileNotFoundError("Track config not found. Tried:\n" + "\n".join(candidates))


def _resolve_tasks_and_episode_configs(
    *,
    eval_track: Optional[str],
    episode_config_path: Optional[str],
    tasks_str: Optional[str],
    save_dir: str,
) -> Tuple[List[str], Optional[Dict[str, Any]], str]:
    episode_configs = None

    if episode_config_path is not None:
        episode_configs = _load_json(episode_config_path)
        all_tasks = list(episode_configs.keys())

        if tasks_str:
            want = tasks_str.split()
            tasks = [t for t in want if t in episode_configs]
            missing = [t for t in want if t not in episode_configs]
            if missing:
                logger.warning("[episode_config_path] tasks not in config: %s", missing)
            episode_configs = {t: episode_configs[t] for t in tasks}
        else:
            tasks = all_tasks

        effective_save_dir = save_dir
        os.makedirs(effective_save_dir, exist_ok=True)
        logger.info(
            "[episode_config_path] total_tasks=%d run_tasks=%d save_dir=%s config=%s",
            len(all_tasks), len(tasks), effective_save_dir, episode_config_path
        )
        return tasks, episode_configs, effective_save_dir

    if eval_track is not None:
        episode_configs = _load_track_episode_config(eval_track)
        all_tasks = list(episode_configs.keys())

        if tasks_str:
            want = tasks_str.split()
            tasks = [t for t in want if t in episode_configs]
            missing = [t for t in want if t not in episode_configs]
            if missing:
                logger.warning("[track] tasks not in track=%s: %s", eval_track, missing)
            episode_configs = {t: episode_configs[t] for t in tasks}
        else:
            tasks = all_tasks

        effective_save_dir = save_dir
        os.makedirs(effective_save_dir, exist_ok=True)
        logger.info(
            "[track] eval_track=%s total_tasks=%d run_tasks=%d save_dir=%s",
            eval_track, len(all_tasks), len(tasks), effective_save_dir
        )
        return tasks, episode_configs, effective_save_dir

    if not tasks_str:
        raise ValueError("Must provide one of --args.eval-track / --args.episode-config-path / --args.tasks")

    tasks = tasks_str.split()
    effective_save_dir = save_dir
    os.makedirs(effective_save_dir, exist_ok=True)

    logger.info("[tasks] num_tasks=%d save_dir=%s", len(tasks), effective_save_dir)
    return tasks, episode_configs, effective_save_dir


def main(args: Args) -> None:
    assert args.save_dir is not None, "save_dir must be provided"
    os.makedirs(args.save_dir, exist_ok=True)

    tasks, episode_configs, effective_save_dir = _resolve_tasks_and_episode_configs(
        eval_track=args.eval_track,
        episode_config_path=args.episode_config_path,
        tasks_str=args.tasks,
        save_dir=args.save_dir,
    )

    if len(tasks) == 0:
        logger.warning("No tasks resolved. Nothing to run.")
        return

    metrics = args.metrics.split()

    policy_args = MibotChoicePolicyArgs(
        host=args.host,
        port=args.port,
        task_id=args.task_id,
        action_chunk_size=args.action_chunk_size,
        replan_steps=args.replan_steps,
        image_size=args.image_size,
        pos_offset_x=args.pos_offset_x,
        pos_offset_y=args.pos_offset_y,
        pos_offset_z=args.pos_offset_z,
        gripper_threshold=args.gripper_threshold,
        gripper_open_value=args.gripper_open_value,
    )
    policy = MibotQwenChoiceVLABenchPolicy(policy_args)

    logger.info(
        "Start choice evaluation (fixed non-cot): tasks=%d n_episode=%d replan_steps=%d action_chunk_size=%d",
        len(tasks), args.n_episode, args.replan_steps, args.action_chunk_size
    )

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_configs,
        max_substeps=1,
        save_dir=effective_save_dir,
        visulization=args.visulization,
        metrics=metrics,
    )

    evaluator.evaluate(policy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)