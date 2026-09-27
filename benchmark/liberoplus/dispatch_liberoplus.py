#!/usr/bin/env python3

"""Dispatch official LIBERO-Plus episodes over consecutive MiBot ports."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import tyro

from catalog import (
    DEFAULT_LIBEROPLUS_ROOT,
    catalog_counts,
    load_catalog,
    select_tasks,
)
from official_runtime import DEFAULT_ASSETS_ROOT, DEFAULT_CONFIG_DIR
from summarize import summarize_results


LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    base_port: int = 10086
    num_workers: int = 1
    python_bin: str = sys.executable
    eval_py: str = str(HERE / "eval_liberoplus.py")

    suites: str = "libero_spatial libero_object libero_goal libero_10"
    perturbations: Optional[str] = None
    task_indices: Optional[str] = None
    task_names: Optional[str] = None
    difficulty_levels: Optional[str] = None
    max_tasks: int = 0
    episodes_per_task: int = 1

    action_chunk_size: int = 10
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 0
    resolution: int = 256
    seed: int = 7
    model_seed: int = 42
    socket_timeout: float = 600.0
    subprocess_timeout: float = 0.0
    render_gpu_device_id: int = -1

    save_dir: str = "results/liberoplus_dynamic"
    vis: bool = False
    env_only: bool = False
    resume: bool = True

    liberoplus_root: str = str(DEFAULT_LIBEROPLUS_ROOT)
    libero_config_dir: str = str(DEFAULT_CONFIG_DIR)
    assets_root: str = str(DEFAULT_ASSETS_ROOT)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _result_path(root: Path, job: Dict[str, Any]) -> Path:
    return (
        root
        / "episodes"
        / str(job["suite"])
        / f"task_{int(job['task_index']):04d}"
        / f"episode_{int(job['episode_index']):04d}.json"
    )


def _server_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _subprocess_env(args: Args) -> Dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    prefixes = [args.liberoplus_root, str(HERE)]
    env["PYTHONPATH"] = ":".join(prefixes + ([current_pythonpath] if current_pythonpath else []))
    env["LIBERO_CONFIG_PATH"] = args.libero_config_dir
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run_job(args: Args, root: Path, port: int, job: Dict[str, Any]) -> Dict[str, Any]:
    destination = _result_path(root, job)
    existing = _load_json(destination)
    if args.resume and existing is not None and existing.get("status") == "completed":
        LOGGER.info(
            "[port %d] SKIP suite=%s task=%d episode=%d",
            port,
            job["suite"],
            job["task_index"],
            job["episode_index"],
        )
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = destination.with_suffix(".log")
    video_path = destination.with_suffix(".mp4")
    command = [
        args.python_bin,
        "-u",
        args.eval_py,
        "--args.host",
        args.host,
        "--args.port",
        str(port),
        "--args.suite",
        str(job["suite"]),
        "--args.task-index",
        str(job["task_index"]),
        "--args.episode-index",
        str(job["episode_index"]),
        "--args.action-chunk-size",
        str(args.action_chunk_size),
        "--args.replan-steps",
        str(args.replan_steps),
        "--args.num-steps-wait",
        str(args.num_steps_wait),
        "--args.max-steps",
        str(args.max_steps),
        "--args.resolution",
        str(args.resolution),
        "--args.seed",
        str(args.seed),
        "--args.model-seed",
        str(args.model_seed),
        "--args.socket-timeout",
        str(args.socket_timeout),
        "--args.render-gpu-device-id",
        str(args.render_gpu_device_id),
        "--args.result-path",
        str(destination),
        "--args.video-path",
        str(video_path),
        "--args.liberoplus-root",
        args.liberoplus_root,
        "--args.libero-config-dir",
        args.libero_config_dir,
        "--args.assets-root",
        args.assets_root,
    ]
    if args.vis:
        command.append("--args.vis")
    if args.env_only:
        command.append("--args.env-only")

    LOGGER.info(
        "[port %d] START suite=%s task=%d episode=%d perturbation=%s",
        port,
        job["suite"],
        job["task_index"],
        job["episode_index"],
        job["perturbation"],
    )
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=_subprocess_env(args),
                timeout=args.subprocess_timeout or None,
            )
            returncode = process.returncode
            timeout_error = None
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            timeout_error = f"Subprocess timed out after {exc.timeout} seconds"

    record = _load_json(destination)
    if record is None:
        record = dict(job)
        record.update(
            {
                "status": "error",
                "success": False,
                "error": timeout_error
                or f"Evaluator exited with return code {returncode} without a result",
            }
        )
        _atomic_write_json(destination, record)
    elif returncode != 0 and record.get("status") == "completed":
        record["status"] = "error"
        record["error"] = f"Evaluator exited with return code {returncode}"
        _atomic_write_json(destination, record)

    record["dispatcher_returncode"] = returncode
    record["dispatcher_duration_sec"] = round(time.monotonic() - start, 3)
    LOGGER.info(
        "[port %d] DONE suite=%s task=%d episode=%d status=%s success=%s time=%.1fs",
        port,
        job["suite"],
        job["task_index"],
        job["episode_index"],
        record.get("status"),
        record.get("success"),
        record["dispatcher_duration_sec"],
    )
    return record


def main(args: Args) -> None:
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if args.episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if not Path(args.python_bin).is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python_bin}")
    if not Path(args.eval_py).is_file():
        raise FileNotFoundError(f"Evaluator not found: {args.eval_py}")

    catalog = load_catalog(args.liberoplus_root)
    selected = select_tasks(
        catalog,
        suites=args.suites,
        perturbations=args.perturbations,
        task_indices=args.task_indices,
        task_names=args.task_names,
        difficulty_levels=args.difficulty_levels,
        max_tasks=args.max_tasks,
    )
    if not selected:
        raise ValueError("No LIBERO-Plus tasks match the requested filters")

    jobs = [
        {**task.to_dict(), "episode_index": episode_index}
        for task in selected
        for episode_index in range(args.episodes_per_task)
    ]
    root = Path(args.save_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "filters": {
            "suites": args.suites,
            "perturbations": args.perturbations,
            "task_indices": args.task_indices,
            "task_names": args.task_names,
            "difficulty_levels": args.difficulty_levels,
            "max_tasks": args.max_tasks,
        },
        "episodes_per_task": args.episodes_per_task,
        "selected_tasks": len(selected),
        "expected_episodes": len(jobs),
        "counts": catalog_counts(selected),
        "tasks": [task.to_dict() for task in selected],
    }
    _atomic_write_json(root / "selection.json", manifest)

    worker_count = min(args.num_workers, len(jobs))
    if not args.env_only:
        unavailable = [
            args.base_port + worker_id
            for worker_id in range(worker_count)
            if not _server_ready(args.host, args.base_port + worker_id)
        ]
        if unavailable:
            raise ConnectionError(f"MiBot server ports are not ready: {unavailable}")

    LOGGER.info(
        "Selected %d tasks / %d episodes; workers=%d; counts=%s",
        len(selected),
        len(jobs),
        worker_count,
        manifest["counts"],
    )

    jobs_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    for job in jobs:
        jobs_queue.put(job)
    for _ in range(worker_count):
        jobs_queue.put(None)

    records: List[Dict[str, Any]] = []
    records_lock = threading.Lock()

    def worker(worker_id: int) -> None:
        port = args.base_port + worker_id
        while True:
            job = jobs_queue.get()
            try:
                if job is None:
                    return
                record = _run_job(args, root, port, job)
                with records_lock:
                    records.append(record)
            except Exception as exc:
                LOGGER.exception("Worker %d failed: %s", worker_id, exc)
                if job is not None:
                    destination = _result_path(root, job)
                    error_record = dict(job)
                    error_record.update(
                        {"status": "error", "success": False, "error": repr(exc)}
                    )
                    _atomic_write_json(destination, error_record)
                    with records_lock:
                        records.append(error_record)
            finally:
                jobs_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(worker_id,), daemon=False)
        for worker_id in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    jobs_queue.join()
    for thread in threads:
        thread.join()

    records.sort(
        key=lambda item: (
            str(item.get("suite")),
            int(item.get("task_index", -1)),
            int(item.get("episode_index", -1)),
        )
    )
    _atomic_write_json(root / "dispatch_results.json", records)
    summary = summarize_results(root, expected_episodes=len(jobs))
    LOGGER.info(
        "Summary: complete=%s completed=%d errors=%d success_rate=%s path=%s",
        summary["complete"],
        summary["overall"]["episodes_completed"],
        summary["overall"]["episodes_error"],
        summary["overall"]["success_rate"],
        root / "summary.json",
    )
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    tyro.cli(main)
