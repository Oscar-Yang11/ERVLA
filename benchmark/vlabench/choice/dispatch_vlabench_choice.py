#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataclasses
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import tyro

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    base_port: int = 10086
    num_workers: int = 4
    task_id: str = "vlabench_choice"

    eval_py: str = str(HERE / "eval_vlabench_choice.py")
    python_bin: str = sys.executable

    replan_steps: int = 5
    action_chunk_size: int = 10
    image_size: int = 480

    eval_track: Optional[str] = None
    tasks: Optional[str] = None
    n_episode: int = 1

    save_dir: str = "results/vlabench_choice_dynamic"
    metrics: str = "success_rate intention_score progress_score"
    visulization: bool = False

    gripper_threshold: float = 0.2
    gripper_open_value: float = 0.04

    pos_offset_x: float = 0.0
    pos_offset_y: float = -0.4
    pos_offset_z: float = 0.78


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
            with open(track_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid track json format: {track_path}, type={type(data)}")
            return data

    raise FileNotFoundError("Track config not found. Tried:\n" + "\n".join(candidates))


def _select_tasks(track_cfg: Dict[str, Any], tasks_str: Optional[str]) -> List[str]:
    all_tasks = list(track_cfg.keys())
    if not tasks_str:
        return all_tasks
    want = tasks_str.split()
    tasks = [t for t in want if t in track_cfg]
    missing = [t for t in want if t not in track_cfg]
    if missing:
        logger.warning("Tasks not found in track config: %s", missing)
    return tasks


def _build_jobs(track_cfg: Dict[str, Any], tasks: List[str], n_episode: int) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for task in tasks:
        eps = track_cfg[task]
        if not isinstance(eps, list):
            raise ValueError(f"Track config for task={task} must be a list, got {type(eps)}")
        num = min(n_episode, len(eps))
        for ep_idx in range(num):
            jobs.append({
                "task": task,
                "episode_index": ep_idx,
                "episode_cfg": eps[ep_idx],
            })
    return jobs


def _write_single_episode_config(tmp_dir: Path, task: str, ep_idx: int, ep_cfg: Any) -> str:
    out_path = tmp_dir / f"{task}__ep{ep_idx:04d}.json"
    with open(out_path, "w") as f:
        json.dump({task: [ep_cfg]}, f)
    return str(out_path)


def _safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load json %s: %r", path, e)
        return None


def _atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp_path, path)


def _rename_video_with_global_episode(src_name: str, ep_idx: int) -> str:
    m = re.match(r"^(\d+)_(.+)$", src_name)
    if m:
        return f"{ep_idx}_{m.group(2)}"
    return f"{ep_idx}_{src_name}"


def _parse_video_metric_from_name(video_name: str) -> Optional[Tuple[int, bool, float]]:
    """
    Parse names like:
      0_success_True_progress_1.00.mp4
      10_success_False_progress_0.00.mp4
    """
    m = re.match(
        r"^(?P<ep>\d+)_success_(?P<success>True|False)_progress_(?P<progress>[0-9.]+)\.mp4$",
        video_name,
    )
    if not m:
        return None

    ep = int(m.group("ep"))
    success = m.group("success") == "True"
    progress = float(m.group("progress"))
    return ep, success, progress


def _coerce_float_score(x: Any, default: float = 0.0) -> float:
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return default
    try:
        s = str(x).strip().lower()
        if s in ("true", "false"):
            return float(int(s == "true"))
        return float(s)
    except Exception:
        return default


def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _get_first_existing(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _normalize_detail_records(detail_info_obj: Any, task: str, ep_idx: int) -> List[Dict[str, Any]]:
    if not isinstance(detail_info_obj, list):
        return []

    records: List[Dict[str, Any]] = []
    for item in detail_info_obj:
        if not isinstance(item, dict):
            continue

        rec = dict(item)
        rec["episode_id"] = int(ep_idx)
        rec["task"] = str(rec.get("task", task))

        success_val = _get_first_existing(
            rec,
            ["success", "success_rate", "is_success", "task_success"],
            False,
        )
        progress_val = _get_first_existing(
            rec,
            ["progress_score", "progress", "task_progress"],
            0.0,
        )
        intention_val = _get_first_existing(
            rec,
            ["intention_score", "intention", "task_intention"],
            0.0,
        )

        rec["success"] = _coerce_bool(success_val)
        rec["consumed_step"] = int(rec.get("consumed_step", 0) or 0)
        rec["intention_score"] = _coerce_float_score(intention_val)
        rec["progress_score"] = _coerce_float_score(progress_val)

        records.append(rec)

    records.sort(key=lambda x: int(x.get("episode_id", -1)))
    return records


def _merge_video_metrics_into_records(
    records: List[Dict[str, Any]],
    task: str,
    ep_idx: int,
    video_metrics: Dict[int, Tuple[bool, float]],
) -> List[Dict[str, Any]]:
    """
    Video filename is usually the most reliable source for success/progress,
    because it is written after the env evaluation. Use it to fix missing or
    incorrect detail_info fields.
    """
    by_ep: Dict[int, Dict[str, Any]] = {}

    for rec in records:
        try:
            eid = int(rec.get("episode_id", ep_idx))
        except Exception:
            eid = ep_idx
        rec["episode_id"] = eid
        by_ep[eid] = rec

    for eid, (success, progress) in video_metrics.items():
        rec = by_ep.get(eid)
        if rec is None:
            rec = {
                "episode_id": eid,
                "task": task,
                "consumed_step": 0,
                "intention_score": 0.0,
            }
            by_ep[eid] = rec

        rec["task"] = str(rec.get("task", task))
        rec["success"] = bool(success)
        rec["progress_score"] = float(progress)

        if "intention_score" not in rec:
            rec["intention_score"] = 0.0
        if "consumed_step" not in rec:
            rec["consumed_step"] = 0

    merged = list(by_ep.values())
    merged.sort(key=lambda x: int(x.get("episode_id", -1)))
    return merged


def _append_flat_detail_records(task_dir: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    detail_path = task_dir / "detail_info.json"
    existing = _safe_load_json(detail_path)
    if not isinstance(existing, list):
        existing = []

    existing = [x for x in existing if isinstance(x, dict)]

    keep_ids = set()
    for r in records:
        try:
            keep_ids.add(int(r["episode_id"]))
        except Exception:
            pass

    # Replace same episode_id records instead of duplicating.
    existing = [
        x for x in existing
        if int(x.get("episode_id", -1)) not in keep_ids
    ]

    existing.extend(records)
    existing.sort(key=lambda x: int(x.get("episode_id", -1)))

    _atomic_write_json(detail_path, existing, indent=2)


def _compute_task_metrics_from_detail(detail_records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not detail_records:
        return {
            "success_rate": 0.0,
            "intention_score": 0.0,
            "progress_score": 0.0,
        }

    n = len(detail_records)
    success_rate = sum(
        1.0 if _coerce_bool(x.get("success", False)) else 0.0
        for x in detail_records
    ) / n
    intention_score = sum(
        _coerce_float_score(x.get("intention_score", 0.0))
        for x in detail_records
    ) / n
    progress_score = sum(
        _coerce_float_score(x.get("progress_score", 0.0))
        for x in detail_records
    ) / n

    return {
        "success_rate": success_rate,
        "intention_score": intention_score,
        "progress_score": progress_score,
    }


def _write_track_metrics(save_root: Path, tasks: List[str]) -> Dict[str, Dict[str, float]]:
    track_metrics: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        detail_path = save_root / task / "detail_info.json"
        detail_records = _safe_load_json(detail_path)
        if not isinstance(detail_records, list):
            detail_records = []
        track_metrics[task] = _compute_task_metrics_from_detail(detail_records)

    _atomic_write_json(save_root / "metrics.json", track_metrics, indent=2)
    return track_metrics


def _flatten_job_outputs(
    tmp_job_root: Path,
    task: str,
    ep_idx: int,
    save_root: Path,
    returncode: int,
    job_error: Optional[str] = None,
    detail_lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    task_dir = save_root / task
    videos_dir = task_dir / "videos"
    logs_dir = task_dir / "worker_logs"
    task_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    detail_info_obj = _safe_load_json(tmp_job_root / task / "detail_info.json")
    normalized_detail_records = _normalize_detail_records(detail_info_obj, task=task, ep_idx=ep_idx)

    src_videos_dir = tmp_job_root / task / "videos"
    moved_videos: List[str] = []
    video_metrics: Dict[int, Tuple[bool, float]] = {}

    if src_videos_dir.exists():
        for video_file in sorted(src_videos_dir.iterdir()):
            if not video_file.is_file():
                continue

            dst_name = _rename_video_with_global_episode(video_file.name, ep_idx)
            dst_path = videos_dir / dst_name

            if dst_path.exists():
                dst_path.unlink()

            shutil.move(str(video_file), str(dst_path))
            moved_videos.append(str(dst_path))

            parsed = _parse_video_metric_from_name(dst_name)
            if parsed is not None:
                parsed_ep, parsed_success, parsed_progress = parsed
                video_metrics[parsed_ep] = (parsed_success, parsed_progress)

    # Use video filename as fallback / correction for success and progress.
    normalized_detail_records = _merge_video_metrics_into_records(
        records=normalized_detail_records,
        task=task,
        ep_idx=ep_idx,
        video_metrics=video_metrics,
    )

    worker_log_src = tmp_job_root / "worker.log"
    worker_log_dst = logs_dir / f"ep_{ep_idx:04d}.log"
    if worker_log_src.exists():
        if worker_log_dst.exists():
            worker_log_dst.unlink()
        shutil.move(str(worker_log_src), str(worker_log_dst))

    if returncode != 0:
        err_path = logs_dir / f"ep_{ep_idx:04d}.error.json"
        with open(err_path, "w") as f:
            json.dump(
                {
                    "task": task,
                    "episode_index": ep_idx,
                    "returncode": returncode,
                    "error": job_error,
                },
                f,
                indent=2,
            )

    # Important: protect read-modify-write of detail_info.json.
    if detail_lock is None:
        _append_flat_detail_records(task_dir, normalized_detail_records)
    else:
        with detail_lock:
            _append_flat_detail_records(task_dir, normalized_detail_records)

    shutil.rmtree(tmp_job_root, ignore_errors=True)

    return {
        "task_dir": str(task_dir),
        "videos_dir": str(videos_dir),
        "worker_log": str(worker_log_dst),
        "moved_videos": moved_videos,
        "num_detail_records": len(normalized_detail_records),
    }


def _run_one_job(
    args: Args,
    port: int,
    job: Dict[str, Any],
    tmp_dir: Path,
    save_root: Path,
    detail_lock: threading.Lock,
):
    task = job["task"]
    ep_idx = int(job["episode_index"])
    cfg_path = _write_single_episode_config(tmp_dir, task, ep_idx, job["episode_cfg"])

    tmp_job_root = save_root / "_tmp_runs" / task / f"ep_{ep_idx:04d}"
    tmp_job_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin, "-u", args.eval_py,
        "--args.host", args.host,
        "--args.port", str(port),
        "--args.task-id", args.task_id,
        "--args.n-episode", "1",
        "--args.save-dir", str(tmp_job_root),
        "--args.replan-steps", str(args.replan_steps),
        "--args.action-chunk-size", str(args.action_chunk_size),
        "--args.image-size", str(args.image_size),
        "--args.metrics", args.metrics,
        "--args.episode-config-path", cfg_path,
        "--args.tasks", task,
        "--args.gripper-threshold", str(args.gripper_threshold),
        "--args.gripper-open-value", str(args.gripper_open_value),
        "--args.pos-offset-x", str(args.pos_offset_x),
        "--args.pos-offset-y", str(args.pos_offset_y),
        "--args.pos-offset-z", str(args.pos_offset_z),
    ]

    if args.visulization:
        cmd.append("--args.visulization")
    else:
        cmd.append("--args.no-visulization")

    job_log = tmp_job_root / "worker.log"
    start = time.time()

    logger.info("[port %d] START task=%s ep=%d", port, task, ep_idx)

    with open(job_log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    dur = time.time() - start

    logger.info(
        "[port %d] DONE  task=%s ep=%d rc=%d time=%.1fs",
        port, task, ep_idx, proc.returncode, dur
    )

    flatten_info = _flatten_job_outputs(
        tmp_job_root=tmp_job_root,
        task=task,
        ep_idx=ep_idx,
        save_root=save_root,
        returncode=proc.returncode,
        job_error=None if proc.returncode == 0 else f"eval_vlabench_choice.py exited with rc={proc.returncode}",
        detail_lock=detail_lock,
    )

    return {
        "task": task,
        "episode_index": ep_idx,
        "port": port,
        "returncode": proc.returncode,
        "duration_sec": dur,
        "task_dir": flatten_info["task_dir"],
        "videos_dir": flatten_info["videos_dir"],
        "worker_log": flatten_info["worker_log"],
        "episode_config_path": cfg_path,
        "num_detail_records": flatten_info["num_detail_records"],
    }


def worker_loop(
    worker_id: int,
    port: int,
    args: Args,
    jobs_q: "queue.Queue[Optional[Dict[str, Any]]]",
    tmp_dir: Path,
    save_root: Path,
    results: List[Dict[str, Any]],
    results_lock: threading.Lock,
    detail_lock: threading.Lock,
):
    while True:
        job = jobs_q.get()
        if job is None:
            jobs_q.task_done()
            return

        try:
            result = _run_one_job(
                args=args,
                port=port,
                job=job,
                tmp_dir=tmp_dir,
                save_root=save_root,
                detail_lock=detail_lock,
            )
            with results_lock:
                results.append(result)

        except Exception as e:
            task = str(job.get("task"))
            ep_idx = int(job.get("episode_index"))
            task_dir = save_root / task
            logs_dir = task_dir / "worker_logs"
            task_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            with open(logs_dir / f"ep_{ep_idx:04d}.error.json", "w") as f:
                json.dump(
                    {
                        "task": task,
                        "episode_index": ep_idx,
                        "port": port,
                        "returncode": -1,
                        "error": repr(e),
                    },
                    f,
                    indent=2,
                )

            with results_lock:
                results.append({
                    "task": task,
                    "episode_index": ep_idx,
                    "port": port,
                    "returncode": -1,
                    "duration_sec": None,
                    "task_dir": str(task_dir),
                    "videos_dir": str(task_dir / "videos"),
                    "worker_log": None,
                    "episode_config_path": None,
                    "error": repr(e),
                    "num_detail_records": 0,
                })

            logger.exception("[port %d] FAILED task=%s ep=%s", port, job.get("task"), job.get("episode_index"))

        finally:
            jobs_q.task_done()


def main(args: Args):
    if args.eval_track is None:
        raise ValueError("For dynamic dispatch, please provide --args.eval-track")

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    track_cfg = _load_track_episode_config(args.eval_track)
    tasks = _select_tasks(track_cfg, args.tasks)
    jobs = _build_jobs(track_cfg, tasks, args.n_episode)

    logger.info(
        "Resolved %d tasks, %d jobs, %d workers (choice/no-cot)",
        len(tasks), len(jobs), args.num_workers
    )

    tmp_cfg_dir = save_root / "_tmp_episode_cfgs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    (save_root / "_tmp_runs").mkdir(parents=True, exist_ok=True)

    jobs_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    for job in jobs:
        jobs_q.put(job)
    for _ in range(args.num_workers):
        jobs_q.put(None)

    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()

    # Protect task/detail_info.json read-modify-write.
    detail_lock = threading.Lock()

    threads = []
    for i in range(args.num_workers):
        port = args.base_port + i
        t = threading.Thread(
            target=worker_loop,
            args=(
                i,
                port,
                args,
                jobs_q,
                tmp_cfg_dir,
                save_root,
                results,
                results_lock,
                detail_lock,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    jobs_q.join()
    for t in threads:
        t.join()

    results = sorted(
        results,
        key=lambda x: (str(x.get("task")), int(x.get("episode_index", -1)))
    )

    track_metrics = _write_track_metrics(save_root, tasks)

    _atomic_write_json(
        save_root / "_dispatch_manifest.json",
        {
            "eval_track": args.eval_track,
            "tasks": tasks,
            "num_workers": args.num_workers,
            "base_port": args.base_port,
            "num_jobs": len(jobs),
            "mode": "choice_no_cot",
            "track_metrics": track_metrics,
            "results": results,
        },
        indent=2,
    )

    shutil.rmtree(save_root / "_tmp_runs", ignore_errors=True)
    shutil.rmtree(save_root / "_tmp_episode_cfgs", ignore_errors=True)

    ok = sum(1 for r in results if r.get("returncode") == 0)
    fail = len(results) - ok

    logger.info("All jobs finished. success=%d fail=%d mode=choice_no_cot", ok, fail)
    logger.info("Track metrics saved to %s", save_root / "metrics.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
