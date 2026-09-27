#!/usr/bin/env python3

"""Summarize LIBERO-Plus episode JSON files by all official dimensions."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def _load_records(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    records: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, str]] = []
    for path in sorted(root.glob("episodes/*/task_*/episode_*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if not isinstance(record, dict):
                raise ValueError(f"expected object, got {type(record).__name__}")
            record["result_path"] = str(path)
            records.append(record)
        except Exception as exc:
            unreadable.append({"path": str(path), "error": repr(exc)})
    return records, unreadable


def _aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = list(records)
    completed = [item for item in values if item.get("status") == "completed"]
    errors = [item for item in values if item.get("status") != "completed"]
    successes = sum(bool(item.get("success", False)) for item in completed)
    steps = [float(item.get("executed_steps", 0) or 0) for item in completed]
    requests = [float(item.get("model_requests", 0) or 0) for item in completed]
    return {
        "episodes_found": len(values),
        "episodes_completed": len(completed),
        "episodes_error": len(errors),
        "successes": successes,
        "success_rate": successes / len(completed) if completed else None,
        "mean_executed_steps": sum(steps) / len(steps) if steps else None,
        "mean_model_requests": sum(requests) / len(requests) if requests else None,
    }


def _group(
    records: Iterable[Dict[str, Any]], key: str
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(key, "unknown"))].append(record)
    return {name: _aggregate(grouped[name]) for name in sorted(grouped)}


def summarize_results(
    root_dir: Path | str,
    *,
    expected_episodes: int | None = None,
    output_path: Path | str | None = None,
) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    records, unreadable = _load_records(root)
    technical_errors = [
        {
            "suite": item.get("suite"),
            "task_index": item.get("task_index"),
            "episode_index": item.get("episode_index"),
            "error": item.get("error", "unknown error"),
            "result_path": item.get("result_path"),
        }
        for item in records
        if item.get("status") != "completed"
    ]
    overall = _aggregate(records)
    expected = expected_episodes
    if expected is None:
        manifest = root / "selection.json"
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8") as handle:
                expected = int(json.load(handle).get("expected_episodes", 0))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_dir": str(root),
        "expected_episodes": expected,
        "complete": bool(
            expected is not None
            and overall["episodes_completed"] == expected
            and overall["episodes_error"] == 0
            and not unreadable
        ),
        "overall": overall,
        "by_suite": _group(records, "suite"),
        "by_perturbation": _group(records, "perturbation"),
        "by_difficulty": _group(records, "difficulty_level"),
        "technical_errors": technical_errors,
        "unreadable_results": unreadable,
    }
    destination = Path(output_path).resolve() if output_path else root / "summary.json"
    _atomic_write_json(destination, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    summary = summarize_results(
        args.root_dir,
        expected_episodes=args.expected_episodes,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

