#!/usr/bin/env python3

"""LIBERO-Plus task catalog and filtering helpers."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


DEFAULT_LIBEROPLUS_ROOT = Path(os.environ.get(
    "LIBEROPLUS_ROOT", Path(__file__).resolve().parents[2] / "third_party/LIBERO-plus"
))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PERTURBATIONS = (
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_SUITE_LOOKUP = {_normalize(value): value for value in SUITES}
_PERTURBATION_LOOKUP = {_normalize(value): value for value in PERTURBATIONS}
for alias, canonical in {
    "layout": "Objects Layout",
    "object": "Objects Layout",
    "camera": "Camera Viewpoints",
    "view": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "initstate": "Robot Initial States",
    "language": "Language Instructions",
    "instruction": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "texture": "Background Textures",
    "noise": "Sensor Noise",
    "sensor": "Sensor Noise",
}.items():
    _PERTURBATION_LOOKUP[_normalize(alias)] = canonical


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_index: int
    official_id: int
    name: str
    perturbation: str
    difficulty_level: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def classification_path(liberoplus_root: Path | str) -> Path:
    return (
        Path(liberoplus_root)
        / "libero/libero/benchmark/task_classification.json"
    )


def load_catalog(
    liberoplus_root: Path | str = DEFAULT_LIBEROPLUS_ROOT,
) -> List[TaskSpec]:
    path = classification_path(liberoplus_root)
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus classification file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(raw).__name__}")

    catalog: List[TaskSpec] = []
    for suite in SUITES:
        entries = raw.get(suite)
        if not isinstance(entries, list):
            raise ValueError(f"Missing task list for {suite!r} in {path}")

        seen_names: Set[str] = set()
        for task_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Invalid catalog entry at {suite}[{task_index}]: {entry!r}"
                )
            official_id = int(entry["id"])
            if official_id != task_index + 1:
                raise ValueError(
                    f"Catalog ID mismatch at {suite}[{task_index}]: "
                    f"expected {task_index + 1}, got {official_id}"
                )
            name = str(entry["name"])
            if name in seen_names:
                raise ValueError(f"Duplicate task name in {suite}: {name}")
            seen_names.add(name)

            perturbation = str(entry["category"])
            if perturbation not in PERTURBATIONS:
                raise ValueError(
                    f"Unknown perturbation category {perturbation!r} in {path}"
                )
            raw_difficulty = entry.get("difficulty_level")
            difficulty = (
                None if raw_difficulty is None else int(raw_difficulty)
            )
            if difficulty is not None and difficulty <= 0:
                raise ValueError(
                    f"Invalid difficulty level for {suite}[{task_index}]: {difficulty}"
                )

            catalog.append(
                TaskSpec(
                    suite=suite,
                    task_index=task_index,
                    official_id=official_id,
                    name=name,
                    perturbation=perturbation,
                    difficulty_level=difficulty,
                )
            )
    return catalog


def _split_tokens(value: Optional[str]) -> List[str]:
    if value is None or not value.strip():
        return []
    return [part for part in re.split(r"[\s,;]+", value.strip()) if part]


def resolve_suites(value: Optional[str]) -> List[str]:
    if value is None or not value.strip():
        return list(SUITES)
    resolved: List[str] = []
    for token in _split_tokens(value):
        canonical = _SUITE_LOOKUP.get(_normalize(token))
        if canonical is None:
            raise ValueError(f"Unknown suite {token!r}. Available: {', '.join(SUITES)}")
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def resolve_perturbations(value: Optional[str]) -> Optional[Set[str]]:
    if value is None or not value.strip():
        return None

    whole = _PERTURBATION_LOOKUP.get(_normalize(value))
    if whole is not None:
        return {whole}

    resolved: Set[str] = set()
    for token in _split_tokens(value):
        canonical = _PERTURBATION_LOOKUP.get(_normalize(token))
        if canonical is None:
            raise ValueError(
                f"Unknown perturbation {token!r}. Available: {', '.join(PERTURBATIONS)}"
            )
        resolved.add(canonical)
    return resolved


def _parse_int_set(value: Optional[str], label: str) -> Optional[Set[int]]:
    tokens = _split_tokens(value)
    if not tokens:
        return None
    try:
        return {int(token) for token in tokens}
    except ValueError as exc:
        raise ValueError(f"{label} must contain integers: {value!r}") from exc


def select_tasks(
    catalog: Sequence[TaskSpec],
    *,
    suites: Optional[str] = None,
    perturbations: Optional[str] = None,
    task_indices: Optional[str] = None,
    task_names: Optional[str] = None,
    difficulty_levels: Optional[str] = None,
    max_tasks: int = 0,
) -> List[TaskSpec]:
    suite_set = set(resolve_suites(suites))
    perturbation_set = resolve_perturbations(perturbations)
    task_index_set = _parse_int_set(task_indices, "task_indices")
    difficulty_set = _parse_int_set(difficulty_levels, "difficulty_levels")
    task_name_set = set(_split_tokens(task_names)) or None

    selected = [
        task
        for task in catalog
        if task.suite in suite_set
        and (perturbation_set is None or task.perturbation in perturbation_set)
        and (task_index_set is None or task.task_index in task_index_set)
        and (task_name_set is None or task.name in task_name_set)
        and (difficulty_set is None or task.difficulty_level in difficulty_set)
    ]

    if task_name_set is not None:
        found = {task.name for task in selected}
        missing = sorted(task_name_set - found)
        if missing:
            raise ValueError(f"Task names were not found under the active filters: {missing}")

    if max_tasks < 0:
        raise ValueError(f"max_tasks must be >= 0, got {max_tasks}")
    if max_tasks:
        selected = selected[:max_tasks]
    return selected


def catalog_counts(tasks: Iterable[TaskSpec]) -> Dict[str, Dict[str, int]]:
    task_list = list(tasks)
    return {
        "suite": dict(sorted(Counter(task.suite for task in task_list).items())),
        "perturbation": dict(
            sorted(Counter(task.perturbation for task in task_list).items())
        ),
        "difficulty": dict(
            sorted(
                Counter(
                    "unclassified"
                    if task.difficulty_level is None
                    else str(task.difficulty_level)
                    for task in task_list
                ).items(),
                key=lambda item: (
                    item[0] == "unclassified",
                    int(item[0]) if item[0] != "unclassified" else 0,
                ),
            )
        ),
    }
