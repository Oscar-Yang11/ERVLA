#!/usr/bin/env python3
"""Build MiBot ECOT JSON from a JSON+video episode tree.

When an existing language ECOT root is supplied, this utility keeps its
language fields. Without one, it starts an empty scaffold and adds the
deterministic parts used by MiBot:

* LAP-style movement text from the recorded EEF trajectory;
* optional future EEF waypoints projected into camera pixels.

The dataset presets cover LIBERO, VLABench, DROID, Bridge, Fractal, and
MolmoAct. Bridge and Fractal use the named EEF position/rotation fields;
MolmoAct uses its 6D direct EEF pose and action gripper channel.

The output layout is the layout consumed by ``JsonDataset``::

    output_root/chunk-0000/episode_0000000_ecot.json

No MDS conversion is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from scipy.spatial.transform import Rotation


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def episode_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("episode_*.json")
        if path.is_file() and not path.name.endswith("_ecot.json")
    )


def source_index(root: Path) -> dict[tuple[str, str], Path]:
    if not root:
        return {}
    files = [root] if root.is_file() else root.rglob("episode_*_ecot.json")
    index: dict[tuple[str, str], Path] = {}
    for path in files:
        if not path.is_file():
            continue
        key = (path.parent.name, path.name.removesuffix("_ecot.json"))
        if key in index:
            raise ValueError(f"duplicate COT episode {key}; select one dataset/suite at a time")
        index[key] = path
    return index


def find_language_cot(
    raw_path: Path,
    index_or_root: Mapping[tuple[str, str], Path] | Path,
) -> Path | None:
    if isinstance(index_or_root, Path):
        candidate = index_or_root / raw_path.parent.name / f"{raw_path.stem}_ecot.json"
        if candidate.is_file():
            return candidate
        # Keep a fallback for older COT trees whose chunk directory does not
        # mirror the raw tree exactly.
        return source_index(index_or_root).get((raw_path.parent.name, raw_path.stem))
    return index_or_root.get((raw_path.parent.name, raw_path.stem))


def task_goal(raw: Mapping[str, Any]) -> str:
    instruction = raw.get("instruction")
    if not isinstance(instruction, Mapping):
        return ""
    for key in ("general", "subtask"):
        values = instruction.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            conversations = item.get("conversations")
            if not isinstance(conversations, list):
                continue
            for turn in conversations:
                if not isinstance(turn, Mapping) or turn.get("from") != "human":
                    continue
                text = str(turn.get("value", ""))
                marker = "Generate robot actions for the task:"
                if marker in text:
                    return text.split(marker, 1)[1].strip()
                if text.strip():
                    return text.strip()
    return ""


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def lap_phrase(
    delta: Iterable[float],
    gripper: float | None,
    open_threshold: float,
    start_gripper: float | None = None,
) -> str:
    """Format world XYZ displacement and local relative XYZ Euler rotation."""
    values = list(delta)
    entries = (
        (0, "move forward", "move back", 100.0, "cm"),
        (1, "move left", "move right", 100.0, "cm"),
        (2, "move up", "move down", 100.0, "cm"),
        (3, "tilt left", "tilt right", 180.0 / math.pi, "degrees"),
        (4, "tilt back", "tilt forward", 180.0 / math.pi, "degrees"),
        (5, "rotate counterclockwise", "rotate clockwise", 180.0 / math.pi, "degrees"),
    )
    phrases: list[str] = []
    for index, positive, negative, scale, unit in entries:
        value = float(values[index]) * scale
        if unit == "degrees":
            value = _round_half_away(value / 5.0) * 5
        else:
            value = _round_half_away(value)
        if value:
            phrases.append(f"{positive if value > 0 else negative} {abs(value)} {unit}")
    if gripper is not None:
        is_open = gripper < open_threshold
        changed = start_gripper is not None and (start_gripper < open_threshold) != is_open
        if changed:
            phrases.append("open gripper" if is_open else "close gripper")
        else:
            phrases.append("keep gripper open" if is_open else "keep gripper closed")
    return ", ".join(phrases)


def _pose_series(raw: Mapping[str, Any], dataset: str = "") -> list[list[float]]:
    # Bridge and Fractal store an 8-D position-plus-quaternion record in
    # ``proprios.direct``.  Use the named EEF position and rotation matrix;
    # taking the first six values of ``direct`` would interpret quaternion
    # components as Euler angles.
    if dataset in ("bridge", "fractal"):
        proprios = raw.get("proprios")
        positions = proprios.get("left_ee_pos") if isinstance(proprios, Mapping) else None
        rotations = proprios.get("left_ee_rotm") if isinstance(proprios, Mapping) else None
        if isinstance(positions, list) and isinstance(rotations, list) and len(positions) == len(rotations):
            try:
                return [
                    [
                        *map(float, position),
                        *Rotation.from_matrix(
                            [rotation[0:3], rotation[3:6], rotation[6:9]]
                        ).as_euler("xyz").tolist(),
                    ]
                    for position, rotation in zip(positions, rotations)
                    if isinstance(position, list)
                    and len(position) >= 3
                    and isinstance(rotation, list)
                    and len(rotation) >= 9
                ]
            except (TypeError, ValueError):
                pass
    if dataset == "droid":
        actions = raw.get("actions")
        direct = actions.get("direct") if isinstance(actions, Mapping) else None
        if isinstance(direct, list) and direct and all(
            isinstance(row, list) and len(row) >= 6 for row in direct
        ):
            return [list(map(float, row[:6])) for row in direct]
    proprios = raw.get("proprios")
    if not isinstance(proprios, Mapping):
        return []
    direct = proprios.get("direct")
    if isinstance(direct, list) and direct and isinstance(direct[0], list) and len(direct[0]) >= 6:
        return [list(map(float, row[:6])) for row in direct]
    positions = proprios.get("left_ee_pos")
    rotations = proprios.get("left_ee_rotm")
    if isinstance(positions, list) and isinstance(rotations, list):
        return [
            [*map(float, position), *Rotation.from_matrix([rotation[0:3], rotation[3:6], rotation[6:9]]).as_euler("xyz").tolist()]
            for position, rotation in zip(positions, rotations)
        ]
    return []


def _gripper_series(raw: Mapping[str, Any], dataset: str) -> tuple[list[float], float, str]:
    if dataset == "libero":
        actions = raw.get("actions")
        values = actions.get("direct") if isinstance(actions, Mapping) else None
        if isinstance(values, list):
            return [float(row[6]) for row in values if isinstance(row, list) and len(row) > 6], 0.0, "actions.direct[:,6]: -1=open/release, +1=close/hold"
    if dataset == "droid":
        actions = raw.get("actions")
        values = actions.get("left_gripper_pos_direct") if isinstance(actions, Mapping) else None
        if isinstance(values, list):
            # LAP's formatter treats values below the threshold as open.  DROID
            # uses the inverse convention: values above 0.5 mean open.
            result: list[float] = []
            for row in values:
                if isinstance(row, list):
                    if not row:
                        continue
                    value = row[0]
                elif isinstance(row, (int, float)):
                    value = row
                else:
                    continue
                result.append(1.0 - float(value))
            return result, 0.5, "actions.left_gripper_pos_direct > 0.5 => open, <= 0.5 => close"
    if dataset in ("bridge", "fractal"):
        actions = raw.get("actions")
        values = actions.get("left_gripper_pos_direct") if isinstance(actions, Mapping) else None
        if isinstance(values, list):
            # The source convention is 1=open and 0=closed.  Normalize it to
            # the formatter's convention (values below 0.5 are open).
            result: list[float] = []
            for row in values:
                value = row[0] if isinstance(row, list) and row else row
                if isinstance(value, (int, float)):
                    result.append(1.0 - float(value))
            if result:
                return result, 0.5, (
                    "actions.left_gripper_pos_direct >= 0.5 => open, "
                    "< 0.5 => close; normalized to 1=open"
                )
        # Older Fractal exports do not carry the named action field.  Its
        # action.direct/proprio.direct gripper channel uses the same source
        # convention, so retain the fallback explicitly.
        for container_name in ("actions", "proprios"):
            container = raw.get(container_name)
            values = container.get("direct") if isinstance(container, Mapping) else None
            if isinstance(values, list) and values and all(isinstance(row, list) and len(row) > 6 for row in values):
                return [1.0 - float(row[6]) for row in values], 0.5, (
                    f"{container_name}.direct[:,6]: 1=open, 0=close; normalized to 1=open"
                )
    if dataset == "molmoact":
        actions = raw.get("actions")
        values = actions.get("direct") if isinstance(actions, Mapping) else None
        if isinstance(values, list) and values and all(isinstance(row, list) and len(row) > 6 for row in values):
            # MolmoAct's action direct channel is 0=open, 1=close.
            return [float(row[6]) for row in values], 0.5, (
                "actions.direct[:,6]: 0=open, 1=close"
            )
    proprios = raw.get("proprios")
    values = proprios.get("direct") if isinstance(proprios, Mapping) else None
    if isinstance(values, list):
        return [float(row[6]) for row in values if isinstance(row, list) and len(row) > 6], 0.5, "proprios.direct[:,6]: 0=open, 1=close"
    return [], 0.5, "unavailable"


def _spans(source: Mapping[str, Any], length: int) -> list[tuple[int, int]]:
    planning = source.get("planning", {})
    if isinstance(planning, Mapping):
        # The released VLABench annotations use ``plan_obj`` when it parsed
        # successfully.  A failed ``plan_obj`` is intentional: the old LAP
        # pass did not split movement text at the normalized fallback spans.
        # Prefer the explicit/parsed plans and only use the normalized plan
        # when no parsed plan object was supplied.
        plans: list[Any] = [planning]
        if "plan_obj" in planning:
            plans.append(planning.get("plan_obj", {}))
        elif "plan_normalized" in planning:
            plans.append(planning.get("plan_normalized", {}))
        for plan in plans:
            if not isinstance(plan, Mapping):
                continue
            if "_parse_error" in plan:
                return []
            subtasks = plan.get("subtasks", [])
            found = {
                (int(task["start"]), int(task["end"]))
                for task in subtasks
                if isinstance(task, Mapping) and "start" in task and "end" in task
                and 0 <= int(task["start"]) <= int(task["end"]) < length
            }
            if found:
                return sorted(found)
    # Frame spans are useful for raw/demo files that have no planning object,
    # but should not override a failed parsed plan above.
    frames = source.get("frames")
    found: set[tuple[int, int]] = set()
    if isinstance(frames, list):
        for frame in frames:
            span = frame.get("subtask_span") if isinstance(frame, Mapping) else None
            if not isinstance(span, Mapping):
                continue
            try:
                start, end = int(span["start"]), int(span["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= start < length and start <= end < length:
                found.add((start, end))
    return sorted(found)


def movement_for_frame(
    poses: list[list[float]],
    grippers: list[float],
    source: Mapping[str, Any],
    frame_index: int,
    horizon: int,
    open_threshold: float,
    merge_repeated_holds: bool = False,
) -> str:
    if not poses or frame_index >= len(poses):
        return ""
    # LAP uses a ten-frame future window as an exclusive endpoint.  Near the
    # end of an episode the endpoint can be one past the last recorded pose;
    # the final pose is then repeated while the gripper state is read from
    # endpoint - 1.  This is why terminal release/hold text is retained.
    target = min(frame_index + horizon, len(poses))
    if target <= frame_index:
        return lap_phrase([0.0] * 6, grippers[frame_index] if grippers else None, open_threshold)
    movement_meta = source.get("movement_annotation_meta", {})
    include_rotation = True
    if isinstance(movement_meta, Mapping) and movement_meta.get("include_rotation") is False:
        include_rotation = False
    if not include_rotation:
        merge_repeated_holds = True
    spans = _spans(source, len(poses))
    # A small subset of the legacy VLABench pass has no rotation text and a
    # one-frame terminal subtask. Its boundary was intentionally not emitted.
    skip_terminal_singleton = not include_rotation and spans and spans[-1][0] == len(poses) - 1 and spans[-1][1] == len(poses) - 1
    boundaries = [
        end + 1
        for _, end in spans
        if frame_index < end + 1 < target
        and not (skip_terminal_singleton and end + 1 == len(poses) - 1)
    ]
    points = [frame_index, *boundaries, target]
    parts: list[str] = []
    def pose_at(index: int) -> list[float]:
        return poses[min(index, len(poses) - 1)]

    def gripper_at(index: int) -> float:
        return grippers[min(index, len(grippers) - 1)]

    for start, end in zip(points, points[1:]):
        start_pose = pose_at(start)
        end_pose = pose_at(end)
        rotation_delta = [0.0, 0.0, 0.0]
        if include_rotation:
            start_rotation = Rotation.from_euler("xyz", start_pose[3:6])
            end_rotation = Rotation.from_euler("xyz", end_pose[3:6])
            rotation_delta = (start_rotation.inv() * end_rotation).as_euler("xyz").tolist()
        delta = [end_pose[axis] - start_pose[axis] for axis in range(3)] + rotation_delta
        phrase = lap_phrase(
            delta,
            gripper_at(end - 1) if grippers else None,
            open_threshold,
            gripper_at(start) if grippers else None,
        )
        # A stationary continuation with the same gripper state adds no
        # instruction. Keep holds when they precede motion or change state.
        if merge_repeated_holds and parts and phrase.startswith("keep gripper ") and parts[-1].endswith(phrase):
            continue
        parts.append(phrase)
    return "; then ".join(parts)


def _matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        return None
    return [[float(x) for x in row] for row in value]


def _matmul4(matrix: list[list[float]], point: list[float]) -> list[float]:
    return [sum(matrix[row][col] * ([*point, 1.0][col]) for col in range(4)) for row in range(3)]


def _camera_transform(calibration: Mapping[str, Any]) -> Any:
    for key in ("camera_from_world", "world_to_camera", "T_camera_from_world", "T_camera_optical_from_world"):
        matrix = _matrix(calibration.get(key))
        if matrix is not None:
            return matrix
    extrinsic = calibration.get("extrinsic")
    if isinstance(extrinsic, Mapping):
        rotation = extrinsic.get("rotation_matrix")
        translation = extrinsic.get("translation_vector")
        if isinstance(rotation, list) and len(rotation) == 3 and isinstance(translation, list) and len(translation) >= 3:
            # R and t map camera coordinates to world coordinates.
            return [[float(rotation[j][i]) for j in range(3)] + [-sum(float(rotation[k][i]) * float(translation[k]) for k in range(3))] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    return None


def _intrinsics(calibration: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    values = calibration.get("intrinsics", calibration.get("intrinsic", calibration))
    if not isinstance(values, Mapping):
        return None
    fx = values.get("fx", values.get("focal_x"))
    fy = values.get("fy", values.get("focal_y"))
    cx = values.get("cx", values.get("ppx", values.get("principal_x")))
    cy = values.get("cy", values.get("ppy", values.get("principal_y")))
    if None in (fx, fy, cx, cy):
        return None
    return float(fx), float(fy), float(cx), float(cy)


def project_waypoints(positions: list[list[float]], frame_index: int, horizon: int, calibration: Mapping[str, Any]) -> tuple[list[list[int]], list[int]]:
    # Future EEF positions are projected into the camera at the current frame,
    # including when a wrist/head camera moves during the episode.
    if "frames" in calibration:
        calibration = {**calibration, **calibration["frames"][frame_index]}
    camera_matrix = _camera_transform(calibration)
    intrinsics = _intrinsics(calibration)
    size = calibration.get("image_size", [calibration.get("width"), calibration.get("height")])
    if camera_matrix is None or intrinsics is None or not isinstance(size, list) or len(size) < 2:
        raise ValueError("calibration needs intrinsics, image_size and a camera/world transform")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("calibration image_size must be positive")
    fx, fy, cx, cy = intrinsics
    points: list[list[int]] = []
    world_from_pose = _matrix(calibration.get("world_from_pose"))
    for index in range(frame_index, min(frame_index + horizon, len(positions))):
        position = positions[index][:3]
        if world_from_pose is not None:
            position = _matmul4(world_from_pose, position)
        x, y, z = _matmul4(camera_matrix, position)
        if z <= 1e-8:
            points.append([-1, -1])
            continue
        u, v = fx * x / z + cx, fy * y / z + cy
        if not (0 <= u < width and 0 <= v < height):
            points.append([-1, -1])
        else:
            points.append([min(width - 1, int(round(u))), min(height - 1, int(round(v)))])
    while len(points) < horizon:
        points.append([-1, -1])
    return points, points[0] if points else [-1, -1]


def calibration_for_episode(
    root: Path | None, raw_path: Path, raw: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if root is None:
        for container in (raw, (raw or {}).get("extra_info")):
            if isinstance(container, Mapping):
                for key in ("camera_calibration", "calibration"):
                    if isinstance(container.get(key), dict):
                        return container[key]
        return None
    if root.is_file():
        return read_json(root)
    candidates = [
        root / raw_path.parent.name / f"{raw_path.stem}.json",
        root / raw_path.parent.name / f"{raw_path.stem}_calibration.json",
        root / f"{raw_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return read_json(candidate)
    raise FileNotFoundError(f"camera calibration not found for {raw_path.name} under {root}")


def _strip_paths(value: Any) -> Any:
    path_keys = {
        "episode_json",
        "episode_json_path",
        "video_path",
        "h5_path",
        "npz_path",
        "data_root",
        "source_path",
    }
    if isinstance(value, dict):
        return {
            key: (None if key in path_keys else _strip_paths(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    # ECOT metadata from older annotation jobs may use arbitrary names such
    # as ``mapped_from_h5_path``.  Strip the value by content as well as by
    # key so a public JSON cannot retain a machine-local absolute path.
    if isinstance(value, str) and os.path.isabs(value):
        return None
    return value


def build_episode(
    raw: dict[str, Any],
    language: dict[str, Any] | None,
    dataset: str,
    calibration: dict[str, Any] | None,
    horizon: int,
    keep_paths: bool,
    recompute_movement: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(language) if language is not None else {}
    if not result:
        goal = task_goal(raw)
        result = {
            "task_name": goal,
            "episode_goal": goal,
            "frames": [],
        }
    frames = result.get("frames")
    if not isinstance(frames, list) or len(frames) != int(raw.get("num_frames", 0)):
        frames = [frames[index] if isinstance(frames, list) and index < len(frames) and isinstance(frames[index], dict) else {} for index in range(int(raw.get("num_frames", 0)))]
        result["frames"] = frames
    poses = _pose_series(raw, dataset)
    grippers, open_threshold, gripper_convention = _gripper_series(raw, dataset)
    source_movement_meta = result.get("movement_annotation_meta", {})
    include_rotation = not (
        isinstance(source_movement_meta, Mapping)
        and source_movement_meta.get("include_rotation") is False
    )
    merge_repeated_holds = dataset in ("libero", "droid", "molmoact") or (
        dataset == "vlabench" and not include_rotation
    )
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if len(poses) != len(frames) or len(grippers) != len(frames):
        raise ValueError("EEF states, gripper signals and episode frames must have equal lengths")
    result.setdefault("suite", dataset)
    result.setdefault("task_name", task_goal(raw))
    result.setdefault("episode_goal", result.get("task_name", task_goal(raw)))
    result["T"] = int(raw.get("num_frames", len(frames)))
    result["T_eff"] = len(frames)
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            frame = {}
            frames[index] = frame
        frame["t"] = index
        frame.setdefault("Grounding", {})
        frame.setdefault("Understanding", {"goal": result.get("episode_goal", "")})
        frame.setdefault("Planning", {})
        acting = frame.setdefault("Acting", {})
        existing_movement = acting.get("movement_description")
        preserve_movement = (
            dataset in ("bridge", "fractal", "molmoact")
            and not recompute_movement
            and isinstance(existing_movement, str)
            and bool(existing_movement.strip())
        )
        if poses and not preserve_movement:
            acting["movement_description"] = movement_for_frame(
                poses, grippers, result, index, horizon, open_threshold,
                merge_repeated_holds=merge_repeated_holds,
            )
        if calibration is not None and poses:
            camera_calibrations = calibration.get("cameras", calibration) if isinstance(calibration, Mapping) else {}
            if isinstance(camera_calibrations, Mapping) and any(key.startswith("cam") and key[3:].isdigit() for key in camera_calibrations):
                projected, pose = {}, {}
                for camera_name, camera in camera_calibrations.items():
                    if isinstance(camera, Mapping):
                        projected[camera_name], pose[camera_name] = project_waypoints(poses, index, horizon, camera)
                if projected:
                    acting["waypoint_px"] = projected
                    acting["gripper_pose"] = pose
            else:
                projected, pose = project_waypoints(poses, index, horizon, calibration)
                acting["waypoint_px"] = {"cam0": projected}
                acting["gripper_pose"] = {"cam0": pose}
    movement_meta = {
        "method": (
            "libero_lap_per_frame_cross_boundary_split_concat"
            if dataset == "libero"
            else "lap_per_frame_cross_boundary_split_concat"
        ),
        "chunk_size": horizon,
        "dataset": dataset,
        "gripper_convention": gripper_convention,
        "translation_frame": "world",
        "rotation": "xyz Euler angles of R_start.inv() * R_end" if include_rotation else "omitted by source annotation",
        "translation_step_cm": 1,
        "rotation_step_degrees": 5,
        "gripper_indices": "segment start and end-1",
        "merge_repeated_holds": merge_repeated_holds,
        "include_rotation": include_rotation,
        "strict_frame_match": True,
        "preserve_existing_language_movement": (
            dataset in ("bridge", "fractal", "molmoact") and not recompute_movement
        ),
    }
    if dataset in ("bridge", "fractal"):
        movement_meta.update(
            {
                "motion_source_rule": (
                    "Use proprios.left_ee_pos and proprios.left_ee_rotm; "
                    "proprios.direct quaternion fields are not treated as Euler"
                ),
                "pose_source": "proprios.left_ee_pos + proprios.left_ee_rotm",
                "boundary_rule": (
                    "If a chunk crosses a COT subtask boundary, summarize each "
                    "segment piece separately and concatenate with '; then ' "
                    "while preserving total chunk size."
                ),
            }
        )
    if dataset == "molmoact":
        movement_meta.update(
            {
                "motion_source_rule": (
                    "Use proprios.direct[:, :6] (x, y, z, roll, pitch, yaw); "
                    "actions.direct[:,6] supplies the gripper state"
                ),
                "boundary_rule": (
                    "If a chunk crosses a COT subtask boundary, summarize each "
                    "segment piece separately and concatenate with '; then ' "
                    "while preserving total chunk size."
                ),
                "dedup_rule": (
                    "Suppress a repeated stationary gripper hold after a "
                    "subtask boundary when it carries no motion."
                ),
            }
        )
    if dataset == "droid":
        movement_meta.update(
            {
                "motion_source_rule": (
                    "Prefer actions.direct[:, :6]; otherwise compute per-step delta "
                    "from proprios.left_ee_pos and proprios.left_ee_rotm"
                ),
                "boundary_rule": (
                    "If a chunk crosses a COT subtask boundary, summarize each "
                    "segment piece separately and concatenate with '; then ' while "
                    "preserving total chunk size."
                ),
                "dedup_rule": (
                    "If a later piece has no motion text and repeats the same "
                    "gripper phrase as the previous piece, suppress that repeated piece."
                ),
            }
        )
    result["movement_annotation_meta"] = movement_meta
    if calibration is not None:
        result["waypoint_annotation_meta"] = {
            "method": "future_gripper_pose_project_world_to_image",
            "dataset": dataset,
            "waypoint_key": "waypoint_px",
            "horizon": horizon,
            "source": "proprios.left_ee_pos or proprios.direct",
            "projection_reference": "R.T @ (P - t), then pinhole intrinsics",
        }
    elif any(isinstance(frame, Mapping) and (frame.get("Acting") or {}).get("waypoint_px") for frame in frames):
        result.setdefault("waypoint_annotation_meta", {
            "method": "preserve_existing_waypoint_px",
            "dataset": dataset,
            "waypoint_key": "waypoint_px",
            "horizon": horizon,
            "source": "language_cot_root",
        })
    if not keep_paths:
        result = _strip_paths(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ECOT JSON for MiBot JSON+video data")
    parser.add_argument(
        "--dataset",
        choices=("libero", "vlabench", "droid", "bridge", "fractal", "molmoact"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="JSON+video train/data root")
    parser.add_argument("--language-cot-root", type=Path, help="Existing language-ECOT JSON root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, help="Optional per-episode camera calibration JSON")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--keep-paths", action="store_true", help="Keep source path metadata in ECOT JSON")
    parser.add_argument(
        "--recompute-movement",
        action="store_true",
        help="Regenerate movement text even when language COT already contains it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_files = episode_files(args.data_root)
    if args.max_episodes is not None:
        raw_files = raw_files[: args.max_episodes]
    if not raw_files:
        raise SystemExit(f"no episode JSON files found under {args.data_root}")
    language_index_or_root: Mapping[tuple[str, str], Path] | Path = (
        args.language_cot_root if args.language_cot_root else {}
    )
    for raw_path in raw_files:
        raw = read_json(raw_path)
        language_path = find_language_cot(raw_path, language_index_or_root)
        language = read_json(language_path) if language_path else None
        if args.language_cot_root and language is None:
            raise FileNotFoundError(f"language COT not found for {raw_path}")
        calibration = calibration_for_episode(args.calibration_root, raw_path, raw)
        built = build_episode(
            raw,
            language,
            args.dataset,
            calibration,
            args.horizon,
            args.keep_paths,
            args.recompute_movement,
        )
        output_path = args.output_root / raw_path.parent.name / f"{raw_path.stem}_ecot.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(built, handle, ensure_ascii=False, indent=2)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
