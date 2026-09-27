#!/usr/bin/env python3
"""Create ECOT for a JSON+video episode with recorded EEF positions.

The input episode stays untouched. Language fields can be copied from an
optional existing ECOT file. If that file is omitted, this script writes the
small goal/plan scaffold needed by the MiBot loader. Movement is computed
from EEF positions and rotations. Camera calibration and tool-tip offsets
are supplied explicitly: projection uses ``R.T @ (P - t)`` and intrinsics.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from scipy.spatial.transform import Rotation

from build_json_cot import _strip_paths, movement_for_frame, task_goal


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _series(raw: Mapping[str, Any], key: str, length: int, dim: int) -> list[list[float]]:
    values = raw.get("proprios", {}).get(key)
    if values is None:
        return []
    if len(values) != length or any(len(row) != dim for row in values):
        raise ValueError(f"proprios.{key} must have shape [{length}, {dim}]")
    return [[float(value) for value in row] for row in values]


def _gripper(raw: Mapping[str, Any], side: str, length: int) -> list[float]:
    for key in (f"{side}_gripper_pos_direct", f"{side}_gripper_pos"):
        for container_name in ("actions", "proprios"):
            container = raw.get(container_name)
            values = container.get(key) if isinstance(container, Mapping) else None
            if not isinstance(values, list):
                continue
            result: list[float] = []
            for row in values[:length]:
                if isinstance(row, list) and row:
                    result.append(float(row[0]))
                elif isinstance(row, (int, float)):
                    result.append(float(row))
            if result:
                return result
    return []


def _goal(raw: Mapping[str, Any]) -> str:
    instruction = raw.get("instruction")
    if isinstance(instruction, Mapping):
        # Prefer the episode-specific instruction over a broader goal.
        for key in ("subtask", "general"):
            entries = instruction.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                for turn in entry.get("conversations", []):
                    if not isinstance(turn, Mapping) or turn.get("from") != "human":
                        continue
                    text = str(turn.get("value", ""))
                    marker = "Generate robot actions for the task:"
                    if marker in text:
                        return text.split(marker, 1)[1].strip()
    return task_goal(raw)


def _rotation(row: list[float] | None) -> list[list[float]] | None:
    if row is None or len(row) != 9:
        return None
    return [[row[row_index * 3 + column] for column in range(3)] for row_index in range(3)]


def _tip(position: list[float], rotation: list[list[float]] | None, offset: list[float]) -> list[float]:
    if rotation is None:
        if any(offset):
            raise ValueError("tool-tip offsets require recorded EEF rotations")
        return position
    return [position[axis] + sum(rotation[axis][j] * offset[j] for j in range(3)) for axis in range(3)]


def _project(point: list[float], extrinsic: Mapping[str, Any], intrinsic: Mapping[str, Any]) -> list[int]:
    data = extrinsic.get("extrinsic", extrinsic)
    rotation = data.get("rotation_matrix") if isinstance(data, Mapping) else None
    translation = data.get("translation_vector") if isinstance(data, Mapping) else None
    if not isinstance(rotation, list) or len(rotation) != 3 or not isinstance(translation, list) or len(translation) < 3:
        raise ValueError("extrinsics need camera-to-world rotation_matrix and translation_vector")
    camera = [
        sum(float(rotation[column][row]) * (point[column] - float(translation[column])) for column in range(3))
        for row in range(3)
    ]
    if camera[2] <= 1e-8:
        return [-1, -1]
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic.get("ppx", intrinsic.get("cx")))
    cy = float(intrinsic.get("ppy", intrinsic.get("cy")))
    return [int(round(fx * camera[0] / camera[2] + cx)), int(round(fy * camera[1] / camera[2] + cy))]


def _copy_language_frames(language: Mapping[str, Any] | None, length: int, goal: str) -> list[dict[str, Any]]:
    source = language.get("frames") if isinstance(language, Mapping) else None
    if source is not None and len(source) != length:
        raise ValueError("language COT and raw episode must have equal frame counts")
    frames: list[dict[str, Any]] = []
    for index in range(length):
        source_frame = source[index] if isinstance(source, list) and index < len(source) and isinstance(source[index], Mapping) else {}
        frame: dict[str, Any] = {"t": index}
        for key in ("Grounding", "Understanding", "Planning", "Acting", "subtask_span"):
            if key in source_frame:
                frame[key] = copy.deepcopy(source_frame[key])
        frame.setdefault("Grounding", {"visible_objects": {}})
        frame.setdefault("Understanding", {"goal": goal})
        frame.setdefault(
            "Planning",
            {
                "plan": [{"id": 1, "subtask": goal}],
                "subtask_name": goal,
                "subtask_reasoning": "Follow the demonstrated EEF trajectory to complete the task.",
            },
        )
        frame.setdefault("Acting", {})
        frames.append(frame)
    return frames


def build_episode(
    raw: dict[str, Any],
    language: dict[str, Any] | None,
    horizon: int,
    intrinsic: dict[str, Any] | None = None,
    extrinsics: list[dict[str, Any]] | dict[str, Any] | None = None,
    camera_name: str = "cam0",
    tip_offsets: dict[str, list[float]] | None = None,
    gripper_open_threshold: float = 0.5,
    gripper_open_above: bool = False,
    merge_repeated_holds: bool = False,
) -> dict[str, Any]:
    length = int(raw.get("num_frames", 0))
    if length < 1 or horizon < 1:
        raise ValueError("num_frames and horizon must be positive")
    goal = str((language or {}).get("episode_goal") or _goal(raw) or "Complete the demonstrated task.")
    frames = _copy_language_frames(language, length, goal)
    positions = {side: _series(raw, f"{side}_ee_pos", length, 3) for side in ("left", "right")}
    rotations = {
        side: [_rotation(row) for row in _series(raw, f"{side}_ee_rotm", length, 9)]
        for side in ("left", "right")
    }
    grippers = {side: _gripper(raw, side, length) for side in ("left", "right")}
    arms = [side for side in ("left", "right") if positions[side]]
    if not arms:
        raise ValueError("episode needs proprios.left_ee_pos or proprios.right_ee_pos in meters")
    poses = {}
    for side in arms:
        if grippers[side] and len(grippers[side]) != length:
            raise ValueError(f"{side} gripper signal must contain {length} frames")
        angles = Rotation.from_matrix(rotations[side]).as_euler("xyz").tolist() if rotations[side] else [[0.0] * 3 for _ in range(length)]
        poses[side] = [position + angle for position, angle in zip(positions[side], angles)]
        grippers[side] = [
            float(value <= gripper_open_threshold if gripper_open_above else value >= gripper_open_threshold)
            for value in grippers[side]
        ]
    if (intrinsic is None) != (extrinsics is None):
        raise ValueError("provide both intrinsic and extrinsic calibration")
    if intrinsic is not None:
        intrinsic = intrinsic.get("intrinsic", intrinsic)
        if isinstance(extrinsics, dict):
            extrinsics = [extrinsics] * length
        if len(extrinsics) != length:
            raise ValueError("per-frame extrinsics must be aligned to this episode; supply one entry per frame")
    tip_offsets = tip_offsets or {}

    for index, frame in enumerate(frames):
        movements = [
            movement_for_frame(poses[side], grippers[side], language or {}, index, horizon, 0.5, merge_repeated_holds)
            for side in arms
        ]
        frame["Acting"]["movement_description"] = (
            "; ".join(f"{side} arm: {text}" for side, text in zip(arms, movements))
            if len(arms) > 1 else movements[0]
        )
        if intrinsic is None:
            continue
        waypoints: dict[str, list[list[int]]] = {}
        gripper_poses: dict[str, list[int]] = {}
        for side in arms:
            points: list[list[int]] = []
            for target in range(index, min(index + horizon, len(positions[side]))):
                rotation = rotations[side][target] if target < len(rotations[side]) else None
                point = _tip(positions[side][target], rotation, tip_offsets.get(side, [0.0, 0.0, 0.0]))
                points.append(_project(point, extrinsics[target], intrinsic))
            while len(points) < horizon:
                points.append([-1, -1])
            waypoints[side] = points
            gripper_poses[side] = points[0]
        frame["Acting"].setdefault("waypoint_px", {})[camera_name] = waypoints if len(arms) > 1 else waypoints[arms[0]]
        frame["Acting"].setdefault("gripper_pose", {})[camera_name] = gripper_poses if len(arms) > 1 else gripper_poses[arms[0]]

    result = {
        "task_name": str((language or {}).get("task_name") or goal),
        "episode_goal": goal,
        "T": length,
        "T_eff": length,
        "frames": frames,
        "movement_annotation_meta": {
            "method": "lap_per_frame_cross_boundary_split_concat",
            "chunk_size": horizon,
            "source": [f"proprios.{side}_ee_pos" for side in arms],
            "translation_frame": "world",
            "rotation": "xyz Euler angles of R_start.inv() * R_end",
            "include_rotation": {side: bool(rotations[side]) for side in arms},
            "translation_step_cm": 1,
            "rotation_step_degrees": 5,
            "gripper_indices": "segment start and end-1",
            "gripper_open_threshold": gripper_open_threshold,
            "gripper_open_above": gripper_open_above,
            "merge_repeated_holds": merge_repeated_holds,
            "joint_positions_used": False,
        },
    }
    if language and "suite" in language:
        result["suite"] = language["suite"]
    if intrinsic is not None:
        result["waypoint_annotation_meta"] = {
            "method": "project_world_to_image",
            "camera": camera_name,
            "waypoint_key": "waypoint_px",
            "horizon": horizon,
            "coordinate_units": "pixels in the calibration image",
            "tip_offsets": tip_offsets,
            "projection": "R.T @ (P - t), then pinhole intrinsics",
        }
    return _strip_paths(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ECOT from recorded EEF states in a JSON+video episode")
    parser.add_argument("--raw-episode", type=Path, required=True)
    parser.add_argument("--language-cot", type=Path)
    parser.add_argument("--intrinsic-file", type=Path, help="Pinhole intrinsics: fx, fy, ppx/cx, ppy/cy")
    parser.add_argument("--extrinsic-file", type=Path, help="Camera-to-world extrinsics: one static object or an episode-aligned list")
    parser.add_argument("--camera-name", default="cam0")
    parser.add_argument("--left-tip-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="Tool-tip offset in the local EEF frame, in meters")
    parser.add_argument("--right-tip-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-open-above", action="store_true", help="Use for gripper signals where larger values mean open")
    parser.add_argument("--merge-repeated-holds", action="store_true", help="Omit a redundant stationary gripper hold after a subtask boundary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = _read(args.raw_episode)
    language = _read(args.language_cot) if args.language_cot else None
    if language is not None and not isinstance(language, dict):
        raise ValueError(f"language COT must be a JSON object: {args.language_cot}")
    output = build_episode(
        raw, language, args.horizon,
        intrinsic=_read(args.intrinsic_file) if args.intrinsic_file else None,
        extrinsics=_read(args.extrinsic_file) if args.extrinsic_file else None,
        camera_name=args.camera_name,
        tip_offsets={"left": args.left_tip_offset, "right": args.right_tip_offset},
        gripper_open_threshold=args.gripper_open_threshold,
        gripper_open_above=args.gripper_open_above,
        merge_repeated_holds=args.merge_repeated_holds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
