#!/usr/bin/env python3
"""Replay source HDF5 demonstrations to annotate simulator boxes and EEF pixels.

LIBERO restores each recorded MuJoCo state. VLABench follows the benchmark's
joint replay with the saved episode configuration. A replay index maps exported
JSON names to source demonstrations when the conversion discarded that mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import h5py
import numpy as np

from annotate_language_cot import video_source
from build_json_cot import _pose_series, episode_files, project_waypoints, read_json


def resolve_source(raw_path, raw, source_root, replay_index):
    key = f"{raw_path.parent.name}/{raw_path.name}"
    record = replay_index.get(key, raw.get("extra_info") or {})
    if isinstance(record, str):
        record = {"source_path": record}
    path = Path(record.get("source_path") or "")
    if source_root is not None:
        path = source_root / (Path(*path.parts[-2:]) if path.is_absolute() else path)
    if path.suffix not in (".hdf5", ".h5") or not path.is_file():
        raise FileNotFoundError(f"{key}: provide the source HDF5 path in --replay-index")
    return path, record


def video_reference(raw, raw_path):
    path, start, _ = video_source(raw, raw_path)
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def align_image(image, reference):
    variants = ((False, False), (True, False), (False, True), (True, True))
    def error(flips):
        candidate = transform_image(image, flips)
        return float(np.abs(candidate.astype(float) - reference.astype(float)).mean())
    flips = min(variants, key=error)
    return flips, error(flips)


def transform_image(image, flips):
    flip_x, flip_y = flips
    return image[::(-1 if flip_y else 1), ::(-1 if flip_x else 1)]


def camera_calibration(model, data, camera_id, width, height, flips):
    # MuJoCo cameras look down -Z; ECOT's pinhole projector uses optical +Z.
    rotation = np.diag([1., -1., -1.]) @ np.asarray(data.cam_xmat[camera_id]).reshape(3, 3).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ data.cam_xpos[camera_id]
    focal = height / (2 * np.tan(np.deg2rad(float(model.cam_fovy[camera_id])) / 2))
    fx, fy, cx, cy = focal, focal, width / 2, height / 2
    if flips[0]:
        fx, cx = -fx, width - 1 - cx
    if flips[1]:
        fy, cy = -fy, height - 1 - cy
    return {"intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "image_size": [width, height], "camera_from_world": transform.tolist()}


def visible_boxes(ids, entities):
    height, width = ids.shape
    boxes = []
    for name, geometry_ids in entities.items():
        ys, xs = np.nonzero(np.isin(ids, geometry_ids))
        if len(xs):
            bbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            boxes.append({"entity": name, "bbox_xyxy": [round(1000 * int(x) / scale) for x, scale in zip(bbox, [width, height] * 2)]})
    return {"cam0": boxes}


def frame_indices(record, length, source_length):
    indices = np.asarray(record.get("frame_indices", list(range(length))), dtype=int)
    if len(indices) != length or np.any(indices < 0) or np.any(indices >= source_length):
        raise ValueError("replay frame_indices must map each exported frame into the source episode")
    if "frame_indices" not in record and length != source_length:
        raise ValueError("exported/source frame counts differ; supply frame_indices in the replay index")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("replay frame_indices must be strictly increasing")
    return indices


def libero_xml(xml, assets_root):
    import robosuite

    root = ET.fromstring(xml)
    for element in root.iter():
        source = element.get("file")
        if source is None:
            continue
        if "/robosuite/" in source:
            path = Path(robosuite.__file__).parent / source.split("/robosuite/", 1)[1]
        elif "/assets/" in source:
            path = assets_root / source.split("/assets/", 1)[1]
        else:
            path = Path(source)
        element.set("file", str(path))
    return ET.tostring(root, encoding="unicode")


def replay_libero(raw, source, record, reference, args):
    import mujoco

    if args.assets_root is None:
        raise ValueError("LIBERO replay needs --assets-root pointing to the LIBERO asset bundle")
    with h5py.File(source, "r", locking=False) as handle:
        demo = handle["data"][record.get("demo", record.get("task_id", "demo_0"))]
        states = demo["states"][:]
        indices = frame_indices(record, int(raw["num_frames"]), len(states))
        positions = demo["obs/ee_pos"][:][indices].tolist()
        model = mujoco.MjModel.from_xml_string(libero_xml(demo.attrs["model_file"], args.assets_root))
    if not np.allclose(np.asarray(_pose_series(raw, "libero"))[:, :3], positions, atol=1e-4):
        raise ValueError("LIBERO source EEF positions do not match the exported episode")
    data = mujoco.MjData(model)
    camera_name = record.get("camera", "agentview")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"unknown replay camera {camera_name}")
    entities = {}
    for geom_id in range(model.ngeom):
        body = int(model.geom_bodyid[geom_id])
        while model.body_parentid[body] != 0:
            body = int(model.body_parentid[body])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world"
        if name == "world" or name.startswith(("robot", "gripper", "mount")):
            continue
        entities.setdefault(name.removesuffix("_main"), []).append(geom_id)
    height, width = reference.shape[:2]
    frames, calibrations = [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for t, index in enumerate(indices):
            state = states[index]
            if len(state) != 1 + model.nq + model.nv:
                raise ValueError("LIBERO state length does not match the recorded MuJoCo model")
            data.time = state[0]
            data.qpos[:] = state[1:1 + model.nq]
            data.qvel[:] = state[1 + model.nq:]
            mujoco.mj_forward(model, data)
            if t == 0:
                renderer.update_scene(data, camera=camera_name)
                flips, error = align_image(renderer.render(), reference)
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=camera_name)
            segmentation = transform_image(renderer.render(), flips)
            ids = np.where(segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), segmentation[..., 0], -1)
            camera = camera_calibration(model, data, camera_id, width, height, flips)
            waypoints, gripper = project_waypoints(positions, t, args.horizon, camera)
            frames.append({"t": t, "Grounding": {"visible_objects": visible_boxes(ids, entities)},
                           "Acting": {"gripper_pose": {"cam0": gripper}, "waypoint_px": {"cam0": waypoints}}})
            calibrations.append(camera)
    return frames, calibrations, {"method": "libero_recorded_mujoco_states", "first_frame_mae": error, "image_flips": flips}


def replay_vlabench(raw, source, record, reference, args):
    from VLABench.envs import load_env

    with h5py.File(source, "r", locking=False) as handle:
        episode = handle["data"][record.get("demo", next(iter(handle["data"])))]
        config = json.loads(episode["meta_info/episode_config"][()].decode())
        ee_states = episode["observation/ee_state"][:]
        joints = episode["observation/q_state"][:].reshape(-1, 7)
    indices = frame_indices(record, int(raw["num_frames"]), len(ee_states))
    positions = ee_states[indices, :3].tolist()
    env = load_env(record.get("task", source.parent.name), reset_wait_step=0, episode_config=config, random_init=False)
    try:
        offset = np.asarray(env.robot.robot_config["position"])
        if not np.allclose(np.asarray(_pose_series(raw, "vlabench"))[:, :3] + offset, positions, atol=1e-4):
            raise ValueError("VLABench source EEF positions do not match the exported episode")
        entities = {name: [int(env.physics.bind(geom).element_id) for geom in entity.geoms]
                    for name, entity in env.task.entities.items()}
        height, width = reference.shape[:2]
        camera_id = int(record.get("camera", 2))
        selected = {int(index): t for t, index in enumerate(indices)}
        frames, calibrations = [], []
        for index in range(int(indices[-1]) + 1):
            if index in selected:
                t = selected[index]
                if t == 0:
                    image = env.physics.render(height=height, width=width, camera_id=camera_id)
                    flips, error = align_image(image, reference)
                seg = env.physics.render(height=height, width=width, camera_id=camera_id, segmentation=True)
                # dm_control returns (object id, object type), in image order.
                seg = transform_image(seg, flips)
                ids = np.where(seg[..., 1] == 5, seg[..., 0], -1)
                camera = camera_calibration(env.physics.model, env.physics.data, camera_id, width, height, flips)
                waypoints, gripper = project_waypoints(positions, t, args.horizon, camera)
                frames.append({"t": t, "Grounding": {"visible_objects": visible_boxes(ids, entities)},
                               "Acting": {"gripper_pose": {"cam0": gripper}, "waypoint_px": {"cam0": waypoints}}})
                # Export calibration in the pose coordinate frame of the JSON.
                world_from_pose = np.eye(4)
                world_from_pose[:3, 3] = offset
                camera["world_from_pose"] = world_from_pose.tolist()
                calibrations.append(camera)
            gripper = 0.0 if ee_states[index, -1] > 0 else 0.04
            env.step(np.r_[joints[index], gripper, gripper])
    finally:
        env.close()
    return frames, calibrations, {"method": "vlabench_episode_config_joint_replay", "first_frame_mae": error, "image_flips": flips}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("libero", "vlabench"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--replay-index", type=Path)
    parser.add_argument("--simulator-root", type=Path)
    parser.add_argument("--assets-root", type=Path, default=os.environ.get("LIBERO_ASSETS_ROOT"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.simulator_root:
        sys.path.insert(0, str(args.simulator_root.resolve()))
        if args.dataset == "vlabench":
            os.environ["VLABENCH_ROOT"] = str(args.simulator_root.resolve())
    replay_index = read_json(args.replay_index) if args.replay_index else {}
    for raw_path in episode_files(args.data_root)[:args.max_episodes]:
        raw = read_json(raw_path)
        source, record = resolve_source(raw_path, raw, args.source_root, replay_index)
        reference = video_reference(raw, raw_path)
        replay = replay_libero if args.dataset == "libero" else replay_vlabench
        frames, cameras, meta = replay(raw, source, record, reference, args)
        output = args.output_root / raw_path.parent.name / f"{raw_path.stem}_ecot.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"frames": frames, "replay_meta": meta}, indent=2) + "\n")
        calibration = args.output_root / "calibration" / raw_path.parent.name / raw_path.name
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_text(json.dumps({"cameras": {"cam0": {"frames": cameras}}}, indent=2) + "\n")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
