#!/usr/bin/env python3
"""Annotate all ECOT fields from raw JSON+video episodes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from annotate_language_cot import ChatClient, annotate_episode, video_source
from build_json_cot import build_episode, calibration_for_episode, episode_files, read_json
from cot_grounding import LLMDetGrounding, propose_objects


DATASETS = ("vlabench", "libero", "droid", "bridge", "fractal", "molmoact")
SIM_DATASETS = ("vlabench", "libero")
COT_FIELDS = {
    "Understanding": ("goal",),
    "Grounding": ("visible_objects",),
    "Planning": ("plan", "subtask_name", "subtask_reasoning"),
    "Acting": ("movement_description", "gripper_pose", "waypoint_px"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("COT_API_BASE", ""))
    parser.add_argument("--api-key-env", default="COT_API_KEY")
    parser.add_argument("--language-model", default=os.environ.get("COT_MODEL", ""))
    parser.add_argument("--proposal-model", default="", help="Defaults to --language-model")
    parser.add_argument("--language-frames", type=int, default=16)
    parser.add_argument("--bbox-source", choices=("auto", "llmdet", "simulator"), default="auto")
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--detector-device", default="")
    parser.add_argument("--detector-threshold", type=float, default=0.4)
    parser.add_argument("--calibration-root", type=Path, help="Shared or per-episode camera calibration JSON")
    parser.add_argument("--simulator-python", default=sys.executable)
    parser.add_argument("--simulator-root", type=Path, help="Benchmark checkout used for replay")
    parser.add_argument("--source-root", type=Path, help="Root of the original simulator HDF5 demonstrations")
    parser.add_argument("--replay-index", type=Path, help="Mapping from exported episodes to original HDF5 demos")
    parser.add_argument("--assets-root", type=Path, default=os.environ.get("LIBERO_ASSETS_ROOT"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.language_frames < 1 or args.horizon < 1 or (args.max_episodes is not None and args.max_episodes < 1):
        parser.error("frame, horizon and episode counts must be positive")
    if args.bbox_source == "auto":
        args.bbox_source = "simulator" if args.dataset in SIM_DATASETS else "llmdet"
    if args.bbox_source == "simulator" and args.dataset not in SIM_DATASETS:
        parser.error("simulator replay is available for VLABench and LIBERO")
    return args


def extract_frames(raw, raw_path, root):
    from decord import VideoReader, cpu
    from PIL import Image

    video, start, end = video_source(raw, raw_path)
    reader = VideoReader(str(video), ctx=cpu(0))
    if end > len(reader):
        raise ValueError(f"video is shorter than the episode: {video}")
    output = root / raw_path.parent.name / raw_path.stem
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for t, index in enumerate(range(start, end)):
        path = output / f"frame_{t:06d}.jpg"
        Image.fromarray(reader[index].asnumpy()).save(path, quality=95)
        paths.append(path)
    return paths


def replay_command(args, raw_path, output_root):
    command = [args.simulator_python, str(Path(__file__).with_name("replay_simulator_cot.py")),
               "--dataset", args.dataset, "--data-root", str(raw_path),
               "--output-root", str(output_root), "--horizon", str(args.horizon)]
    for key in ("simulator_root", "source_root", "replay_index", "assets_root"):
        value = getattr(args, key)
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def validate_complete(cot, length):
    frames = cot.get("frames", [])
    if len(frames) != length:
        raise ValueError("CoT frame count does not match the raw episode")
    for t, frame in enumerate(frames):
        for section, keys in COT_FIELDS.items():
            for key in keys:
                value = frame.get(section, {}).get(key)
                if value is None or value == "" or value == {} or value == []:
                    raise ValueError(f"frame {t} is missing {section}.{key}")
    # A camera may legitimately have no detections; invisible EEF points use [-1, -1].


def main():
    args = parse_args()
    raw_files = episode_files(args.data_root)[:args.max_episodes]
    if not raw_files:
        raise SystemExit(f"no episode JSON files under {args.data_root}")
    if args.dry_run:
        print(f"{len(raw_files)} episodes: JSON+video -> VLM proposals -> {args.bbox_source} -> language CoT -> LAP movement + projection")
        if args.bbox_source == "simulator":
            print(shlex.join(replay_command(args, raw_files[0], args.output_root / "_simulator")))
        print(f"language model: {args.language_model or '<MODEL>'}; API: {args.api_base or '<API_BASE>'}")
        return
    client = ChatClient(args.api_base, args.language_model, args.api_key_env)
    proposal_client = ChatClient(args.api_base, args.proposal_model or args.language_model, args.api_key_env)
    detector = None
    if args.bbox_source == "llmdet" and not args.detector_model:
        raise SystemExit("set --detector-model for real-data grounding")
    for raw_path in raw_files:
        raw = read_json(raw_path)
        output = args.output_root / raw_path.parent.name / f"{raw_path.stem}_ecot.json"
        if output.is_file() and not args.overwrite:
            validate_complete(read_json(output), int(raw["num_frames"]))
            print(f"skip complete {output}")
            continue
        calibration = calibration_for_episode(args.calibration_root, raw_path, raw)
        if args.bbox_source == "llmdet" and calibration is None:
            raise ValueError(f"{raw_path.name}: supply --calibration-root or embedded camera_calibration for gripper/trajectory projection")
        cot = build_episode(raw, None, args.dataset, calibration, args.horizon, False, recompute_movement=True)
        frame_paths = extract_frames(raw, raw_path, args.output_root / "_frames")
        cot["object_proposals"] = propose_objects(proposal_client, frame_paths)
        if args.bbox_source == "llmdet":
            if detector is None:
                detector = LLMDetGrounding(args.detector_model, args.detector_device, args.detector_threshold)
            detector.annotate(cot, frame_paths, cot["object_proposals"])
        else:
            sim_root = args.output_root / "_simulator"
            subprocess.run(replay_command(args, raw_path, sim_root), check=True)
            replay = read_json(sim_root / raw_path.parent.name / output.name)
            if len(replay["frames"]) != len(cot["frames"]):
                raise ValueError("simulator replay frame count mismatch")
            for target, source in zip(cot["frames"], replay["frames"]):
                target["Grounding"] = source["Grounding"]
                target["Acting"].update(source["Acting"])
            cot["bbox_source"] = "simulator"
            cot["replay_meta"] = replay["replay_meta"]
            if calibration is None:
                calibration = calibration_for_episode(sim_root / "calibration", raw_path, raw)
        cot = annotate_episode(raw, raw_path, cot, args.dataset, client, args.language_frames)
        # Planning defines LAP's segment boundaries, so recompute movement last.
        cot = build_episode(raw, cot, args.dataset, calibration, args.horizon, False, recompute_movement=True)
        validate_complete(cot, int(raw["num_frames"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(cot, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
