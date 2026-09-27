#!/usr/bin/env python3
"""Generate ECOT goal, plan, subtask name and rationale with a vision chat API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from build_json_cot import (
    _gripper_series, _pose_series, build_episode, calibration_for_episode,
    episode_files, read_json, task_goal,
)
from cot_prompts import PLANNING_PROMPT, REASONING_PROMPT, SYSTEM_PROMPT


class ChatClient:
    """A configurable /chat/completions endpoint; credentials stay in the environment."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "COT_API_KEY"):
        if not base_url or not model:
            raise ValueError("set --api-base and --language-model (or COT_API_BASE/COT_MODEL)")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        key = os.environ.get(api_key_env)
        if key:
            self.headers["Authorization"] = f"Bearer {key}"

    def complete(self, prompt: str, context: dict, images: list[tuple[int, Image.Image]]) -> dict:
        content = [{"type": "text", "text": prompt + "\nContext:\n" + json.dumps(context, ensure_ascii=False)}]
        for frame_index, image in images:
            image = image.convert("RGB")
            image.thumbnail((768, 768))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.extend([
                {"type": "text", "text": f"Frame {frame_index}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
            ])
        response = requests.post(
            self.url, headers=self.headers,
            json={"model": self.model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ], "temperature": 0, "max_tokens": 4096},
            timeout=180,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"]
        text = message.rsplit("</think>", 1)[-1].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("annotation API must return a JSON object")
        return result


def video_source(raw: dict, raw_path: Path) -> tuple[Path, int, int]:
    for camera in ("base", "ego", "wrist_left", "wrist_right"):
        entries = (raw.get("observations") or {}).get(camera)
        if not entries:
            continue
        if len(entries) != 1:
            raise ValueError(f"expected one continuous video for {raw_path.name}/{camera}")
        entry = entries[0]
        video = Path(entry["path"])
        if not video.is_absolute():
            video = next((parent / video for parent in raw_path.parents if (parent / video).is_file()), video)
        if not video.is_file():
            raise FileNotFoundError(video)
        start = int(entry.get("start", 0))
        end = int(entry.get("end", start + int(raw["num_frames"])))
        if end - start != int(raw["num_frames"]):
            raise ValueError(f"video slice and num_frames disagree: {raw_path}")
        return video, start, end
    raise ValueError(f"no camera video in {raw_path}")


def sample_indices(start: int, end: int, count: int) -> list[int]:
    return np.unique(np.linspace(start, end, min(count, end - start + 1), dtype=int)).tolist()


def validate_plan(plan: dict, length: int) -> list[dict]:
    if not isinstance(plan.get("goal"), str) or not plan["goal"].strip():
        raise ValueError("planner returned an empty goal")
    subtasks = plan.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError("planner returned no subtasks")
    next_start = 0
    for index, task in enumerate(subtasks, 1):
        if task.get("id") != index or type(task.get("start")) is not int or type(task.get("end")) is not int:
            raise ValueError("subtasks need sequential ids and integer frame indices")
        if task["start"] != next_start or not task["start"] <= task["end"] < length:
            raise ValueError("subtask spans must cover the episode without gaps or overlaps")
        for key in ("name", "planning"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise ValueError(f"subtask {index} is missing {key}")
        next_start = task["end"] + 1
    if next_start != length:
        raise ValueError("plan does not cover the final frame")
    return subtasks


def annotate_episode(raw: dict, raw_path: Path, cot: dict, dataset: str, client: ChatClient, num_frames: int = 16) -> dict:
    from decord import VideoReader, cpu

    video, start, end = video_source(raw, raw_path)
    reader = VideoReader(str(video), ctx=cpu(0))
    if end > len(reader):
        raise ValueError(f"video is shorter than the episode: {video}")
    length = int(raw["num_frames"])
    poses = _pose_series(raw, dataset)
    grippers, threshold, convention = _gripper_series(raw, dataset)

    def evidence(indices):
        return [{
            "t": t, "eef_pose": poses[t], "gripper": grippers[t],
            "visible_objects": cot["frames"][t].get("Grounding", {}).get("visible_objects", {}),
            "movement": cot["frames"][t].get("Acting", {}).get("movement_description", ""),
        } for t in indices]

    def images(indices):
        return [(t, Image.fromarray(reader[start + t].asnumpy())) for t in indices]

    indices = sample_indices(0, length - 1, num_frames)
    context = {
        "instruction": task_goal(raw), "dataset": dataset, "num_frames": length,
        "object_proposals": cot.get("object_proposals", []),
        "gripper_convention": convention, "normalized_open_below": threshold,
        "evidence": evidence(indices),
    }
    plan = client.complete(PLANNING_PROMPT, context, images(indices))
    subtasks = validate_plan(plan, length)
    full_plan = [{"id": task["id"], "subtask": task["name"]} for task in subtasks]
    # Complete all model calls before changing the output file.
    rationales = []
    for task in subtasks:
        indices = sample_indices(task["start"], task["end"], 3)
        reasoning = client.complete(REASONING_PROMPT, {
            "goal": plan["goal"], "plan": full_plan, "current_subtask": task,
            "gripper_convention": convention, "evidence": evidence(indices),
        }, images(indices)).get("subtask_reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(f"empty rationale for subtask {task['id']}")
        rationales.append(reasoning.strip())
    cot["episode_goal"] = plan["goal"]
    cot["planning"] = {"planner_model": client.model, "plan_obj": plan}
    cot["language_annotation_meta"] = {"reasoning_scope": "subtask", "prompt_version": 1}
    for task, reasoning in zip(subtasks, rationales):
        for t in range(task["start"], task["end"] + 1):
            frame = cot["frames"][t]
            frame["Understanding"] = {"goal": plan["goal"]}
            frame["Planning"] = {"plan": full_plan, "subtask_name": task["name"], "subtask_reasoning": reasoning}
            frame["subtask_span"] = {key: task[key] for key in ("id", "start", "end")}
            frame.setdefault("Acting", {})["subtask_end"] = task["end"]
    return cot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("vlabench", "libero", "droid", "bridge", "fractal", "molmoact"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", "--ecot-root", dest="output_root", type=Path, required=True)
    parser.add_argument("--language-model", default=os.environ.get("COT_MODEL", ""))
    parser.add_argument("--api-base", default=os.environ.get("COT_API_BASE", ""))
    parser.add_argument("--api-key-env", default="COT_API_KEY")
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--language-frames", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.language_frames < 1:
        parser.error("--language-frames must be positive")
    client = ChatClient(args.api_base, args.language_model, args.api_key_env)
    for raw_path in episode_files(args.data_root)[:args.max_episodes]:
        raw = read_json(raw_path)
        output = args.output_root / raw_path.parent.name / f"{raw_path.stem}_ecot.json"
        calibration = calibration_for_episode(args.calibration_root, raw_path, raw)
        cot = read_json(output) if output.is_file() else build_episode(raw, None, args.dataset, calibration, args.horizon, False)
        cot = annotate_episode(raw, raw_path, cot, args.dataset, client, args.language_frames)
        cot = build_episode(raw, cot, args.dataset, calibration, args.horizon, False, recompute_movement=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(cot, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
