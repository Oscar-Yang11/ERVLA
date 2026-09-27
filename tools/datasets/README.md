# CoT annotation inputs

`annotate_cot_from_raw.py` takes raw episode JSON files and their videos. It
generates VLM object proposals, grounding, language annotations, LAP movement,
and projected EEF points. The language model must accept images through a
`/chat/completions` API. Set `COT_API_BASE` to its base URL, `COT_API_KEY` to
the credential, and `COT_MODEL` to its model name. No API keys or endpoints
are embedded in the code.

The prompts are in `cot_prompts.py`. The planner returns `goal` and contiguous,
inclusive subtask intervals; a second call generates a short rationale for each
subtask. These produce the existing `Understanding` and `Planning` fields on
every frame. A subtask's rationale is shared by its frames. The prompts use the
collaborator annotation schema; they are not recovered copies of the original
annotation prompts.

## Camera calibration

Real-data annotation reads `camera_calibration` from the episode or its
`extra_info`, or from `--calibration-root`. That argument can be one shared
JSON file, or a directory containing `<chunk>/<episode>.json` for each episode:

```json
{
  "cameras": {
    "cam0": {
      "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
      "image_size": [640, 480],
      "camera_from_world": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ]
    }
  }
}
```

Replace these example values with the recorded calibration for the video.
`camera_from_world` maps world points into optical coordinates (+Z forward).
If EEF positions are in the robot base frame, also provide `world_from_pose`,
a 4×4 transform from that frame into world coordinates. Camera parameters must
match the video's resolution, crop and orientation. A moving camera can supply
a `frames` list of per-frame calibration dictionaries under `cam0`; projection
uses the current camera pose for all future waypoints.

`cam0` denotes the selected base/ego video. `gripper_pose` and `waypoint_px` are
pixels; invisible points use `[-1, -1]`. Detection boxes follow the existing
grounding scripts' 0–1000 coordinate convention. Extra calibrated views may be
provided under `cam1`, etc.; the language/grounding stages use the primary view.

## Simulator sources

`replay_simulator_cot.py` runs in the benchmark's simulation environment.
LIBERO uses the source HDF5 `model_file`, `states` and `obs/ee_pos`, plus the
LIBERO asset bundle. VLABench uses `meta_info/episode_config`,
`observation/q_state` and `observation/ee_state` with the benchmark's joint
replay. No precomputed bbox annotations are required.

When raw JSON includes `extra_info.source_path` pointing to the original HDF5,
the replay uses it directly. `--source-root` relocates that path to
`<source-root>/<suite-or-task>/<filename>`. An export that only points to a
Parquet file needs a `--replay-index` with the source mapping from conversion:

```json
{
  "chunk-0000/episode_0000000.json": {
    "source_path": "task_name/episode_123.hdf5",
    "task": "task_name",
    "camera": 2
  }
}
```

For LIBERO, an index entry can instead specify `demo: "demo_0"` and
`camera: "agentview"`. If conversion selected only some source frames, include
the ordered `frame_indices` list. The source EEF positions are checked against
the exported JSON before replay. The rendered image orientation is aligned to
the recorded video's first frame and saved in `replay_meta` with its image
error, so alignment can be inspected. Replay also exports camera calibration
under `_simulator/calibration`.

VLABench joint replay follows the official augmentation script; its contact
dynamics can differ from the original rollout. LIBERO restores recorded full
states. Check a replay overlay when using a new dataset conversion or simulator
version.

## Running individual stages

```bash
# Language fields only, retaining existing geometry when available
bash scripts/generate_language_cot.sh \
  --dataset molmoact --data-root path/to/dataset/json \
  --output-root outputs/cot/molmoact

# Movement and projection only
python tools/datasets/build_json_cot.py \
  --dataset molmoact --data-root path/to/dataset/json \
  --calibration-root path/to/dataset/camera-calibrations \
  --output-root outputs/cot/molmoact

# Offline request/schema/geometry checks; no remote models
python -m unittest discover -s tests -p test_cot_annotation.py
```

The full command checks that all eight training fields are present before
saving an episode. An empty detection list for a camera is allowed. It skips
existing complete output files; use `--overwrite` to regenerate them.
