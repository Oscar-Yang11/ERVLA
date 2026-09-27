#!/usr/bin/env python3
"""Run the existing VLM-proposal and LLMDet grounding stages.

The repository keeps the model-specific annotation implementations in the
data-construction checkout.  This small adapter gives them one stable entry
point and preserves the original three-stage flow:

    keyframes/raw_pic -> VLM object proposal -> LLMDet grounding -> ECOT JSON

The ECOT directory must already contain the language/planning scaffold.  The
grounding stage updates only ``Grounding.visible_objects``.  Use ``--dry-run``
to inspect the commands when model names or checkpoints are not available.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROPOSAL_SCRIPTS = {
    # The unified raw-pic proposal script is used for all datasets whose
    # frames are extracted before this adapter is called.  The old Bridge
    # video-reading entry point is still supported by generate_cot.sh when an
    # existing Bridge ECOT tree is supplied.
    "bridge": ("1_vlm_proposal", "unified_ecot_proposal_qwen35.py"),
    "fractal": ("1_vlm_proposal", "unified_ecot_proposal_qwen35.py"),
    "molmoact": ("1_vlm_proposal", "unified_ecot_proposal_qwen35.py"),
    "droid": ("1_vlm_proposal", "unified_ecot_proposal_qwen35.py"),
}
GROUNDING_SCRIPT = "2_llmdet_grounding/run_llmdet_grounding_{dataset}_episode_rulemerge_writeecot.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PROPOSAL_SCRIPTS), required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--ecot-root", type=Path, required=True)
    parser.add_argument("--raw-pic-root", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--bbox-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--proposal-model", default="", help="VLM client model name")
    parser.add_argument("--detector-model", default="", help="Local LLMDet checkpoint")
    parser.add_argument("--proposal-workers", type=int, default=1)
    parser.add_argument("--grounding-workers", type=int, default=1)
    parser.add_argument("--gpus", default="", help="GPU ids for grounding, for example 0")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the two commands without requiring model names/checkpoints",
    )
    return parser.parse_args()


def _print_command(label: str, command: list[str]) -> None:
    print(f"[cot] {label}: {shlex.join(command)}")


def main() -> None:
    args = parse_args()
    proposal_dir = args.proposal_dir
    bbox_dir = args.bbox_dir

    proposal_subdir, proposal_filename = PROPOSAL_SCRIPTS[args.dataset]
    proposal_script = args.pipeline_root / proposal_subdir / proposal_filename
    grounding_script = args.pipeline_root / GROUNDING_SCRIPT.format(dataset=args.dataset)
    proposal_model = args.proposal_model or "<PROPOSAL_MODEL>"
    detector_model = args.detector_model or "<DETECTOR_MODEL>"

    proposal_command = [
        args.python_bin,
        str(proposal_script),
        "--output-dir",
        str(proposal_dir),
        "--workers",
        str(args.proposal_workers),
        "--model",
        proposal_model,
        "--shard-size",
        str(args.shard_size),
    ]
    proposal_command.extend(
        ["--datasets", args.dataset, "--raw-pic-root", str(args.raw_pic_root)]
    )
    if args.no_resume:
        proposal_command.append("--no-resume")

    grounding_command = [
        args.python_bin,
        str(grounding_script),
        "--proposal_shard_glob",
        str(proposal_dir / "shard-*.jsonl"),
        "--raw_pic_root",
        str(args.raw_pic_root),
        "--bbox_output_root",
        str(bbox_dir),
        "--viz_output_root",
        str(bbox_dir / "viz"),
        "--episode_label_output_json",
        str(bbox_dir / "episode_labels_rule_merged.json"),
        "--model_path",
        detector_model,
        "--ecot_root",
        str(args.ecot_root),
        "--num_workers",
        str(args.grounding_workers),
        "--write_ecot",
        "--done_file",
        str(bbox_dir / "grounding.done"),
        "--error_file",
        str(bbox_dir / "grounding.errors.jsonl"),
    ]
    if args.gpus:
        grounding_command.extend(["--gpus", args.gpus])
    if args.max_episodes is not None:
        grounding_command.extend(["--max_episodes", str(args.max_episodes)])
    if args.no_viz:
        grounding_command.append("--no_viz")

    _print_command("proposal", proposal_command)
    _print_command("grounding + ECOT writeback", grounding_command)
    if args.dry_run:
        return
    proposal_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    if not args.proposal_model:
        raise SystemExit("set --proposal-model or use --dry-run")
    if not args.detector_model:
        raise SystemExit("set --detector-model or use --dry-run")
    if not proposal_script.is_file():
        raise SystemExit(f"proposal script not found: {proposal_script}")
    if not grounding_script.is_file():
        raise SystemExit(f"grounding script not found: {grounding_script}")

    environment = os.environ.copy()
    proposal_utils = args.pipeline_root / "1_vlm_proposal" / "utils"
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(proposal_utils) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    subprocess.run(proposal_command, check=True, env=environment)
    subprocess.run(grounding_command, check=True, env=environment)


if __name__ == "__main__":
    main()
